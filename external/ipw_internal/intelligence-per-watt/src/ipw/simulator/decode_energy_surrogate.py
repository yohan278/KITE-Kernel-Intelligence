"""Decode Energy Surrogate Model: Grid Parameters → Per-Query Decode Energy.

Builds surrogate models predicting per-query decode energy (J) from grid
parameters (model, quantization, sequence lengths, batch size) and model
architecture features, using inference sweep data across all batch sizes.

Decode is fundamentally different from prefill:
  - Autoregressive: generates seq_out tokens one at a time
  - Memory-bandwidth bound at low batch sizes (loads model weights each step)
  - Transitions to compute-bound at high batch sizes
  - KV cache grows with (seq_in + seq_out), affecting memory traffic
  - Dominates total inference energy (10-100x prefill)

Compares five approaches:
  A-Ridge:    Log-linear Ridge (interpretable power-law exponents)
  B-GBR:     GradientBoostingRegressor (full engineered features)
  C-Roofline: Roofline-residual model (predict ratio to physics baseline)
  D-ScaleInv: Scale-invariant features only (normalized/ratio features)
  E-PowerLaw: Direct OLS power-law fit on raw log features

Evaluated with 10-fold CV and leave-one-model-out CV to assess
generalization to unseen model sizes.

Usage:
    python -m ipw.simulator.decode_energy_surrogate /path/to/data/
    python -m ipw.simulator.decode_energy_surrogate /path/to/data/ --output-dir ./out
    python -m ipw.simulator.decode_energy_surrogate /path/to/data/ --exclude-models 32B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Qwen3 architecture configs from HuggingFace config.json files.
QWEN3_ARCHITECTURES: dict[str, dict[str, int]] = {
    "Qwen/Qwen3-0.6B": {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "vocab_size": 151936,
    },
    "Qwen/Qwen3-4B": {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9216,
        "vocab_size": 151936,
    },
    "Qwen/Qwen3-14B": {
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 17408,
        "vocab_size": 151936,
    },
    "Qwen/Qwen3-32B": {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 25600,
        "vocab_size": 151936,
    },
}

# Hardware specs for roofline model.
HARDWARE_SPECS = {
    "A100SXM4": {
        "peak_fp16_tflops": 312.0,
        "peak_fp8_tflops": 312.0,  # A100 has no FP8 tensor cores; use FP16 rate
        "hbm_bw_gb_s": 2039.0,
        "tdp_w": 400,
        "memory_gb": 80,
    },
    "H100": {
        "peak_fp16_tflops": 989.4,
        "peak_fp8_tflops": 1978.9,
        "hbm_bw_gb_s": 3352.0,
        "tdp_w": 700,
        "memory_gb": 80,
    },
}

# Roofline calibration constants (from inference_model.py).
ETA_DECODE = 0.5    # decode compute efficiency
ALPHA = 0.65        # power fraction of TDP

# Features for the log-linear Ridge model (interpretable power-law exponents).
LOG_LINEAR_FEATURES = [
    "log_model_size_bytes",
    "log_total_decode_flops",
    "log_seq_out",
    "log_seq_in",
    "log_batch_size",
    "bytes_per_param",
]

# Features for GradientBoostingRegressor (full engineered set).
GBR_FEATURES = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "seq_in",
    "seq_out",
    "batch_size",
    "bytes_per_param",
    "head_dim",
    "gqa_ratio",
    "params_per_layer",
    "total_params",
    "model_size_bytes",
    "total_decode_flops",
    "decode_attention_flops",
    "kv_cache_bytes_full",
    "decode_arithmetic_intensity",
    "mem_bound_time_s",
    "compute_bound_time_s",
]

# Features for roofline-residual model (Approach C).
ROOFLINE_RESIDUAL_FEATURES = [
    "log_seq_out",
    "log_seq_in",
    "log_batch_size",
    "bytes_per_param",
    "decode_compute_memory_ratio",
    "kv_to_model_ratio",
]

# Scale-invariant features (Approach D).
SCALE_INVARIANT_FEATURES = [
    "log_seq_out",
    "log_seq_in",
    "log_batch_size",
    "bytes_per_param",
    "decode_compute_memory_ratio",
    "kv_to_model_ratio",
    "model_memory_fraction",
    "context_length_ratio",
]

# Power-law OLS features (Approach E).
POWER_LAW_FEATURES = [
    "log_total_params",
    "log_seq_out",
    "log_seq_in",
    "log_batch_size",
    "quant_binary",
]

# Regex for parsing batch/sequence config from directory names.
_CONFIG_RE = re.compile(r"B(\d+)_Sin(\d+)_Sout(\d+)")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Parse summary.json files, extracting decode phase energy.

    Expected directory layout:
        data_dir/<model>/<quant>/B<N>_Sin<S>_Sout<M>/<profile_dir>/summary.json
    """
    data_dir = Path(data_dir)
    summary_files = sorted(data_dir.glob("*/*/*/*/summary.json"))

    rows: list[dict] = []
    skipped = 0
    hw_label = None

    for summary_path in summary_files:
        try:
            with open(summary_path) as f:
                summary = json.load(f)

            config_dir = summary_path.parent.parent.name
            m = _CONFIG_RE.match(config_dir)
            if m is None:
                skipped += 1
                continue

            batch_size = int(m.group(1))
            seq_in = int(m.group(2))
            seq_out = int(m.group(3))

            quant_dir = summary_path.parent.parent.parent.name
            if quant_dir not in ("fp16", "fp8"):
                skipped += 1
                continue

            model = summary["model"]
            total_queries = summary["total_queries"]
            phase = summary.get("phase_summary", {})
            decode = phase.get("decode", {})

            if not decode or total_queries <= 0:
                skipped += 1
                continue

            if hw_label is None:
                hw_label = summary.get("hardware_label", "unknown")

            decode_energy_j = decode["total_energy_j"] / total_queries

            rows.append({
                "model": model,
                "quant": quant_dir,
                "batch_size": batch_size,
                "seq_in": seq_in,
                "seq_out": seq_out,
                "total_queries": total_queries,
                "decode_energy_j": decode_energy_j,
                "decode_mean_power_w": decode.get("mean_power_w", np.nan),
                "decode_mean_duration_ms": decode.get("mean_duration_ms", np.nan),
                "decode_mean_energy_per_output_token_j": decode.get(
                    "mean_energy_per_output_token_j", np.nan
                ),
                "hardware_label": summary.get("hardware_label", "unknown"),
                "summary_path": str(summary_path),
            })

        except (json.JSONDecodeError, KeyError, TypeError):
            skipped += 1
            continue

    df = pd.DataFrame(rows)

    # Drop rows with zero or negative energy (measurement artifacts)
    bad_mask = df["decode_energy_j"] <= 0
    if bad_mask.any():
        df = df[~bad_mask].reset_index(drop=True)
        print(f"  Dropped {bad_mask.sum()} rows with zero/negative decode energy")

    print(f"\n=== Data Loading (Decode Phase) ===")
    print(f"  Loaded {len(df)} configs from {data_dir}")
    print(f"  Hardware: {hw_label}")
    if skipped:
        print(f"  Skipped {skipped} files (missing/malformed data)")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Quants: {sorted(df['quant'].unique())}")
    print(f"  Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"  Seq in: {sorted(df['seq_in'].unique())}")
    print(f"  Seq out: {sorted(df['seq_out'].unique())}")
    print(f"  Decode energy range: [{df['decode_energy_j'].min():.4f}, "
          f"{df['decode_energy_j'].max():.4f}] J")

    return df


# ---------------------------------------------------------------------------
# Architecture Features
# ---------------------------------------------------------------------------


def add_architecture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Qwen3 architecture parameters onto DataFrame by model column."""
    arch_df = pd.DataFrame.from_dict(QWEN3_ARCHITECTURES, orient="index")
    arch_df.index.name = "model"
    arch_df = arch_df.reset_index()

    n_before = len(df)
    df = df.merge(arch_df, on="model", how="left")

    missing = df["hidden_size"].isna().sum()
    if missing:
        print(f"  WARNING: {missing}/{n_before} rows missing architecture features")

    return df


# ---------------------------------------------------------------------------
# Derived Features
# ---------------------------------------------------------------------------


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for decode energy prediction."""
    df = df.copy()

    # Quantization
    df["bytes_per_param"] = df["quant"].map({"fp16": 2.0, "fp8": 1.0})
    df["quant_binary"] = df["quant"].map({"fp16": 0.0, "fp8": 1.0})

    # Attention geometry
    df["head_dim"] = df["hidden_size"] // df["num_attention_heads"]
    df["gqa_ratio"] = df["num_attention_heads"] / df["num_key_value_heads"]

    # Parameter counts per layer
    kv_proj_size = df["num_key_value_heads"] * df["head_dim"]
    attn_params = df["hidden_size"] * (
        df["hidden_size"] + 2 * kv_proj_size + df["hidden_size"]
    )
    mlp_params = 3 * df["hidden_size"] * df["intermediate_size"]
    df["params_per_layer"] = attn_params + mlp_params

    # Total params (layers + embedding)
    df["total_params"] = (
        df["num_hidden_layers"] * df["params_per_layer"]
        + df["vocab_size"] * df["hidden_size"]
    )
    df["model_size_bytes"] = df["total_params"] * df["bytes_per_param"]

    # --- Decode-specific FLOPs ---
    # Each output token: one forward pass = 2 * total_params FLOPs
    # Total decode FLOPs = 2 * total_params * seq_out * batch_size
    df["total_decode_flops"] = (
        2.0 * df["total_params"] * df["seq_out"] * df["batch_size"]
    )

    # Decode attention FLOPs: each generated token t attends to (seq_in + t) prior tokens
    # Sum over t=1..seq_out: sum of (seq_in + t) = seq_out*seq_in + seq_out*(seq_out+1)/2
    # Per layer per head: 2 * head_dim * (above) for QK^T + softmax@V
    avg_context_length = df["seq_in"] + (df["seq_out"].astype(float) + 1) / 2
    df["decode_attention_flops"] = (
        df["num_hidden_layers"]
        * 2
        * df["num_attention_heads"]
        * df["head_dim"]
        * df["seq_out"]
        * avg_context_length
        * df["batch_size"]
    )

    # KV cache at full context (seq_in + seq_out) — peak memory
    df["kv_cache_bytes_full"] = (
        2  # K + V
        * df["num_key_value_heads"]
        * df["head_dim"]
        * (df["seq_in"] + df["seq_out"])
        * df["bytes_per_param"]
        * df["num_hidden_layers"]
        * df["batch_size"]
    )

    # Decode arithmetic intensity (much lower than prefill — memory bound)
    # Per step: 2*total_params FLOPs / (model_size_bytes + kv_cache_bytes_per_step)
    # Using average KV cache size over decode steps
    avg_kv_per_step = (
        2
        * df["num_key_value_heads"]
        * df["head_dim"]
        * avg_context_length
        * df["bytes_per_param"]
        * df["num_hidden_layers"]
        * df["batch_size"]
    )
    flops_per_step = 2.0 * df["total_params"] * df["batch_size"]
    bytes_per_step = df["model_size_bytes"] + avg_kv_per_step
    df["decode_arithmetic_intensity"] = flops_per_step / bytes_per_step

    # --- Roofline features ---
    hw_label = df["hardware_label"].iloc[0] if "hardware_label" in df.columns else "H100"
    hw = HARDWARE_SPECS.get(hw_label, HARDWARE_SPECS["H100"])

    df["peak_tflops"] = df["quant"].map({
        "fp16": hw["peak_fp16_tflops"],
        "fp8": hw["peak_fp8_tflops"],
    })

    # Memory-bound time: each step loads model weights + KV cache
    # Total memory loaded over seq_out steps ≈ seq_out * (model_bytes + avg_kv_per_step)
    total_mem_loaded = df["seq_out"] * bytes_per_step
    df["mem_bound_time_s"] = total_mem_loaded / (hw["hbm_bw_gb_s"] * 1e9)

    # Compute-bound time: total_decode_flops / peak_compute
    df["compute_bound_time_s"] = (
        df["total_decode_flops"] / (df["peak_tflops"] * 1e12 * ETA_DECODE)
    )

    # Roofline time = max(memory, compute) — this captures the transition
    df["roofline_time_s"] = np.maximum(
        df["mem_bound_time_s"], df["compute_bound_time_s"]
    )
    df["roofline_energy_j"] = df["roofline_time_s"] * ALPHA * hw["tdp_w"]
    df["energy_ratio"] = df["decode_energy_j"] / df["roofline_energy_j"]

    # Compute/memory ratio for decode (< 1 = memory bound, > 1 = compute bound)
    df["decode_compute_memory_ratio"] = (
        df["compute_bound_time_s"] / df["mem_bound_time_s"]
    )

    # Scale-invariant features
    df["kv_to_model_ratio"] = df["kv_cache_bytes_full"] / df["model_size_bytes"]
    df["model_memory_fraction"] = df["model_size_bytes"] / (hw["memory_gb"] * 1e9)
    df["context_length_ratio"] = df["seq_out"].astype(float) / df["seq_in"].astype(float)

    # Log features
    df["log_decode_energy_j"] = np.log(df["decode_energy_j"])
    df["log_model_size_bytes"] = np.log(df["model_size_bytes"].astype(float))
    df["log_total_decode_flops"] = np.log(df["total_decode_flops"].astype(float))
    df["log_seq_in"] = np.log(df["seq_in"].astype(float))
    df["log_seq_out"] = np.log(df["seq_out"].astype(float))
    df["log_batch_size"] = np.log(df["batch_size"].astype(float))
    df["log_total_params"] = np.log(df["total_params"].astype(float))
    df["log_energy_ratio"] = np.log(df["energy_ratio"].clip(lower=1e-10))

    return df


# ---------------------------------------------------------------------------
# EDA Plots
# ---------------------------------------------------------------------------


def generate_eda_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Generate EDA plots for decode energy."""
    output_dir = Path(output_dir)

    df = df.copy()
    df["model_short"] = df["model"].str.replace("Qwen/Qwen3-", "Qwen3-")
    hw = df["hardware_label"].iloc[0] if "hardware_label" in df.columns else "GPU"

    palette = sns.color_palette("tab10", df["model_short"].nunique())
    batch_sizes = sorted(df["batch_size"].unique())

    # --- Plot 1: Decode energy vs seq_out, one panel per batch size ---
    n_panels = min(len(batch_sizes), 5)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    for bi, bs in enumerate(batch_sizes[:5]):
        ax = axes[0, bi]
        sub_bs = df[df["batch_size"] == bs]
        for i, (model, grp) in enumerate(sub_bs.groupby("model_short")):
            for quant, sub in grp.groupby("quant"):
                marker = "o" if quant == "fp16" else "^"
                ax.scatter(
                    sub["seq_out"], sub["decode_energy_j"],
                    label=f"{model} ({quant})" if bi == 0 else None,
                    marker=marker, s=30, alpha=0.8, color=palette[i],
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Output Sequence Length")
        ax.set_ylabel("Decode Energy (J)" if bi == 0 else "")
        ax.set_title(f"B={bs}")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=6, ncol=1)
    fig.suptitle(f"Decode Energy vs Output Sequence Length ({hw})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "eda_decode_energy_vs_seq_out.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_decode_energy_vs_seq_out.png")

    # --- Plot 2: Decode energy vs batch size, one panel per model ---
    models = sorted(df["model_short"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0, mi]
        sub_m = df[df["model_short"] == model]
        for seq_out in sorted(sub_m["seq_out"].unique()):
            sub_s = sub_m[sub_m["seq_out"] == seq_out]
            for quant, sub in sub_s.groupby("quant"):
                marker = "o" if quant == "fp16" else "^"
                lbl = f"Sout={seq_out} ({quant})" if mi == 0 else None
                ax.scatter(sub["batch_size"], sub["decode_energy_j"],
                           label=lbl, marker=marker, s=30, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Decode Energy (J)" if mi == 0 else "")
        ax.set_title(model)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.suptitle(f"Decode Energy vs Batch Size ({hw})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "eda_decode_energy_vs_batch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_decode_energy_vs_batch.png")

    # --- Plot 3: Energy per output token vs model size, colored by batch ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    e_per_tok = df["decode_energy_j"] / df["seq_out"]
    sc = ax.scatter(
        df["model_size_bytes"] / 1e9, e_per_tok,
        c=np.log2(df["batch_size"]), cmap="coolwarm", s=20, alpha=0.6,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Model Size (GB)")
    ax.set_ylabel("Energy per Output Token (J/tok)")
    ax.set_title("Energy Efficiency: J per Output Token")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log₂(batch_size)")
    ax.grid(True, alpha=0.3)

    # Energy per token vs batch size, colored by model
    ax = axes[1]
    for i, (model, grp) in enumerate(df.groupby("model_short")):
        e_tok = grp["decode_energy_j"] / grp["seq_out"]
        ax.scatter(grp["batch_size"], e_tok,
                   label=model, s=20, alpha=0.5, color=palette[i])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Energy per Output Token (J/tok)")
    ax.set_title("Batching Efficiency")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Decode Energy Efficiency ({hw})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "eda_decode_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_decode_efficiency.png")

    # --- Plot 4: Roofline residual ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for i, (model, grp) in enumerate(df.groupby("model_short")):
        ax.scatter(grp["roofline_energy_j"], grp["decode_energy_j"],
                   label=model, s=20, alpha=0.6, color=palette[i])
    lo = min(df["roofline_energy_j"].min(), df["decode_energy_j"].min()) * 0.5
    hi = max(df["roofline_energy_j"].max(), df["decode_energy_j"].max()) * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Roofline Predicted Energy (J)")
    ax.set_ylabel("Actual Decode Energy (J)")
    ax.set_title("Actual vs Roofline Energy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, (model, grp) in enumerate(df.groupby("model_short")):
        ax.scatter(grp["batch_size"], grp["energy_ratio"],
                   label=model, s=20, alpha=0.6, color=palette[i])
    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Actual / Roofline Energy Ratio")
    ax.set_title("Energy Ratio vs Batch Size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Decode Roofline Residual Analysis ({hw})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "eda_decode_roofline_residual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_decode_roofline_residual.png")

    # --- Plot 5: Correlation heatmap ---
    numeric_cols = [
        "batch_size", "seq_in", "seq_out", "bytes_per_param",
        "total_params", "model_size_bytes",
        "total_decode_flops", "decode_attention_flops",
        "kv_cache_bytes_full", "decode_arithmetic_intensity",
        "mem_bound_time_s", "compute_bound_time_s",
        "energy_ratio", "decode_energy_j",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Decode Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_decode_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_decode_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# CV Helpers
# ---------------------------------------------------------------------------


def _cv_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    *,
    collect_preds: bool = False,
) -> dict:
    """Run cross-validation and return metric dict (on log scale)."""
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)

        y_te_orig = np.exp(y_te)
        y_hat_orig = np.exp(y_hat)

        maes.append(mean_absolute_error(y_te_orig, y_hat_orig))
        r2s.append(r2_score(y_te_orig, y_hat_orig))
        mapes.append(mean_absolute_percentage_error(y_te_orig, y_hat_orig))

        if collect_preds:
            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)

    result = {
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        "mape_mean": np.mean(mapes), "mape_std": np.std(mapes),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


def _cv_evaluate_roofline_residual(
    model,
    X: np.ndarray,
    y_log_ratio: np.ndarray,
    roofline_energy: np.ndarray,
    actual_energy: np.ndarray,
    kf: KFold,
    *,
    collect_preds: bool = False,
) -> dict:
    """CV for roofline-residual approach."""
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_log_ratio[train_idx]

        model.fit(X_tr, y_tr)
        y_hat_log_ratio = model.predict(X_te)

        y_hat_orig = np.exp(y_hat_log_ratio) * roofline_energy[test_idx]
        y_te_orig = actual_energy[test_idx]

        maes.append(mean_absolute_error(y_te_orig, y_hat_orig))
        r2s.append(r2_score(y_te_orig, y_hat_orig))
        mapes.append(mean_absolute_percentage_error(y_te_orig, y_hat_orig))

        if collect_preds:
            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)

    result = {
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        "mape_mean": np.mean(mapes), "mape_std": np.std(mapes),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


# ---------------------------------------------------------------------------
# Train Models (10-fold CV)
# ---------------------------------------------------------------------------


def train_models(df: pd.DataFrame) -> dict:
    """Fit all five approaches, evaluate with 10-fold CV."""
    print(f"\n=== 10-Fold Cross-Validation ===")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results: dict = {}

    y = df["log_decode_energy_j"].values

    # --- Approach A: Log-linear Ridge ---
    avail_ll = [f for f in LOG_LINEAR_FEATURES if f in df.columns]
    X_ridge = df[avail_ll].values
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    print(f"  A-Ridge ({len(avail_ll)} features)...", end=" ", flush=True)
    ridge_cv = _cv_evaluate(ridge_pipe, X_ridge, y, kf, collect_preds=True)
    print(f"R²={ridge_cv['r2_mean']:.4f}±{ridge_cv['r2_std']:.4f}  "
          f"MAPE={ridge_cv['mape_mean']:.4f}±{ridge_cv['mape_std']:.4f}")
    ridge_pipe.fit(X_ridge, y)
    results["A-Ridge"] = {
        "model": ridge_pipe, "features": avail_ll,
        "cv": ridge_cv, "target": "log_energy",
    }

    # --- Approach B: GBR ---
    avail_gbr = [f for f in GBR_FEATURES if f in df.columns]
    X_gbr = df[avail_gbr].values
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        learning_rate=0.1, random_state=42,
    )
    print(f"  B-GBR ({len(avail_gbr)} features)...", end=" ", flush=True)
    gbr_cv = _cv_evaluate(gbr, X_gbr, y, kf, collect_preds=True)
    print(f"R²={gbr_cv['r2_mean']:.4f}±{gbr_cv['r2_std']:.4f}  "
          f"MAPE={gbr_cv['mape_mean']:.4f}±{gbr_cv['mape_std']:.4f}")
    gbr.fit(X_gbr, y)
    results["B-GBR"] = {
        "model": gbr, "features": avail_gbr,
        "cv": gbr_cv, "target": "log_energy",
    }

    # --- Approach C: Roofline-Residual ---
    avail_roof = [f for f in ROOFLINE_RESIDUAL_FEATURES if f in df.columns]
    X_roof = df[avail_roof].values
    y_log_ratio = df["log_energy_ratio"].values
    roofline_e = df["roofline_energy_j"].values
    actual_e = df["decode_energy_j"].values

    roof_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    print(f"  C-Roofline ({len(avail_roof)} features)...", end=" ", flush=True)
    roof_cv = _cv_evaluate_roofline_residual(
        roof_pipe, X_roof, y_log_ratio, roofline_e, actual_e, kf, collect_preds=True,
    )
    print(f"R²={roof_cv['r2_mean']:.4f}±{roof_cv['r2_std']:.4f}  "
          f"MAPE={roof_cv['mape_mean']:.4f}±{roof_cv['mape_std']:.4f}")
    roof_pipe.fit(X_roof, y_log_ratio)
    results["C-Roofline"] = {
        "model": roof_pipe, "features": avail_roof,
        "cv": roof_cv, "target": "log_ratio",
    }

    # --- Approach D: Scale-Invariant ---
    avail_si = [f for f in SCALE_INVARIANT_FEATURES if f in df.columns]
    X_si = df[avail_si].values
    si_gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        learning_rate=0.1, random_state=42,
    )
    print(f"  D-ScaleInv ({len(avail_si)} features)...", end=" ", flush=True)
    si_cv = _cv_evaluate(si_gbr, X_si, y, kf, collect_preds=True)
    print(f"R²={si_cv['r2_mean']:.4f}±{si_cv['r2_std']:.4f}  "
          f"MAPE={si_cv['mape_mean']:.4f}±{si_cv['mape_std']:.4f}")
    si_gbr.fit(X_si, y)
    results["D-ScaleInv"] = {
        "model": si_gbr, "features": avail_si,
        "cv": si_cv, "target": "log_energy",
    }

    # --- Approach E: Power-Law OLS ---
    avail_pl = [f for f in POWER_LAW_FEATURES if f in df.columns]
    X_pl = df[avail_pl].values
    ols = LinearRegression()
    print(f"  E-PowerLaw ({len(avail_pl)} features)...", end=" ", flush=True)
    pl_cv = _cv_evaluate(ols, X_pl, y, kf, collect_preds=True)
    print(f"R²={pl_cv['r2_mean']:.4f}±{pl_cv['r2_std']:.4f}  "
          f"MAPE={pl_cv['mape_mean']:.4f}±{pl_cv['mape_std']:.4f}")
    ols.fit(X_pl, y)
    results["E-PowerLaw"] = {
        "model": ols, "features": avail_pl,
        "cv": pl_cv, "target": "log_energy",
    }

    # For backward compat with feature_importance()
    results["ridge_model"] = ridge_pipe
    results["ridge_features"] = avail_ll
    results["ridge_cv"] = ridge_cv
    results["gbr_model"] = gbr
    results["gbr_features"] = avail_gbr
    results["gbr_cv"] = gbr_cv

    return results


# ---------------------------------------------------------------------------
# Leave-One-Model-Out CV
# ---------------------------------------------------------------------------


def leave_one_model_out_cv(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """LOMO CV: train on N-1 models, predict held-out model. All 5 approaches."""
    output_dir = Path(output_dir)

    print(f"\n=== Leave-One-Model-Out Cross-Validation ===")

    y = df["log_decode_energy_j"].values
    groups = df["model"].values
    roofline_e = df["roofline_energy_j"].values
    actual_e = df["decode_energy_j"].values
    y_log_ratio = df["log_energy_ratio"].values

    avail_gbr = [f for f in GBR_FEATURES if f in df.columns]
    avail_roof = [f for f in ROOFLINE_RESIDUAL_FEATURES if f in df.columns]
    avail_si = [f for f in SCALE_INVARIANT_FEATURES if f in df.columns]
    avail_pl = [f for f in POWER_LAW_FEATURES if f in df.columns]
    avail_ll = [f for f in LOG_LINEAR_FEATURES if f in df.columns]

    approach_configs = {
        "A-Ridge": {
            "features": avail_ll, "target": "log_energy",
            "make_estimator": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]),
        },
        "B-GBR": {
            "features": avail_gbr, "target": "log_energy",
            "make_estimator": lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=5,
                learning_rate=0.1, random_state=42,
            ),
        },
        "C-Roofline": {
            "features": avail_roof, "target": "log_ratio",
            "make_estimator": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]),
        },
        "D-ScaleInv": {
            "features": avail_si, "target": "log_energy",
            "make_estimator": lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=5,
                learning_rate=0.1, random_state=42,
            ),
        },
        "E-PowerLaw": {
            "features": avail_pl, "target": "log_energy",
            "make_estimator": lambda: LinearRegression(),
        },
    }

    logo = LeaveOneGroupOut()
    lomo_results: list[dict] = []

    for approach_name, cfg in approach_configs.items():
        X = df[cfg["features"]].values
        target = y_log_ratio if cfg["target"] == "log_ratio" else y
        all_true, all_pred, all_models = [], [], []

        for train_idx, test_idx in logo.split(X, target, groups):
            held_out_model = groups[test_idx[0]]
            n_test = len(test_idx)

            estimator = cfg["make_estimator"]()
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = target[train_idx]

            estimator.fit(X_tr, y_tr)
            y_hat_raw = estimator.predict(X_te)

            if cfg["target"] == "log_ratio":
                y_hat_orig = np.exp(y_hat_raw) * roofline_e[test_idx]
            else:
                y_hat_orig = np.exp(y_hat_raw)
            y_te_orig = actual_e[test_idx]

            mape = mean_absolute_percentage_error(y_te_orig, y_hat_orig)
            r2 = r2_score(y_te_orig, y_hat_orig)

            lomo_results.append({
                "estimator": approach_name,
                "held_out": held_out_model,
                "n_test": n_test,
                "mape": mape,
                "r2": r2,
                "mae": mean_absolute_error(y_te_orig, y_hat_orig),
            })

            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)
            all_models.extend([held_out_model] * n_test)

            print(
                f"  {approach_name:12s} held-out={held_out_model:20s}  "
                f"n={n_test:3d}  MAPE={mape:.4f}  R²={r2:.4f}"
            )

        # Residual plot
        fig, ax = plt.subplots(figsize=(8, 6))
        all_true_arr = np.array(all_true)
        all_pred_arr = np.array(all_pred)
        models_arr = np.array(all_models)

        for model_label in sorted(set(all_models)):
            mask = models_arr == model_label
            short = model_label.replace("Qwen/Qwen3-", "Qwen3-")
            ax.scatter(all_true_arr[mask], all_pred_arr[mask],
                       label=short, s=20, alpha=0.7)

        lo = min(all_true_arr.min(), all_pred_arr.min()) * 0.5
        hi = max(all_true_arr.max(), all_pred_arr.max()) * 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
        ax.set_xlabel("Actual Decode Energy (J)")
        ax.set_ylabel("Predicted Decode Energy (J)")
        ax.set_title(f"LOMO CV: {approach_name}")
        ax.legend(fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"lomo_decode_{approach_name.lower().replace('-', '_')}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    lomo_df = pd.DataFrame(lomo_results)

    # --- Comparison table ---
    print(f"\n  {'='*72}")
    print(f"  LOMO CV Comparison — Decode Energy (all approaches)")
    print(f"  {'='*72}")
    print(f"  {'Approach':<14s} {'Mean MAPE':>10s} {'Mean R²':>10s} {'Mean MAE':>10s} "
          f"{'Folds<0.30':>11s}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")
    for est in approach_configs:
        sub = lomo_df[lomo_df["estimator"] == est]
        print(f"  {est:<14s} {sub['mape'].mean():>10.4f} {sub['r2'].mean():>10.4f} "
              f"{sub['mae'].mean():>10.2f} "
              f"{(sub['mape'] < 0.30).sum():>5d}/{len(sub)}")
    print(f"  {'='*72}")

    # Per-fold detail table
    print(f"\n  Per-fold LOMO detail (MAPE):")
    held_out_models = sorted(lomo_df["held_out"].unique())
    header = f"  {'Held-out':<22s}"
    for est in approach_configs:
        header += f"  {est:>12s}"
    print(header)
    print(f"  {'-'*22}" + f"  {'-'*12}" * len(approach_configs))
    for hom in held_out_models:
        short = hom.replace("Qwen/Qwen3-", "")
        row = f"  {short:<22s}"
        for est in approach_configs:
            sub = lomo_df[(lomo_df["estimator"] == est) & (lomo_df["held_out"] == hom)]
            if len(sub) > 0:
                row += f"  {sub['mape'].iloc[0]:>12.4f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    return {"lomo_df": lomo_df}


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def feature_importance(
    model_results: dict,
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """Generate feature importance plots and Ridge coefficient table."""
    output_dir = Path(output_dir)

    ridge_model = model_results["ridge_model"]
    gbr_model = model_results["gbr_model"]
    gbr_features = model_results["gbr_features"]
    ridge_features = model_results["ridge_features"]

    ridge_coefs = ridge_model.named_steps["ridge"].coef_
    ridge_intercept = ridge_model.named_steps["ridge"].intercept_

    coef_df = pd.DataFrame({
        "feature": ridge_features,
        "scaled_coefficient": ridge_coefs,
    }).sort_values("scaled_coefficient", ascending=False, key=abs).reset_index(drop=True)

    print(f"\n=== Ridge Log-Linear Coefficients (Decode) ===")
    print(f"  Intercept: {ridge_intercept:.4f}")
    print(coef_df.to_string(index=False))

    gbr_imp = pd.DataFrame({
        "feature": gbr_features,
        "importance": gbr_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n=== GBR Feature Importances (Decode) ===")
    print(gbr_imp.to_string(index=False))

    # --- Plot: Feature importance comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    top_n = min(15, len(gbr_imp))
    ax.barh(
        gbr_imp["feature"][:top_n][::-1],
        gbr_imp["importance"][:top_n][::-1],
        color=sns.color_palette("viridis", top_n),
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("GBR Feature Importances (Decode)")

    ax = axes[1]
    colors = ["#2ca02c" if c >= 0 else "#d62728" for c in coef_df["scaled_coefficient"]]
    ax.barh(
        coef_df["feature"][::-1],
        coef_df["scaled_coefficient"][::-1],
        color=colors[::-1],
    )
    ax.set_xlabel("Scaled Coefficient")
    ax.set_title("Ridge Log-Linear Coefficients (Decode)")
    ax.axvline(0, color="k", lw=0.8)

    fig.suptitle("Feature Importance Comparison (Decode)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "decode_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved decode_feature_importance.png")

    # --- Partial Dependence Plots ---
    top3_idx = [gbr_features.index(f) for f in gbr_imp["feature"][:3]]
    X_gbr = df[gbr_features].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat_idx in enumerate(top3_idx):
        feat_name = gbr_features[feat_idx]
        PartialDependenceDisplay.from_estimator(
            gbr_model, X_gbr, [feat_idx],
            feature_names=gbr_features, ax=axes[i],
        )
        axes[i].set_title(f"PDP: {feat_name}")

    fig.suptitle("Partial Dependence (GBR Decode, top 3 features)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "decode_partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved decode_partial_dependence.png")

    return {
        "ridge_coefficients": coef_df,
        "gbr_importances": gbr_imp,
    }


# ---------------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    df: pd.DataFrame,
    model_results: dict,
    lomo_results: dict,
    importance_results: dict,
    output_dir: str | Path,
) -> None:
    """Persist CSV, JSON summary, and print findings."""
    output_dir = Path(output_dir)

    csv_cols = [
        "model", "quant", "batch_size", "seq_in", "seq_out", "total_queries",
        "decode_energy_j", "decode_mean_power_w", "decode_mean_duration_ms",
        "decode_mean_energy_per_output_token_j",
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "bytes_per_param", "head_dim", "gqa_ratio", "params_per_layer",
        "total_params", "model_size_bytes",
        "total_decode_flops", "decode_attention_flops",
        "kv_cache_bytes_full", "decode_arithmetic_intensity",
        "mem_bound_time_s", "compute_bound_time_s",
        "roofline_energy_j", "energy_ratio",
        "decode_compute_memory_ratio", "kv_to_model_ratio",
        "model_memory_fraction", "context_length_ratio",
        "log_decode_energy_j", "log_model_size_bytes",
        "log_total_decode_flops", "log_seq_in", "log_seq_out", "log_batch_size",
    ]
    available_cols = [c for c in csv_cols if c in df.columns]
    csv_path = output_dir / "decode_energy_data.csv"
    df[available_cols].to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path} ({len(df)} rows, {len(available_cols)} columns)")

    def _safe_val(v):
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    lomo_df = lomo_results["lomo_df"]

    tenfold_cv = {}
    for name in ["A-Ridge", "B-GBR", "C-Roofline", "D-ScaleInv", "E-PowerLaw"]:
        if name in model_results and "cv" in model_results[name]:
            tenfold_cv[name] = {
                k: _safe_val(v) for k, v in model_results[name]["cv"].items()
                if k not in ("all_true", "all_pred")
            }

    summary = {
        "phase": "decode",
        "n_configs": len(df),
        "models": sorted(df["model"].unique().tolist()),
        "quants": sorted(df["quant"].unique().tolist()),
        "batch_sizes": sorted(df["batch_size"].unique().tolist()),
        "seq_in_lengths": sorted(df["seq_in"].unique().tolist()),
        "seq_out_lengths": sorted(df["seq_out"].unique().tolist()),
        "tenfold_cv": tenfold_cv,
        "lomo_cv": [
            {k: _safe_val(v) for k, v in row.items()}
            for _, row in lomo_df.iterrows()
        ],
        "ridge_coefficients": importance_results["ridge_coefficients"].to_dict(orient="records"),
        "gbr_top_features": importance_results["gbr_importances"].head(5).to_dict(orient="records"),
    }

    json_path = output_dir / "decode_energy_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_safe_val)
    print(f"  Saved {json_path}")

    # --- Print findings summary ---
    print(f"\n{'='*60}")
    print(f"  Decode Energy Surrogate — Summary of Findings")
    print(f"{'='*60}")
    batch_sizes = sorted(df["batch_size"].unique())
    print(f"  Data: {len(df)} configs, {len(df['model'].unique())} models, "
          f"{len(df['quant'].unique())} quants, batch sizes={batch_sizes}")
    print(f"  Seq out: {sorted(df['seq_out'].unique())}")

    print(f"\n  10-fold CV:")
    for name in ["A-Ridge", "B-GBR", "C-Roofline", "D-ScaleInv", "E-PowerLaw"]:
        if name in model_results and "cv" in model_results[name]:
            cv = model_results[name]["cv"]
            print(f"    {name:14s}: R²={cv['r2_mean']:.4f}  MAPE={cv['mape_mean']:.4f}")

    print(f"\n  LOMO CV (generalization to unseen models):")
    for est in ["A-Ridge", "B-GBR", "C-Roofline", "D-ScaleInv", "E-PowerLaw"]:
        sub = lomo_df[lomo_df["estimator"] == est]
        if len(sub) > 0:
            print(f"    {est:14s}: mean MAPE={sub['mape'].mean():.4f}, "
                  f"mean R²={sub['r2'].mean():.4f}")

    print(f"\n  Top GBR features: "
          f"{', '.join(importance_results['gbr_importances']['feature'][:3])}")
    print(f"\n  Ridge coefficients:")
    for _, row in importance_results["ridge_coefficients"].iterrows():
        print(f"    {row['feature']:30s}  coef={row['scaled_coefficient']:.4f}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    exclude_models: list[str] | None = None,
) -> dict:
    """Run the full decode energy surrogate analysis."""
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir.parent / "decode_energy_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Decode Energy Surrogate Model")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    if exclude_models:
        print(f"  Excluding models matching: {exclude_models}")
    print(f"{'='*60}")

    df = load_data(data_dir)
    if len(df) == 0:
        print("ERROR: No data loaded. Check data directory.")
        sys.exit(1)

    if exclude_models:
        for pattern in exclude_models:
            before = len(df)
            df = df[~df["model"].str.lower().str.contains(pattern.lower())]
            dropped = before - len(df)
            if dropped:
                print(f"  Filtered out {dropped} rows matching '{pattern}'")
        print(f"  Remaining: {len(df)} rows, models={sorted(df['model'].unique())}")

    df = add_architecture_features(df)
    df = compute_derived_features(df)

    print(f"\n=== EDA Plots ===")
    generate_eda_plots(df, output_dir)

    model_results = train_models(df)
    lomo_results = leave_one_model_out_cv(df, output_dir)
    importance_results = feature_importance(model_results, df, output_dir)
    save_artifacts(df, model_results, lomo_results, importance_results, output_dir)

    return {
        "df": df,
        "model_results": model_results,
        "lomo_results": lomo_results,
        "importance_results": importance_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Decode energy surrogate model: grid params → per-query decode energy",
    )
    parser.add_argument(
        "data_dir", type=str,
        help="Path to data directory containing model/quant/config subdirs",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output artifacts (default: decode_energy_output/ next to data)",
    )
    parser.add_argument(
        "--exclude-models", type=str, nargs="+", default=None,
        help="Model name substrings to exclude (e.g. 32B 0.6B)",
    )
    args = parser.parse_args(argv)

    run_pipeline(args.data_dir, args.output_dir, args.exclude_models)
    print("\nDone.")


if __name__ == "__main__":
    main()
