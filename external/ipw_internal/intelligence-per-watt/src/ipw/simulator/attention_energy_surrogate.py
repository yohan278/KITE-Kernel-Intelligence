"""Attention Operator Latency Surrogate Model.

Builds surrogate models predicting per-operator latency (time_s) for
attention-related operators from attention.csv profiling data across
models, batch sizes, sequence lengths, and KV cache sizes.

Six operators, each with distinct physics:
  - attention_prefill:  Compute-bound (FLOPs ∝ seq²), no KV cache
  - attention_decode:   Memory-bound (bandwidth-limited), uses KV cache
  - kv_cache_append:    Memory-only (flops=0), bandwidth_gb_s populated
  - kv_cache_evict:     Memory-only (flops=0), bandwidth_gb_s populated
  - mqa_gqa_expansion:  Compute (FLOPs populated), prefill variant
  - sliding_window_attention: Compute-bound (FLOPs ∝ seq²), prefill variant

Models compared (per operator):
  A-Ridge:      Log-linear Ridge on log features
  B-GBR:        GradientBoostingRegressor on full feature set
  C-Roofline:   Roofline-residual (predict log(time/roofline_time))
  D-Poly2Ridge: Degree-2 polynomial in log space with Ridge
  E-PowerLaw:   OLS on raw log features
  F-RF:         RandomForestRegressor

Usage:
    python -m ipw.simulator.attention_energy_surrogate /path/to/profiles/
    python -m ipw.simulator.attention_energy_surrogate /path/to/profiles/ --output-dir ./out
    python -m ipw.simulator.attention_energy_surrogate /path/to/profiles/ --operators attention_prefill attention_decode
    python -m ipw.simulator.attention_energy_surrogate /path/to/profiles/ --exclude-models 30B-A3B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ARCHITECTURES: dict[str, dict[str, int | float]] = {
    "Qwen_Qwen3-0.6B": {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "num_experts": 0,
        "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-1.7B": {
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 6144,
        "vocab_size": 151936,
        "num_experts": 0,
        "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-4B": {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9216,
        "vocab_size": 151936,
        "num_experts": 0,
        "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-8B": {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "num_experts": 0,
        "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-14B": {
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 17408,
        "vocab_size": 151936,
        "num_experts": 0,
        "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-30B-A3B": {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 6144,
        "vocab_size": 151936,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    },
    "Qwen_Qwen3-32B": {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "num_experts": 0,
        "num_experts_per_tok": 0,
    },
    "zai-org_GLM-4.7-Flash": {
        "hidden_size": 2048,
        "num_hidden_layers": 47,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "intermediate_size": 10240,
        "vocab_size": 154880,
        "num_experts": 64,
        "num_experts_per_tok": 4,
    },
}

HARDWARE_SPECS = {
    "A100SXM4": {
        "peak_fp16_tflops": 312.0,
        "hbm_bw_gb_s": 2039.0,
        "tdp_w": 400,
        "memory_gb": 80,
    },
}

# Operator → variant mapping for reference.
OPERATOR_VARIANTS: dict[str, str] = {
    "attention_prefill": "prefill",
    "attention_decode": "decode",
    "kv_cache_append": "decode",
    "kv_cache_evict": "decode",
    "mqa_gqa_expansion": "prefill",
    "sliding_window_attention": "prefill",
}

# Operators where flops=0 (memory-only ops).
MEMORY_ONLY_OPS = {"kv_cache_append", "kv_cache_evict"}

# Operators where kv_cache_size is meaningful.
KV_CACHE_OPS = {"attention_decode", "kv_cache_append", "kv_cache_evict"}

# Feature lists per approach — compute ops (flops > 0, no kv_cache_size).
FEATURES_COMPUTE = {
    "A-Ridge": [
        "log_flops", "log_tokens", "log_hidden_size",
        "log_batch_size", "log_seq_len",
    ],
    "B-GBR": [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "batch_size", "seq_len", "flops",
        "head_dim", "gqa_ratio", "tokens", "is_moe",
        "log_flops", "log_tokens", "log_batch_size", "log_seq_len",
    ],
    "C-Roofline": [
        "log_seq_len", "log_batch_size", "log_tokens",
        "gqa_ratio", "is_moe",
    ],
    "D-Poly2Ridge": [
        "log_flops", "log_tokens", "log_hidden_size",
        "log_batch_size", "log_seq_len",
    ],
    "E-PowerLaw": [
        "log_flops", "log_tokens", "log_hidden_size", "log_seq_len",
    ],
    "F-RF": [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "batch_size", "seq_len", "flops",
        "head_dim", "gqa_ratio", "tokens", "is_moe",
        "log_flops", "log_tokens", "log_batch_size", "log_seq_len",
    ],
}

# Feature lists — decode ops with kv_cache_size (flops > 0).
FEATURES_DECODE = {
    "A-Ridge": [
        "log_flops", "log_tokens", "log_hidden_size",
        "log_batch_size", "log_seq_len", "log_kv_cache_size",
    ],
    "B-GBR": [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "batch_size", "seq_len", "kv_cache_size", "flops",
        "head_dim", "gqa_ratio", "tokens", "is_moe",
        "log_flops", "log_tokens", "log_batch_size", "log_seq_len",
        "log_kv_cache_size",
    ],
    "C-Roofline": [
        "log_seq_len", "log_batch_size", "log_tokens",
        "log_kv_cache_size", "gqa_ratio", "is_moe",
    ],
    "D-Poly2Ridge": [
        "log_flops", "log_tokens", "log_hidden_size",
        "log_batch_size", "log_seq_len", "log_kv_cache_size",
    ],
    "E-PowerLaw": [
        "log_flops", "log_tokens", "log_hidden_size",
        "log_seq_len", "log_kv_cache_size",
    ],
    "F-RF": [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "batch_size", "seq_len", "kv_cache_size", "flops",
        "head_dim", "gqa_ratio", "tokens", "is_moe",
        "log_flops", "log_tokens", "log_batch_size", "log_seq_len",
        "log_kv_cache_size",
    ],
}

# Feature lists — memory-only ops (flops=0, bandwidth populated).
FEATURES_MEMORY = {
    "A-Ridge": [
        "log_bandwidth_gb_s", "log_tokens", "log_hidden_size",
        "log_batch_size", "log_seq_len", "log_kv_cache_size",
    ],
    "B-GBR": [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "batch_size", "seq_len", "kv_cache_size", "bandwidth_gb_s",
        "head_dim", "gqa_ratio", "tokens", "is_moe",
        "log_bandwidth_gb_s", "log_tokens", "log_batch_size", "log_seq_len",
        "log_kv_cache_size",
    ],
    "D-Poly2Ridge": [
        "log_bandwidth_gb_s", "log_tokens", "log_hidden_size",
        "log_batch_size", "log_seq_len", "log_kv_cache_size",
    ],
    "E-PowerLaw": [
        "log_bandwidth_gb_s", "log_tokens", "log_hidden_size",
        "log_seq_len", "log_kv_cache_size",
    ],
    "F-RF": [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "batch_size", "seq_len", "kv_cache_size", "bandwidth_gb_s",
        "head_dim", "gqa_ratio", "tokens", "is_moe",
        "log_bandwidth_gb_s", "log_tokens", "log_batch_size", "log_seq_len",
        "log_kv_cache_size",
    ],
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Load attention.csv files from profiles directory.

    Expected layout: data_dir/<model>/<gpu>/<quant>/attention.csv
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*/*/*/attention.csv"))

    frames: list[pd.DataFrame] = []
    skipped = 0

    for csv_path in csv_files:
        try:
            quant = csv_path.parent.name
            gpu = csv_path.parent.parent.name
            model = csv_path.parent.parent.parent.name

            df_file = pd.read_csv(csv_path)
            df_file["model"] = model
            df_file["gpu"] = gpu
            df_file["quant"] = quant
            frames.append(df_file)
        except Exception:
            skipped += 1
            continue

    if not frames:
        print(f"ERROR: No attention.csv files found under {data_dir}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Clean: drop rows with missing or non-positive time_s
    bad_mask = df["time_s"].isna() | (df["time_s"] <= 0)
    if bad_mask.any():
        df = df[~bad_mask].reset_index(drop=True)
        print(f"  Dropped {bad_mask.sum()} rows with missing/non-positive time_s")

    print(f"\n=== Data Loading ===")
    print(f"  Loaded {len(df)} rows from {len(csv_files)} CSV files under {data_dir}")
    if skipped:
        print(f"  Skipped {skipped} files (read errors)")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Operators: {sorted(df['operator_name'].unique())}")
    print(f"  Variants: {sorted(df['variant'].unique())}")
    print(f"  Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"  Seq lengths: {sorted(df['seq_len'].unique())}")
    print(f"  Time range: [{df['time_s'].min():.2e}, {df['time_s'].max():.2e}] s")

    return df


# ---------------------------------------------------------------------------
# Architecture Features
# ---------------------------------------------------------------------------


def add_architecture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge model architecture parameters onto DataFrame by model column."""
    arch_df = pd.DataFrame.from_dict(MODEL_ARCHITECTURES, orient="index")
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
    """Engineer features from architecture params, operator data, and hardware."""
    df = df.copy()

    # Tokens = batch_size * seq_len
    df["tokens"] = df["batch_size"] * df["seq_len"]

    # Attention geometry
    df["head_dim"] = df["hidden_size"] // df["num_attention_heads"]
    df["gqa_ratio"] = df["num_attention_heads"] / df["num_key_value_heads"]

    # MoE indicator
    df["is_moe"] = (df["num_experts"] > 0).astype(float)

    # Fill NaN in bandwidth_gb_s and kv_cache_size
    df["bandwidth_gb_s"] = df["bandwidth_gb_s"].fillna(0.0)
    df["kv_cache_size"] = df["kv_cache_size"].fillna(0.0)

    # Roofline time (for compute ops)
    hw = HARDWARE_SPECS["A100SXM4"]
    peak_flops_per_s = hw["peak_fp16_tflops"] * 1e12
    df["roofline_time_s"] = df["flops"].astype(float) / peak_flops_per_s

    # Log features
    df["log_time_s"] = np.log(df["time_s"].clip(lower=1e-15))
    df["log_flops"] = np.log(df["flops"].astype(float).clip(lower=1.0))
    df["log_tokens"] = np.log(df["tokens"].astype(float).clip(lower=1.0))
    df["log_hidden_size"] = np.log(df["hidden_size"].astype(float))
    df["log_batch_size"] = np.log(df["batch_size"].astype(float).clip(lower=1.0))
    df["log_seq_len"] = np.log(df["seq_len"].astype(float))
    df["log_bandwidth_gb_s"] = np.log(df["bandwidth_gb_s"].clip(lower=1e-10))
    df["log_kv_cache_size"] = np.log(df["kv_cache_size"].clip(lower=1.0))

    # Roofline ratio (only meaningful when flops > 0)
    df["time_roofline_ratio"] = np.where(
        df["roofline_time_s"] > 0,
        df["time_s"] / df["roofline_time_s"],
        np.nan,
    )
    df["log_time_roofline_ratio"] = np.where(
        df["roofline_time_s"] > 0,
        np.log((df["time_s"] / df["roofline_time_s"]).clip(lower=1e-10)),
        np.nan,
    )

    return df


# ---------------------------------------------------------------------------
# EDA Plots
# ---------------------------------------------------------------------------


def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate EDA plots for attention operator latency data."""
    output_dir = Path(output_dir)

    df = df.copy()
    df["model_short"] = df["model"].str.replace("Qwen_Qwen3-", "Q3-").str.replace(
        "zai-org_GLM-4.7-Flash", "GLM-4.7"
    )

    operators = sorted(df["operator_name"].unique())
    model_names = sorted(df["model_short"].unique())
    palette = sns.color_palette("tab10", len(model_names))

    # --- Plot 1: Latency vs seq_len grid (one subplot per operator, B=1) ---
    n_ops = len(operators)
    ncols = 3
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, op in enumerate(operators):
        ax = axes[idx // ncols, idx % ncols]
        sub_op = df[(df["operator_name"] == op) & (df["batch_size"] == 1)]
        for mi, model in enumerate(model_names):
            sub = sub_op[sub_op["model_short"] == model]
            if len(sub) > 0:
                ax.scatter(sub["seq_len"], sub["time_s"], label=model, s=15,
                           alpha=0.7, color=palette[mi])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Seq Length")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{op} ({OPERATOR_VARIANTS.get(op, '?')})", fontsize=10)
        ax.grid(True, alpha=0.3)
    for idx in range(n_ops, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.suptitle("Attention Operator Latency vs Sequence Length (B=1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "eda_attn_latency_vs_seq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_attn_latency_vs_seq.png")

    # --- Plot 2: FLOPs vs latency (compute ops only) ---
    compute_ops = [op for op in operators if op not in MEMORY_ONLY_OPS]
    fig, ax = plt.subplots(figsize=(10, 7))
    op_palette = sns.color_palette("Set2", len(compute_ops))
    for ci, op in enumerate(compute_ops):
        sub = df[(df["operator_name"] == op) & (df["flops"] > 0)]
        if len(sub) > 0:
            ax.scatter(sub["flops"], sub["time_s"], label=op, s=8, alpha=0.4,
                       color=op_palette[ci])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Time (s)")
    ax.set_title("Attention Operator Latency vs FLOPs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_attn_flops_vs_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_attn_flops_vs_latency.png")

    # --- Plot 3: Decode ops — latency vs kv_cache_size ---
    decode_ops = [op for op in operators if op in KV_CACHE_OPS]
    if decode_ops:
        fig, axes = plt.subplots(1, len(decode_ops), figsize=(6 * len(decode_ops), 5),
                                 squeeze=False)
        for di, op in enumerate(decode_ops):
            ax = axes[0, di]
            sub_op = df[(df["operator_name"] == op) & (df["batch_size"] == 1)]
            for mi, model in enumerate(model_names):
                sub = sub_op[sub_op["model_short"] == model]
                if len(sub) > 0:
                    ax.scatter(sub["kv_cache_size"], sub["time_s"], label=model,
                               s=15, alpha=0.7, color=palette[mi])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("KV Cache Size")
            ax.set_ylabel("Time (s)")
            ax.set_title(op)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=5, ncol=2)
        fig.suptitle("Decode Ops: Latency vs KV Cache Size (B=1)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / "eda_attn_decode_vs_kvcache.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved eda_attn_decode_vs_kvcache.png")

    # --- Plot 4: Time breakdown heatmap ---
    sub_b1 = df[df["batch_size"] == 1].copy()
    total_per_config = sub_b1.groupby(["model_short", "seq_len"])["time_s"].transform("sum")
    sub_b1["time_frac"] = sub_b1["time_s"] / total_per_config
    pivot = sub_b1.pivot_table(
        index="operator_name", columns="model_short",
        values="time_frac", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Mean Time Fraction per Attention Op (B=1, avg over seq lengths)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_attn_time_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_attn_time_heatmap.png")


# ---------------------------------------------------------------------------
# Cross-Validation Helpers
# ---------------------------------------------------------------------------


def _cv_evaluate(
    model, X: np.ndarray, y: np.ndarray, kf: KFold,
    *, collect_preds: bool = False,
) -> dict:
    """K-fold CV predicting log_time_s, metrics on original scale."""
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_hat = model.predict(X[test_idx])
        y_te_orig = np.exp(y[test_idx])
        y_hat_orig = np.exp(y_hat)
        maes.append(mean_absolute_error(y_te_orig, y_hat_orig))
        r2s.append(r2_score(y_te_orig, y_hat_orig))
        mapes.append(mean_absolute_percentage_error(y_te_orig, y_hat_orig))
        if collect_preds:
            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)

    result = {
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "mape_mean": float(np.mean(mapes)), "mape_std": float(np.std(mapes)),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


def _cv_evaluate_roofline_residual(
    model, X: np.ndarray, y_log_ratio: np.ndarray,
    roofline_time: np.ndarray, actual_time: np.ndarray, kf: KFold,
    *, collect_preds: bool = False,
) -> dict:
    """CV for roofline-residual: predict log(time/roofline), evaluate on original."""
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y_log_ratio[train_idx])
        y_hat_log_ratio = model.predict(X[test_idx])
        y_hat_orig = np.exp(y_hat_log_ratio) * roofline_time[test_idx]
        y_te_orig = actual_time[test_idx]
        maes.append(mean_absolute_error(y_te_orig, y_hat_orig))
        r2s.append(r2_score(y_te_orig, y_hat_orig))
        mapes.append(mean_absolute_percentage_error(y_te_orig, y_hat_orig))
        if collect_preds:
            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)

    result = {
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "mape_mean": float(np.mean(mapes)), "mape_std": float(np.std(mapes)),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


# ---------------------------------------------------------------------------
# Feature Selection Per Operator
# ---------------------------------------------------------------------------


def _get_feature_dict(op_name: str) -> dict[str, list[str]]:
    """Return the correct feature dict for an operator."""
    if op_name in MEMORY_ONLY_OPS:
        return FEATURES_MEMORY
    if op_name in KV_CACHE_OPS:
        return FEATURES_DECODE
    return FEATURES_COMPUTE


def _make_estimator(approach: str):
    """Create a fresh estimator for the given approach."""
    if approach == "A-Ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if approach == "B-GBR":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            learning_rate=0.1, random_state=42,
        )
    if approach == "C-Roofline":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if approach == "D-Poly2Ridge":
        return Pipeline([
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if approach == "E-PowerLaw":
        return LinearRegression()
    if approach == "F-RF":
        return RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42,
        )
    raise ValueError(f"Unknown approach: {approach}")


# ---------------------------------------------------------------------------
# Per-Operator Model Training
# ---------------------------------------------------------------------------


def train_models_for_operator(
    df_op: pd.DataFrame, op_name: str,
) -> dict:
    """Fit all approaches for one attention operator with adaptive CV."""
    n_samples = len(df_op)
    n_splits = min(10, max(2, n_samples // 3))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y = df_op["log_time_s"].values
    feat_dict = _get_feature_dict(op_name)
    is_memory_only = op_name in MEMORY_ONLY_OPS
    results: dict = {}

    approaches = list(feat_dict.keys())
    # Also add F-RF if present
    if "F-RF" not in approaches:
        approaches.append("F-RF")

    for approach in approaches:
        if approach == "C-Roofline" and is_memory_only:
            continue  # No roofline for memory-only ops

        if approach == "C-Roofline":
            # Roofline residual
            valid_mask = df_op["log_time_roofline_ratio"].notna()
            df_c = df_op[valid_mask]
            if len(df_c) < 6:
                continue
            feat_list = feat_dict.get(approach, [])
            avail = [f for f in feat_list if f in df_c.columns]
            X = df_c[avail].values
            y_log_ratio = df_c["log_time_roofline_ratio"].values
            roofline_t = df_c["roofline_time_s"].values
            actual_t = df_c["time_s"].values
            n_splits_c = min(10, max(2, len(df_c) // 3))
            kf_c = KFold(n_splits=n_splits_c, shuffle=True, random_state=42)

            estimator = _make_estimator(approach)
            cv = _cv_evaluate_roofline_residual(
                estimator, X, y_log_ratio, roofline_t, actual_t, kf_c,
                collect_preds=True,
            )
            estimator_final = _make_estimator(approach)
            estimator_final.fit(X, y_log_ratio)
            results[approach] = {
                "model": estimator_final, "features": avail,
                "cv": cv, "target": "log_ratio",
            }
        else:
            feat_list = feat_dict.get(approach, [])
            avail = [f for f in feat_list if f in df_op.columns]
            X = df_op[avail].values

            estimator = _make_estimator(approach)
            cv = _cv_evaluate(estimator, X, y, kf, collect_preds=True)
            estimator_final = _make_estimator(approach)
            estimator_final.fit(X, y)
            results[approach] = {
                "model": estimator_final, "features": avail,
                "cv": cv, "target": "log_time_s",
            }

    return results


def train_all_operators(df: pd.DataFrame) -> dict[str, dict]:
    """Train surrogate models for all attention operators."""
    print(f"\n=== Per-Operator Cross-Validation ===")

    operators = sorted(df["operator_name"].unique())
    all_results: dict[str, dict] = {}
    approach_names = ["A-Ridge", "B-GBR", "C-Roofline", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]

    print(f"  {'Operator':<30s} {'N':>5s}", end="")
    for a in approach_names:
        print(f"  {a:>12s}", end="")
    print()
    print(f"  {'-'*30} {'-'*5}" + f"  {'-'*12}" * len(approach_names))

    for op in operators:
        df_op = df[df["operator_name"] == op].reset_index(drop=True)
        n = len(df_op)

        if n < 6:
            print(f"  {op:<30s} {n:>5d}  SKIPPED (too few samples)")
            continue

        results = train_models_for_operator(df_op, op)
        all_results[op] = results

        parts = [f"  {op:<30s} {n:>5d}"]
        for approach in approach_names:
            if approach in results:
                mape = results[approach]["cv"]["mape_mean"]
                parts.append(f"  {mape:>11.4f}")
            else:
                parts.append(f"  {'N/A':>12s}")
        print("".join(parts))

    return all_results


# ---------------------------------------------------------------------------
# Leave-One-Model-Out CV
# ---------------------------------------------------------------------------


def leave_one_model_out_cv(
    df: pd.DataFrame, output_dir: str | Path,
) -> dict:
    """LOMO CV per operator per approach."""
    output_dir = Path(output_dir)
    print(f"\n=== Leave-One-Model-Out Cross-Validation ===")

    operators = sorted(df["operator_name"].unique())
    logo = LeaveOneGroupOut()
    lomo_rows: list[dict] = []

    approach_names = ["A-Ridge", "B-GBR", "C-Roofline", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]

    for op in operators:
        df_op = df[df["operator_name"] == op].reset_index(drop=True)
        if len(df_op) < 6 or len(df_op["model"].unique()) < 2:
            continue

        is_memory_only = op in MEMORY_ONLY_OPS
        feat_dict = _get_feature_dict(op)
        y = df_op["log_time_s"].values
        groups = df_op["model"].values

        for approach in approach_names:
            if approach == "C-Roofline" and is_memory_only:
                continue
            if approach not in feat_dict and approach not in ("F-RF",):
                continue

            feat_list = feat_dict.get(approach, [])
            avail = [f for f in feat_list if f in df_op.columns]

            if approach == "C-Roofline":
                valid_mask = df_op["log_time_roofline_ratio"].notna()
                df_c = df_op[valid_mask].reset_index(drop=True)
                if len(df_c) < 6 or len(df_c["model"].unique()) < 2:
                    continue
                X = df_c[avail].values
                target = df_c["log_time_roofline_ratio"].values
                roofline_t = df_c["roofline_time_s"].values
                actual_t = df_c["time_s"].values
                groups_c = df_c["model"].values
            else:
                X = df_op[avail].values
                target = y
                groups_c = groups

            for train_idx, test_idx in logo.split(X, target, groups_c):
                held_out = groups_c[test_idx[0]]
                n_test = len(test_idx)

                estimator = _make_estimator(approach)
                estimator.fit(X[train_idx], target[train_idx])
                y_hat_raw = estimator.predict(X[test_idx])

                if approach == "C-Roofline":
                    y_hat_orig = np.exp(y_hat_raw) * roofline_t[test_idx]
                    y_te_orig = actual_t[test_idx]
                else:
                    y_hat_orig = np.exp(y_hat_raw)
                    y_te_orig = np.exp(target[test_idx])

                mape = mean_absolute_percentage_error(y_te_orig, y_hat_orig)
                r2 = r2_score(y_te_orig, y_hat_orig) if n_test > 1 else float("nan")

                lomo_rows.append({
                    "operator": op,
                    "approach": approach,
                    "held_out": held_out,
                    "n_test": n_test,
                    "mape": mape,
                    "r2": r2,
                    "mae": mean_absolute_error(y_te_orig, y_hat_orig),
                })

    lomo_df = pd.DataFrame(lomo_rows)

    if len(lomo_df) == 0:
        print("  No LOMO results (insufficient data).")
        return {"lomo_df": lomo_df}

    # --- Summary table ---
    print(f"\n  LOMO CV — Mean MAPE per Operator x Approach")
    header = f"  {'Operator':<30s}"
    for a in approach_names:
        header += f"  {a:>12s}"
    print(header)
    print(f"  {'-'*30}" + f"  {'-'*12}" * len(approach_names))

    for op in sorted(lomo_df["operator"].unique()):
        row_str = f"  {op:<30s}"
        for a in approach_names:
            sub = lomo_df[(lomo_df["operator"] == op) & (lomo_df["approach"] == a)]
            if len(sub) > 0:
                row_str += f"  {sub['mape'].mean():>12.4f}"
            else:
                row_str += f"  {'N/A':>12s}"
        print(row_str)

    print(f"\n  LOMO CV — Overall Aggregate")
    print(f"  {'Approach':<14s} {'Mean MAPE':>10s} {'Mean R²':>10s}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    for a in approach_names:
        sub = lomo_df[lomo_df["approach"] == a]
        if len(sub) > 0:
            print(f"  {a:<14s} {sub['mape'].mean():>10.4f} {sub['r2'].mean():>10.4f}")

    # --- Per-model aggregate ---
    print(f"\n  LOMO CV — Mean MAPE by Held-Out Model (B-GBR)")
    gbr_lomo = lomo_df[lomo_df["approach"] == "B-GBR"]
    if len(gbr_lomo) > 0:
        by_model = gbr_lomo.groupby("held_out")["mape"].mean().sort_values()
        for model, mape in by_model.items():
            short = str(model).replace("Qwen_Qwen3-", "Q3-").replace(
                "zai-org_GLM-4.7-Flash", "GLM-4.7"
            )
            print(f"    {short:<20s} {mape:.4f}")

    # --- LOMO residual plots per operator (best approach) ---
    for op in sorted(lomo_df["operator"].unique()):
        sub_op = lomo_df[lomo_df["operator"] == op]
        best_approach = sub_op.groupby("approach")["mape"].mean().idxmin()

        df_op = df[df["operator_name"] == op].reset_index(drop=True)
        feat_dict = _get_feature_dict(op)
        feat_list = feat_dict.get(best_approach, [])
        avail = [f for f in feat_list if f in df_op.columns]

        if best_approach == "C-Roofline":
            valid_mask = df_op["log_time_roofline_ratio"].notna()
            df_plot = df_op[valid_mask].reset_index(drop=True)
            X = df_plot[avail].values
            target = df_plot["log_time_roofline_ratio"].values
            roofline_t = df_plot["roofline_time_s"].values
            actual_t = df_plot["time_s"].values
            groups_plot = df_plot["model"].values
        else:
            df_plot = df_op
            X = df_plot[avail].values
            target = df_plot["log_time_s"].values
            groups_plot = df_plot["model"].values

        all_true_arr, all_pred_arr, all_models_arr = [], [], []
        for train_idx, test_idx in logo.split(X, target, groups_plot):
            held_out = groups_plot[test_idx[0]]
            est = _make_estimator(best_approach)
            est.fit(X[train_idx], target[train_idx])
            y_hat_raw = est.predict(X[test_idx])

            if best_approach == "C-Roofline":
                y_hat_orig = np.exp(y_hat_raw) * roofline_t[test_idx]
                y_te_orig = actual_t[test_idx]
            else:
                y_hat_orig = np.exp(y_hat_raw)
                y_te_orig = np.exp(target[test_idx])

            all_true_arr.extend(y_te_orig)
            all_pred_arr.extend(y_hat_orig)
            all_models_arr.extend([held_out] * len(test_idx))

        if not all_true_arr:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        true_arr = np.array(all_true_arr)
        pred_arr = np.array(all_pred_arr)
        models_arr = np.array(all_models_arr)

        for model_label in sorted(set(all_models_arr)):
            mask = models_arr == model_label
            short = model_label.replace("Qwen_Qwen3-", "Q3-").replace(
                "zai-org_GLM-4.7-Flash", "GLM-4.7"
            )
            ax.scatter(true_arr[mask], pred_arr[mask], label=short, s=15, alpha=0.7)

        lo = min(true_arr.min(), pred_arr.min()) * 0.5
        hi = max(true_arr.max(), pred_arr.max()) * 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
        ax.set_xlabel("Actual Time (s)")
        ax.set_ylabel("Predicted Time (s)")
        ax.set_title(f"LOMO: {op} ({best_approach})")
        ax.legend(fontsize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"lomo_attn_{op}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved LOMO residual plots for {len(lomo_df['operator'].unique())} operators")

    return {"lomo_df": lomo_df}


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def feature_importance(
    all_results: dict[str, dict],
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """Generate feature importance: Ridge coefficients + GBR importances."""
    output_dir = Path(output_dir)

    ridge_rows: list[dict] = []
    gbr_rows: list[dict] = []

    for op, results in all_results.items():
        if "A-Ridge" in results:
            model = results["A-Ridge"]["model"]
            features = results["A-Ridge"]["features"]
            coefs = model.named_steps["ridge"].coef_
            for feat, coef in zip(features, coefs):
                ridge_rows.append({"operator": op, "feature": feat, "coefficient": float(coef)})

        if "B-GBR" in results:
            model = results["B-GBR"]["model"]
            features = results["B-GBR"]["features"]
            importances = model.feature_importances_
            for feat, imp in zip(features, importances):
                gbr_rows.append({"operator": op, "feature": feat, "importance": float(imp)})

    ridge_df = pd.DataFrame(ridge_rows)
    gbr_df = pd.DataFrame(gbr_rows)

    print(f"\n=== Feature Importance (GBR) ===")
    for op in sorted(gbr_df["operator"].unique()):
        sub = gbr_df[gbr_df["operator"] == op].sort_values("importance", ascending=False)
        top3 = ", ".join(f"{r['feature']}({r['importance']:.3f})" for _, r in sub.head(3).iterrows())
        print(f"  {op:<30s}: {top3}")

    # Heatmaps
    if len(gbr_df) > 0:
        pivot = gbr_df.pivot_table(
            index="feature", columns="operator", values="importance", aggfunc="mean",
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("GBR Feature Importance across Attention Operators")
        fig.tight_layout()
        fig.savefig(output_dir / "attn_feature_importance_heatmap.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved attn_feature_importance_heatmap.png")

    if len(ridge_df) > 0:
        pivot_r = ridge_df.pivot_table(
            index="feature", columns="operator", values="coefficient", aggfunc="mean",
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_r, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("Ridge Coefficients across Attention Operators")
        fig.tight_layout()
        fig.savefig(output_dir / "attn_ridge_coefficients_heatmap.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved attn_ridge_coefficients_heatmap.png")

    return {"ridge_coefficients": ridge_df, "gbr_importances": gbr_df}


# ---------------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    df: pd.DataFrame,
    all_results: dict[str, dict],
    lomo_results: dict,
    importance_results: dict,
    output_dir: str | Path,
) -> None:
    """Persist CSV, JSON summary, and print findings."""
    output_dir = Path(output_dir)

    csv_cols = [
        "operator_name", "variant", "model", "gpu", "quant",
        "batch_size", "seq_len", "kv_cache_size",
        "time_s", "flops", "bandwidth_gb_s",
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "num_experts", "num_experts_per_tok",
        "tokens", "head_dim", "gqa_ratio", "is_moe",
        "roofline_time_s",
        "log_time_s", "log_flops", "log_tokens",
        "log_hidden_size", "log_batch_size", "log_seq_len",
        "log_kv_cache_size", "log_bandwidth_gb_s",
    ]
    available_cols = [c for c in csv_cols if c in df.columns]
    csv_path = output_dir / "attention_latency_data.csv"
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

    # Per-operator CV summary
    op_summary: dict = {}
    for op, results in all_results.items():
        op_entry: dict = {}
        for approach_name, approach_data in results.items():
            if "cv" in approach_data:
                op_entry[approach_name] = {
                    k: _safe_val(v) for k, v in approach_data["cv"].items()
                    if k not in ("all_true", "all_pred")
                }
        op_summary[op] = op_entry

    lomo_df = lomo_results.get("lomo_df", pd.DataFrame())
    lomo_list = []
    if len(lomo_df) > 0:
        lomo_list = [
            {k: _safe_val(v) for k, v in row.items()}
            for _, row in lomo_df.iterrows()
        ]

    summary = {
        "n_rows": len(df),
        "models": sorted(df["model"].unique().tolist()),
        "operators": sorted(df["operator_name"].unique().tolist()),
        "batch_sizes": sorted(int(x) for x in df["batch_size"].unique()),
        "seq_lengths": sorted(int(x) for x in df["seq_len"].unique()),
        "per_operator_cv": op_summary,
        "lomo_cv": lomo_list,
    }

    json_path = output_dir / "attention_latency_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_safe_val)
    print(f"  Saved {json_path}")

    # Print findings
    approach_names = ["A-Ridge", "B-GBR", "C-Roofline", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]
    print(f"\n{'='*80}")
    print(f"  Attention Latency Surrogate — Summary of Findings")
    print(f"{'='*80}")
    print(f"  Data: {len(df)} rows, {len(df['model'].unique())} models, "
          f"{len(df['operator_name'].unique())} operators")

    print(f"\n  Per-Operator K-Fold CV (MAPE):")
    print(f"  {'Operator':<30s}", end="")
    for a in approach_names:
        print(f"  {a:>12s}", end="")
    print()
    for op in sorted(all_results.keys()):
        print(f"  {op:<30s}", end="")
        for a in approach_names:
            if a in all_results[op]:
                mape = all_results[op][a]["cv"]["mape_mean"]
                print(f"  {mape:>12.4f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    if len(lomo_df) > 0:
        print(f"\n  LOMO CV (generalization to unseen models):")
        for a in approach_names:
            sub = lomo_df[lomo_df["approach"] == a]
            if len(sub) > 0:
                print(f"    {a:<14s}: mean MAPE={sub['mape'].mean():.4f}, "
                      f"mean R²={sub['r2'].mean():.4f}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    operators: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> dict:
    """Run the full attention latency surrogate analysis."""
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir.parent / "attention_latency_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  Attention Latency Surrogate Model")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    if operators:
        print(f"  Operators filter: {operators}")
    if exclude_models:
        print(f"  Excluding models matching: {exclude_models}")
    print(f"{'='*80}")

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
        df = df.reset_index(drop=True)
        print(f"  Remaining: {len(df)} rows, models={sorted(df['model'].unique())}")

    if operators:
        df = df[df["operator_name"].isin(operators)].reset_index(drop=True)
        print(f"  Filtered to operators: {sorted(df['operator_name'].unique())}, "
              f"{len(df)} rows")

    df = add_architecture_features(df)
    df = compute_derived_features(df)

    print(f"\n=== EDA Plots ===")
    generate_eda_plots(df, output_dir)

    all_results = train_all_operators(df)
    lomo_results = leave_one_model_out_cv(df, output_dir)
    importance_results = feature_importance(all_results, df, output_dir)
    save_artifacts(df, all_results, lomo_results, importance_results, output_dir)

    return {
        "df": df,
        "all_results": all_results,
        "lomo_results": lomo_results,
        "importance_results": importance_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Attention latency surrogate: per-operator latency prediction",
    )
    parser.add_argument(
        "data_dir", type=str,
        help="Path to profiles directory containing model/gpu/quant subdirs",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--operators", type=str, nargs="+", default=None,
        help="Only train surrogates for these operators",
    )
    parser.add_argument(
        "--exclude-models", type=str, nargs="+", default=None,
        help="Model name substrings to exclude",
    )
    args = parser.parse_args(argv)

    run_pipeline(args.data_dir, args.output_dir, args.operators, args.exclude_models)
    print("\nDone.")


if __name__ == "__main__":
    main()
