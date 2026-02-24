"""Operator Latency Surrogate Model: Per-Operator Latency Prediction.

Builds surrogate models predicting per-operator latency (time_s) from
operator profiling data (token_ops.csv) across models, batch sizes,
and sequence lengths.  One model is trained per operator (16 operators)
using the same five modeling approaches as the prefill/decode surrogates.

Compares five approaches:
  A-Ridge:    Log-linear Ridge (interpretable power-law exponents)
  B-GBR:     GradientBoostingRegressor (full engineered features)
  C-Roofline: Roofline-residual model (predict ratio to physics baseline)
  D-ScaleInv: Scale-invariant features only (normalized/ratio features)
  E-PowerLaw: Direct OLS power-law fit on raw log features

Evaluated with adaptive K-fold CV and leave-one-model-out CV to assess
generalization to unseen model architectures.

Usage:
    python -m ipw.simulator.operator_energy_surrogate /path/to/profiles/
    python -m ipw.simulator.operator_energy_surrogate /path/to/profiles/ --output-dir ./out
    python -m ipw.simulator.operator_energy_surrogate /path/to/profiles/ --operators linear_qkv softmax
    python -m ipw.simulator.operator_energy_surrogate /path/to/profiles/ --exclude-models 30B-A3B
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model architecture configs (HuggingFace config.json values).
# MoE fields: num_experts / num_experts_per_tok (0 = dense model).
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

# Hardware specs for roofline model.
HARDWARE_SPECS = {
    "A100SXM4": {
        "peak_fp16_tflops": 312.0,
        "hbm_bw_gb_s": 2039.0,
        "tdp_w": 400,
        "memory_gb": 80,
    },
}

# Operator category mapping.
OPERATOR_CATEGORIES: dict[str, str] = {
    "linear_qkv": "linear",
    "linear_o": "linear",
    "mlp_up": "linear",
    "mlp_gate": "linear",
    "mlp_down": "linear",
    "lm_head": "linear",
    "rmsnorm": "normalization",
    "layernorm": "normalization",
    "silu_activation": "activation",
    "gelu_activation": "activation",
    "embedding": "embedding",
    "residual_add": "elementwise",
    "dropout": "elementwise",
    "rotary_embedding": "attention_helper",
    "softmax": "attention_helper",
    "cross_entropy_loss": "attention_helper",
}

# Features for the log-linear Ridge model (Approach A).
# Embedding ops get a separate feature list since flops=0.
LOG_LINEAR_FEATURES = [
    "log_flops",
    "log_tokens",
    "log_hidden_size",
    "log_batch_size",
    "log_seq_len",
]

LOG_LINEAR_FEATURES_EMBEDDING = [
    "log_bandwidth_gb_s",
    "log_tokens",
    "log_hidden_size",
    "log_batch_size",
    "log_seq_len",
]

# Features for GBR (Approach B).
GBR_FEATURES = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "batch_size",
    "seq_len",
    "flops",
    "head_dim",
    "gqa_ratio",
    "tokens",
    "is_moe",
    "log_flops",
    "log_tokens",
    "log_batch_size",
    "log_seq_len",
]

GBR_FEATURES_EMBEDDING = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "batch_size",
    "seq_len",
    "bandwidth_gb_s",
    "head_dim",
    "gqa_ratio",
    "tokens",
    "is_moe",
    "log_bandwidth_gb_s",
    "log_tokens",
    "log_batch_size",
    "log_seq_len",
]

# Features for roofline-residual (Approach C).
ROOFLINE_RESIDUAL_FEATURES = [
    "log_seq_len",
    "log_batch_size",
    "log_tokens",
    "arithmetic_intensity",
    "is_moe",
]

# Scale-invariant features (Approach D).
SCALE_INVARIANT_FEATURES = [
    "log_seq_len",
    "log_batch_size",
    "gqa_ratio",
    "is_moe",
    "arithmetic_intensity",
    "log_tokens",
]

SCALE_INVARIANT_FEATURES_EMBEDDING = [
    "log_seq_len",
    "log_batch_size",
    "gqa_ratio",
    "is_moe",
    "log_bandwidth_gb_s",
    "log_tokens",
]

# Power-law OLS features (Approach E).
POWER_LAW_FEATURES = [
    "log_flops",
    "log_tokens",
    "log_hidden_size",
    "log_seq_len",
]

POWER_LAW_FEATURES_EMBEDDING = [
    "log_bandwidth_gb_s",
    "log_tokens",
    "log_hidden_size",
    "log_seq_len",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Load token_ops.csv files from profiles directory.

    Expected layout: data_dir/<model>/<gpu>/<quant>/token_ops.csv
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*/*/*/token_ops.csv"))

    frames: list[pd.DataFrame] = []
    skipped = 0

    for csv_path in csv_files:
        try:
            # Extract model/gpu/quant from path
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
        print(f"ERROR: No token_ops.csv files found under {data_dir}")
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
    print(f"  GPUs: {sorted(df['gpu'].unique())}")
    print(f"  Quants: {sorted(df['quant'].unique())}")
    print(f"  Operators: {sorted(df['operator_name'].unique())}")
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

    # Fill NaN in bandwidth_gb_s with 0 for non-embedding ops
    df["bandwidth_gb_s"] = df["bandwidth_gb_s"].fillna(0.0)

    # Arithmetic intensity (for ops with flops > 0)
    # Estimate bytes accessed as flops / arithmetic_intensity_ratio
    # For simplicity: use flops / (peak_tflops * 1e12) as compute time proxy
    hw = HARDWARE_SPECS["A100SXM4"]
    peak_flops_per_s = hw["peak_fp16_tflops"] * 1e12

    # Roofline time: max(compute_time, memory_time)
    # compute_time = flops / peak_flops
    # For memory time we'd need bytes accessed; approximate from flops / AI
    # We use a simple compute-bound roofline for ops with flops > 0
    df["roofline_time_s"] = df["flops"].astype(float) / peak_flops_per_s

    # Arithmetic intensity approximation: flops / (model_size_bytes_approx)
    # Use a simple proxy: flops / (hidden_size * tokens * 2 bytes)
    bytes_proxy = df["hidden_size"].astype(float) * df["tokens"].astype(float) * 2.0
    bytes_proxy = bytes_proxy.clip(lower=1.0)
    df["arithmetic_intensity"] = df["flops"].astype(float) / bytes_proxy

    # Log features (clip to avoid log(0))
    df["log_time_s"] = np.log(df["time_s"].clip(lower=1e-15))
    df["log_flops"] = np.log(df["flops"].astype(float).clip(lower=1.0))
    df["log_tokens"] = np.log(df["tokens"].astype(float).clip(lower=1.0))
    df["log_hidden_size"] = np.log(df["hidden_size"].astype(float))
    df["log_batch_size"] = np.log(df["batch_size"].astype(float).clip(lower=1.0))
    df["log_seq_len"] = np.log(df["seq_len"].astype(float))
    df["log_bandwidth_gb_s"] = np.log(df["bandwidth_gb_s"].clip(lower=1e-10))

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

    # Operator category
    df["operator_category"] = df["operator_name"].map(OPERATOR_CATEGORIES).fillna("other")

    return df


# ---------------------------------------------------------------------------
# EDA Plots
# ---------------------------------------------------------------------------


def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate EDA plots for operator latency data."""
    output_dir = Path(output_dir)

    df = df.copy()
    df["model_short"] = df["model"].str.replace("Qwen_Qwen3-", "Q3-").str.replace(
        "zai-org_GLM-4.7-Flash", "GLM-4.7"
    )

    operators = sorted(df["operator_name"].unique())
    palette = sns.color_palette("tab10", df["model_short"].nunique())
    model_names = sorted(df["model_short"].unique())

    # --- Plot 1: Latency vs seq_len grid (one subplot per operator) ---
    n_ops = len(operators)
    ncols = 4
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
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
        ax.set_title(op, fontsize=10)
        ax.grid(True, alpha=0.3)
    # Hide unused axes
    for idx in range(n_ops, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.suptitle("Operator Latency vs Sequence Length (B=1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "eda_latency_vs_seq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_latency_vs_seq.png")

    # --- Plot 2: FLOPs vs latency scatter (all operators, colored by category) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    cats = sorted(df["operator_category"].unique())
    cat_palette = sns.color_palette("Set2", len(cats))
    for ci, cat in enumerate(cats):
        sub = df[df["operator_category"] == cat]
        # Skip zero-flop rows for this plot
        sub_pos = sub[sub["flops"] > 0]
        if len(sub_pos) > 0:
            ax.scatter(sub_pos["flops"], sub_pos["time_s"], label=cat, s=8,
                       alpha=0.4, color=cat_palette[ci])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Time (s)")
    ax.set_title("Operator Latency vs FLOPs (colored by category)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_flops_vs_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_flops_vs_latency.png")

    # --- Plot 3: Operator time breakdown heatmap (fraction of total per model/seq_len) ---
    # For batch_size=1, compute fraction of total time per operator
    sub_b1 = df[df["batch_size"] == 1].copy()
    total_per_config = sub_b1.groupby(["model_short", "seq_len"])["time_s"].transform("sum")
    sub_b1["time_frac"] = sub_b1["time_s"] / total_per_config

    # Pivot: rows=operator, cols=model_short, values=mean time fraction across seq_lens
    pivot = sub_b1.pivot_table(
        index="operator_name", columns="model_short",
        values="time_frac", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Mean Time Fraction per Operator (B=1, averaged over seq lengths)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_operator_time_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_operator_time_heatmap.png")


# ---------------------------------------------------------------------------
# Cross-Validation Helpers
# ---------------------------------------------------------------------------


def _cv_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    *,
    collect_preds: bool = False,
) -> dict:
    """Run cross-validation predicting log_time_s, evaluate on original scale."""
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)

        # Metrics on original scale
        y_te_orig = np.exp(y_te)
        y_hat_orig = np.exp(y_hat)

        maes.append(mean_absolute_error(y_te_orig, y_hat_orig))
        r2s.append(r2_score(y_te_orig, y_hat_orig))
        mapes.append(mean_absolute_percentage_error(y_te_orig, y_hat_orig))

        if collect_preds:
            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)

    result = {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mape_mean": float(np.mean(mapes)),
        "mape_std": float(np.std(mapes)),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


def _cv_evaluate_roofline_residual(
    model,
    X: np.ndarray,
    y_log_ratio: np.ndarray,
    roofline_time: np.ndarray,
    actual_time: np.ndarray,
    kf: KFold,
    *,
    collect_preds: bool = False,
) -> dict:
    """CV for roofline-residual approach: predict log(time/roofline), evaluate on original."""
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_log_ratio[train_idx]

        model.fit(X_tr, y_tr)
        y_hat_log_ratio = model.predict(X_te)

        # Reconstruct: predicted_time = exp(log_ratio) * roofline_time
        y_hat_orig = np.exp(y_hat_log_ratio) * roofline_time[test_idx]
        y_te_orig = actual_time[test_idx]

        maes.append(mean_absolute_error(y_te_orig, y_hat_orig))
        r2s.append(r2_score(y_te_orig, y_hat_orig))
        mapes.append(mean_absolute_percentage_error(y_te_orig, y_hat_orig))

        if collect_preds:
            all_true.extend(y_te_orig)
            all_pred.extend(y_hat_orig)

    result = {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mape_mean": float(np.mean(mapes)),
        "mape_std": float(np.std(mapes)),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


# ---------------------------------------------------------------------------
# Per-Operator Model Training
# ---------------------------------------------------------------------------


def _get_features_for_operator(op_name: str, approach: str) -> list[str]:
    """Return the correct feature list for an operator and approach."""
    is_embedding = (op_name == "embedding")
    if approach == "A-Ridge":
        return LOG_LINEAR_FEATURES_EMBEDDING if is_embedding else LOG_LINEAR_FEATURES
    if approach == "B-GBR":
        return GBR_FEATURES_EMBEDDING if is_embedding else GBR_FEATURES
    if approach == "C-Roofline":
        return ROOFLINE_RESIDUAL_FEATURES
    if approach == "D-ScaleInv":
        return SCALE_INVARIANT_FEATURES_EMBEDDING if is_embedding else SCALE_INVARIANT_FEATURES
    if approach == "E-PowerLaw":
        return POWER_LAW_FEATURES_EMBEDDING if is_embedding else POWER_LAW_FEATURES
    return []


def train_models_for_operator(
    df_op: pd.DataFrame,
    op_name: str,
) -> dict:
    """Fit all five approaches for one operator with adaptive CV.

    Returns dict with model objects, CV results, and feature names.
    """
    n_samples = len(df_op)
    n_splits = min(10, max(2, n_samples // 3))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y = df_op["log_time_s"].values
    results: dict = {}
    is_embedding = (op_name == "embedding")

    # --- Approach A: Log-linear Ridge ---
    feat_a = _get_features_for_operator(op_name, "A-Ridge")
    avail_a = [f for f in feat_a if f in df_op.columns]
    X_a = df_op[avail_a].values
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge_cv = _cv_evaluate(ridge_pipe, X_a, y, kf, collect_preds=True)
    ridge_pipe.fit(X_a, y)
    results["A-Ridge"] = {
        "model": ridge_pipe, "features": avail_a,
        "cv": ridge_cv, "target": "log_time_s",
    }

    # --- Approach B: GBR ---
    feat_b = _get_features_for_operator(op_name, "B-GBR")
    avail_b = [f for f in feat_b if f in df_op.columns]
    X_b = df_op[avail_b].values
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        learning_rate=0.1, random_state=42,
    )
    gbr_cv = _cv_evaluate(gbr, X_b, y, kf, collect_preds=True)
    gbr.fit(X_b, y)
    results["B-GBR"] = {
        "model": gbr, "features": avail_b,
        "cv": gbr_cv, "target": "log_time_s",
    }

    # --- Approach C: Roofline-Residual (skip for embedding where flops=0) ---
    if not is_embedding:
        feat_c = _get_features_for_operator(op_name, "C-Roofline")
        avail_c = [f for f in feat_c if f in df_op.columns]
        # Only use rows with valid roofline ratio
        valid_mask = df_op["log_time_roofline_ratio"].notna()
        df_c = df_op[valid_mask]
        if len(df_c) >= 6:
            X_c = df_c[avail_c].values
            y_log_ratio = df_c["log_time_roofline_ratio"].values
            roofline_t = df_c["roofline_time_s"].values
            actual_t = df_c["time_s"].values
            n_splits_c = min(10, max(2, len(df_c) // 3))
            kf_c = KFold(n_splits=n_splits_c, shuffle=True, random_state=42)

            roof_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            roof_cv = _cv_evaluate_roofline_residual(
                roof_pipe, X_c, y_log_ratio, roofline_t, actual_t, kf_c,
                collect_preds=True,
            )
            roof_pipe.fit(X_c, y_log_ratio)
            results["C-Roofline"] = {
                "model": roof_pipe, "features": avail_c,
                "cv": roof_cv, "target": "log_ratio",
            }

    # --- Approach D: Scale-Invariant ---
    feat_d = _get_features_for_operator(op_name, "D-ScaleInv")
    avail_d = [f for f in feat_d if f in df_op.columns]
    X_d = df_op[avail_d].values
    si_gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        learning_rate=0.1, random_state=42,
    )
    si_cv = _cv_evaluate(si_gbr, X_d, y, kf, collect_preds=True)
    si_gbr.fit(X_d, y)
    results["D-ScaleInv"] = {
        "model": si_gbr, "features": avail_d,
        "cv": si_cv, "target": "log_time_s",
    }

    # --- Approach E: Power-Law OLS ---
    feat_e = _get_features_for_operator(op_name, "E-PowerLaw")
    avail_e = [f for f in feat_e if f in df_op.columns]
    X_e = df_op[avail_e].values
    ols = LinearRegression()
    pl_cv = _cv_evaluate(ols, X_e, y, kf, collect_preds=True)
    ols.fit(X_e, y)
    results["E-PowerLaw"] = {
        "model": ols, "features": avail_e,
        "cv": pl_cv, "target": "log_time_s",
    }

    return results


def train_all_operators(df: pd.DataFrame) -> dict[str, dict]:
    """Train surrogate models for all operators.

    Returns dict mapping operator_name → approach results.
    """
    print(f"\n=== Per-Operator Cross-Validation ===")

    operators = sorted(df["operator_name"].unique())
    all_results: dict[str, dict] = {}

    print(f"  {'Operator':<22s} {'N':>5s}  {'A-Ridge':>12s}  {'B-GBR':>12s}  "
          f"{'C-Roofline':>12s}  {'D-ScaleInv':>12s}  {'E-PowerLaw':>12s}")
    print(f"  {'-'*22} {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for op in operators:
        df_op = df[df["operator_name"] == op].reset_index(drop=True)
        n = len(df_op)

        if n < 6:
            print(f"  {op:<22s} {n:>5d}  SKIPPED (too few samples)")
            continue

        results = train_models_for_operator(df_op, op)
        all_results[op] = results

        # Print summary row
        parts = [f"  {op:<22s} {n:>5d}"]
        for approach in ["A-Ridge", "B-GBR", "C-Roofline", "D-ScaleInv", "E-PowerLaw"]:
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
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """LOMO CV: train on N-1 models, predict held-out model. Per operator, all approaches."""
    output_dir = Path(output_dir)
    print(f"\n=== Leave-One-Model-Out Cross-Validation ===")

    operators = sorted(df["operator_name"].unique())
    logo = LeaveOneGroupOut()
    lomo_rows: list[dict] = []

    for op in operators:
        df_op = df[df["operator_name"] == op].reset_index(drop=True)
        n = len(df_op)
        if n < 6:
            continue

        is_embedding = (op == "embedding")
        y = df_op["log_time_s"].values
        groups = df_op["model"].values

        # Check we have at least 2 groups
        if len(df_op["model"].unique()) < 2:
            continue

        approaches = ["A-Ridge", "B-GBR", "D-ScaleInv", "E-PowerLaw"]
        if not is_embedding:
            approaches.insert(2, "C-Roofline")

        for approach in approaches:
            features = _get_features_for_operator(op, approach)
            avail_feat = [f for f in features if f in df_op.columns]

            if approach == "C-Roofline":
                valid_mask = df_op["log_time_roofline_ratio"].notna()
                df_c = df_op[valid_mask].reset_index(drop=True)
                if len(df_c) < 6 or len(df_c["model"].unique()) < 2:
                    continue
                X = df_c[avail_feat].values
                target = df_c["log_time_roofline_ratio"].values
                roofline_t = df_c["roofline_time_s"].values
                actual_t = df_c["time_s"].values
                groups_c = df_c["model"].values
            else:
                X = df_op[avail_feat].values
                target = y
                groups_c = groups

            for train_idx, test_idx in logo.split(X, target, groups_c):
                held_out = groups_c[test_idx[0]]
                n_test = len(test_idx)

                if approach == "A-Ridge":
                    estimator = Pipeline([
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=1.0)),
                    ])
                elif approach in ("B-GBR", "D-ScaleInv"):
                    estimator = GradientBoostingRegressor(
                        n_estimators=200, max_depth=4, min_samples_leaf=5,
                        learning_rate=0.1, random_state=42,
                    )
                elif approach == "C-Roofline":
                    estimator = Pipeline([
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=1.0)),
                    ])
                else:  # E-PowerLaw
                    estimator = LinearRegression()

                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr = target[train_idx]

                estimator.fit(X_tr, y_tr)
                y_hat_raw = estimator.predict(X_te)

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

    # --- Summary table: mean MAPE per operator × approach ---
    print(f"\n  LOMO CV — Mean MAPE per Operator × Approach")
    approach_names = ["A-Ridge", "B-GBR", "C-Roofline", "D-ScaleInv", "E-PowerLaw"]
    header = f"  {'Operator':<22s}"
    for a in approach_names:
        header += f"  {a:>12s}"
    print(header)
    print(f"  {'-'*22}" + f"  {'-'*12}" * len(approach_names))

    for op in sorted(lomo_df["operator"].unique()):
        row_str = f"  {op:<22s}"
        for a in approach_names:
            sub = lomo_df[(lomo_df["operator"] == op) & (lomo_df["approach"] == a)]
            if len(sub) > 0:
                row_str += f"  {sub['mape'].mean():>12.4f}"
            else:
                row_str += f"  {'N/A':>12s}"
        print(row_str)

    # --- Per-approach aggregate ---
    print(f"\n  LOMO CV — Overall Aggregate")
    print(f"  {'Approach':<14s} {'Mean MAPE':>10s} {'Mean R²':>10s}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    for a in approach_names:
        sub = lomo_df[lomo_df["approach"] == a]
        if len(sub) > 0:
            print(f"  {a:<14s} {sub['mape'].mean():>10.4f} {sub['r2'].mean():>10.4f}")

    # --- Residual plots per operator (best approach) ---
    for op in sorted(lomo_df["operator"].unique()):
        sub_op = lomo_df[lomo_df["operator"] == op]
        # Pick approach with lowest mean MAPE for this operator's plot
        best_approach = sub_op.groupby("approach")["mape"].mean().idxmin()

        # Re-run to collect predictions for plotting
        df_op = df[df["operator_name"] == op].reset_index(drop=True)
        is_embedding = (op == "embedding")
        features = _get_features_for_operator(op, best_approach)
        avail_feat = [f for f in features if f in df_op.columns]

        if best_approach == "C-Roofline":
            valid_mask = df_op["log_time_roofline_ratio"].notna()
            df_plot = df_op[valid_mask].reset_index(drop=True)
            X = df_plot[avail_feat].values
            target = df_plot["log_time_roofline_ratio"].values
            roofline_t = df_plot["roofline_time_s"].values
            actual_t = df_plot["time_s"].values
            groups_plot = df_plot["model"].values
        else:
            df_plot = df_op
            X = df_plot[avail_feat].values
            target = df_plot["log_time_s"].values
            groups_plot = df_plot["model"].values

        all_true_arr, all_pred_arr, all_models_arr = [], [], []
        for train_idx, test_idx in logo.split(X, target, groups_plot):
            held_out = groups_plot[test_idx[0]]

            if best_approach == "A-Ridge":
                est = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            elif best_approach in ("B-GBR", "D-ScaleInv"):
                est = GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, min_samples_leaf=5,
                    learning_rate=0.1, random_state=42,
                )
            elif best_approach == "C-Roofline":
                est = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            else:
                est = LinearRegression()

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

        if len(all_true_arr) == 0:
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
        fig.savefig(
            output_dir / f"lomo_{op}.png", dpi=150, bbox_inches="tight",
        )
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
    """Generate feature importance analysis: Ridge coefficients + GBR importances."""
    output_dir = Path(output_dir)

    ridge_rows: list[dict] = []
    gbr_rows: list[dict] = []

    for op, results in all_results.items():
        # Ridge coefficients
        if "A-Ridge" in results:
            ridge_model = results["A-Ridge"]["model"]
            features = results["A-Ridge"]["features"]
            coefs = ridge_model.named_steps["ridge"].coef_
            for feat, coef in zip(features, coefs):
                ridge_rows.append({
                    "operator": op,
                    "feature": feat,
                    "coefficient": float(coef),
                })

        # GBR feature importances
        if "B-GBR" in results:
            gbr_model = results["B-GBR"]["model"]
            features = results["B-GBR"]["features"]
            importances = gbr_model.feature_importances_
            for feat, imp in zip(features, importances):
                gbr_rows.append({
                    "operator": op,
                    "feature": feat,
                    "importance": float(imp),
                })

    ridge_df = pd.DataFrame(ridge_rows)
    gbr_df = pd.DataFrame(gbr_rows)

    # --- Print top features per operator ---
    print(f"\n=== Feature Importance (GBR) ===")
    for op in sorted(gbr_df["operator"].unique()):
        sub = gbr_df[gbr_df["operator"] == op].sort_values("importance", ascending=False)
        top3 = ", ".join(f"{r['feature']}({r['importance']:.3f})" for _, r in sub.head(3).iterrows())
        print(f"  {op:<22s}: {top3}")

    # --- Cross-operator summary heatmap ---
    if len(gbr_df) > 0:
        pivot = gbr_df.pivot_table(
            index="feature", columns="operator", values="importance", aggfunc="mean",
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("GBR Feature Importance across Operators")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_importance_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved feature_importance_heatmap.png")

    if len(ridge_df) > 0:
        pivot_r = ridge_df.pivot_table(
            index="feature", columns="operator", values="coefficient", aggfunc="mean",
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(pivot_r, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("Ridge Coefficients across Operators")
        fig.tight_layout()
        fig.savefig(output_dir / "ridge_coefficients_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ridge_coefficients_heatmap.png")

    return {
        "ridge_coefficients": ridge_df,
        "gbr_importances": gbr_df,
    }


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

    # CSV with all features
    csv_cols = [
        "operator_name", "model", "gpu", "quant", "batch_size", "seq_len",
        "time_s", "flops", "bandwidth_gb_s",
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "num_experts", "num_experts_per_tok",
        "tokens", "head_dim", "gqa_ratio", "is_moe",
        "roofline_time_s", "arithmetic_intensity",
        "operator_category",
        "log_time_s", "log_flops", "log_tokens",
        "log_hidden_size", "log_batch_size", "log_seq_len",
    ]
    available_cols = [c for c in csv_cols if c in df.columns]
    csv_path = output_dir / "operator_latency_data.csv"
    df[available_cols].to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path} ({len(df)} rows, {len(available_cols)} columns)")

    # JSON summary
    def _safe_val(v):
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    # Build per-operator CV summary
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

    # LOMO summary
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

    json_path = output_dir / "operator_latency_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_safe_val)
    print(f"  Saved {json_path}")

    # --- Print findings summary ---
    print(f"\n{'='*70}")
    print(f"  Operator Latency Surrogate — Summary of Findings")
    print(f"{'='*70}")
    print(f"  Data: {len(df)} rows, {len(df['model'].unique())} models, "
          f"{len(df['operator_name'].unique())} operators")
    print(f"  Batch sizes: {sorted(int(x) for x in df['batch_size'].unique())}")
    print(f"  Seq lengths: {sorted(int(x) for x in df['seq_len'].unique())}")

    print(f"\n  Per-Operator K-Fold CV (MAPE):")
    approach_names = ["A-Ridge", "B-GBR", "C-Roofline", "D-ScaleInv", "E-PowerLaw"]
    print(f"  {'Operator':<22s}", end="")
    for a in approach_names:
        print(f"  {a:>12s}", end="")
    print()

    for op in sorted(all_results.keys()):
        print(f"  {op:<22s}", end="")
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
    """Run the full operator latency surrogate analysis.

    Parameters
    ----------
    data_dir : path to profiles directory containing model/gpu/quant subdirs
    output_dir : directory for saved artifacts
    operators : if set, only train surrogates for these operators
    exclude_models : list of model name substrings to exclude

    Returns
    -------
    dict with DataFrames, model objects, and CV results.
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir.parent / "operator_latency_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Operator Latency Surrogate Model")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    if operators:
        print(f"  Operators filter: {operators}")
    if exclude_models:
        print(f"  Excluding models matching: {exclude_models}")
    print(f"{'='*70}")

    # Load data
    df = load_data(data_dir)
    if len(df) == 0:
        print("ERROR: No data loaded. Check data directory.")
        sys.exit(1)

    # Filter excluded models
    if exclude_models:
        for pattern in exclude_models:
            before = len(df)
            df = df[~df["model"].str.lower().str.contains(pattern.lower())]
            dropped = before - len(df)
            if dropped:
                print(f"  Filtered out {dropped} rows matching '{pattern}'")
        df = df.reset_index(drop=True)
        print(f"  Remaining: {len(df)} rows, models={sorted(df['model'].unique())}")

    # Filter operators
    if operators:
        df = df[df["operator_name"].isin(operators)].reset_index(drop=True)
        print(f"  Filtered to operators: {sorted(df['operator_name'].unique())}, "
              f"{len(df)} rows")

    # Architecture features
    df = add_architecture_features(df)

    # Derived features
    df = compute_derived_features(df)

    # EDA plots
    print(f"\n=== EDA Plots ===")
    generate_eda_plots(df, output_dir)

    # Train models per operator (K-fold CV)
    all_results = train_all_operators(df)

    # LOMO CV
    lomo_results = leave_one_model_out_cv(df, output_dir)

    # Feature importance
    importance_results = feature_importance(all_results, df, output_dir)

    # Save artifacts
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
        description="Operator latency surrogate: per-operator latency prediction",
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
        help="Only train surrogates for these operators (e.g. linear_qkv softmax)",
    )
    parser.add_argument(
        "--exclude-models", type=str, nargs="+", default=None,
        help="Model name substrings to exclude (e.g. 30B-A3B)",
    )
    args = parser.parse_args(argv)

    run_pipeline(args.data_dir, args.output_dir, args.operators, args.exclude_models)
    print("\nDone.")


if __name__ == "__main__":
    main()
