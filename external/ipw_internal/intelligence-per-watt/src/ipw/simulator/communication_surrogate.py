"""Communication Operator Latency Surrogate Model.

Builds surrogate models predicting per-operator latency (time_s) for
communication operators from communication.csv profiling data.

Seven operations with distinct physics:
  - allreduce, allgather, reduce_scatter, send_recv_p2p: collective/p2p comms
    (parameterized by num_gpus × message_size_bytes; model-independent for
     pure network ops)
  - pipeline_bubble_idle, pipeline_stage_forward: pipeline-parallel scheduling
    (model-dependent)
  - tensor_parallel_split: TP split overhead (model-dependent)

Key differences from token_ops/attention surrogates:
  - Features: num_gpus, message_size_bytes (NOT batch_size/seq_len)
  - Some operations are model-independent (pure hardware)
  - bandwidth_gb_s populated for some ops

Models compared (per operator):
  A-Ridge:      Log-linear Ridge
  B-GBR:        GradientBoostingRegressor
  C-Roofline:   bandwidth-residual model
  D-Poly2Ridge: Degree-2 polynomial in log space
  E-PowerLaw:   OLS on raw log features
  F-RF:         RandomForestRegressor

Usage:
    python -m ipw.simulator.communication_surrogate /path/to/profiles/
    python -m ipw.simulator.communication_surrogate /path/to/profiles/ --output-dir ./out
    python -m ipw.simulator.communication_surrogate /path/to/profiles/ --operators allreduce tensor_parallel_split
    python -m ipw.simulator.communication_surrogate /path/to/profiles/ --exclude-models 30B-A3B
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
        "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "intermediate_size": 3072, "vocab_size": 151936,
        "num_experts": 0, "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-1.7B": {
        "hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "intermediate_size": 6144, "vocab_size": 151936,
        "num_experts": 0, "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-4B": {
        "hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 9216, "vocab_size": 151936,
        "num_experts": 0, "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-8B": {
        "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 12288, "vocab_size": 151936,
        "num_experts": 0, "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-14B": {
        "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40,
        "num_key_value_heads": 8, "intermediate_size": 17408, "vocab_size": 151936,
        "num_experts": 0, "num_experts_per_tok": 0,
    },
    "Qwen_Qwen3-30B-A3B": {
        "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
        "num_key_value_heads": 4, "intermediate_size": 6144, "vocab_size": 151936,
        "num_experts": 128, "num_experts_per_tok": 8,
    },
    "Qwen_Qwen3-32B": {
        "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 64,
        "num_key_value_heads": 8, "intermediate_size": 25600, "vocab_size": 151936,
        "num_experts": 0, "num_experts_per_tok": 0,
    },
    "zai-org_GLM-4.7-Flash": {
        "hidden_size": 2048, "num_hidden_layers": 47, "num_attention_heads": 20,
        "num_key_value_heads": 20, "intermediate_size": 10240, "vocab_size": 154880,
        "num_experts": 64, "num_experts_per_tok": 4,
    },
}

# Ops that are model-independent (pure network/hardware).
MODEL_INDEPENDENT_OPS = {"allreduce", "allgather", "reduce_scatter", "send_recv_p2p"}

# Features for model-independent ops (no architecture features needed).
FEATURES_NETWORK = {
    "A-Ridge": ["log_message_size_bytes", "log_num_gpus"],
    "B-GBR": ["num_gpus", "message_size_bytes", "log_num_gpus", "log_message_size_bytes"],
    "D-Poly2Ridge": ["log_message_size_bytes", "log_num_gpus"],
    "E-PowerLaw": ["log_message_size_bytes", "log_num_gpus"],
    "F-RF": ["num_gpus", "message_size_bytes", "log_num_gpus", "log_message_size_bytes"],
}

# Features for model-dependent ops (include architecture).
FEATURES_MODEL_DEP = {
    "A-Ridge": [
        "log_message_size_bytes", "log_num_gpus", "log_hidden_size",
        "log_num_hidden_layers",
    ],
    "B-GBR": [
        "num_gpus", "message_size_bytes", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "is_moe", "head_dim", "gqa_ratio",
        "log_num_gpus", "log_message_size_bytes", "log_hidden_size",
        "log_num_hidden_layers",
    ],
    "D-Poly2Ridge": [
        "log_message_size_bytes", "log_num_gpus", "log_hidden_size",
        "log_num_hidden_layers",
    ],
    "E-PowerLaw": [
        "log_message_size_bytes", "log_num_gpus", "log_hidden_size",
        "log_num_hidden_layers",
    ],
    "F-RF": [
        "num_gpus", "message_size_bytes", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "is_moe", "head_dim", "gqa_ratio",
        "log_num_gpus", "log_message_size_bytes", "log_hidden_size",
        "log_num_hidden_layers",
    ],
}

# Bandwidth-residual features (Approach C) — for ops with bandwidth_gb_s.
FEATURES_BW_RESIDUAL = ["log_message_size_bytes", "log_num_gpus"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Load communication.csv files from profiles directory."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*/*/*/communication.csv"))

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

    if not frames:
        print(f"ERROR: No communication.csv files found under {data_dir}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Drop rows with zero or negative time
    bad_mask = df["time_s"].isna() | (df["time_s"] <= 0)
    if bad_mask.any():
        dropped = bad_mask.sum()
        df = df[~bad_mask].reset_index(drop=True)
        print(f"  Dropped {dropped} rows with missing/non-positive time_s")

    print(f"\n=== Data Loading ===")
    print(f"  Loaded {len(df)} rows from {len(csv_files)} CSV files")
    if skipped:
        print(f"  Skipped {skipped} files")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Operations: {sorted(df['operation'].unique())}")
    print(f"  num_gpus: {sorted(df['num_gpus'].unique())}")
    msg_sizes = sorted(df["message_size_bytes"].unique())
    print(f"  message_size_bytes: {msg_sizes[:3]}...{msg_sizes[-1]} ({len(msg_sizes)} values)")
    print(f"  Time range: [{df['time_s'].min():.2e}, {df['time_s'].max():.2e}] s")

    return df


def add_architecture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge model architecture parameters."""
    arch_df = pd.DataFrame.from_dict(MODEL_ARCHITECTURES, orient="index")
    arch_df.index.name = "model"
    arch_df = arch_df.reset_index()
    n_before = len(df)
    df = df.merge(arch_df, on="model", how="left")
    missing = df["hidden_size"].isna().sum()
    if missing:
        print(f"  WARNING: {missing}/{n_before} rows missing architecture features")
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features."""
    df = df.copy()
    df["head_dim"] = df["hidden_size"] // df["num_attention_heads"]
    df["gqa_ratio"] = df["num_attention_heads"] / df["num_key_value_heads"]
    df["is_moe"] = (df["num_experts"] > 0).astype(float)
    df["bandwidth_gb_s"] = df["bandwidth_gb_s"].fillna(0.0)

    df["log_time_s"] = np.log(df["time_s"].clip(lower=1e-15))
    df["log_message_size_bytes"] = np.log(df["message_size_bytes"].astype(float).clip(lower=1.0))
    df["log_num_gpus"] = np.log(df["num_gpus"].astype(float).clip(lower=1.0))
    df["log_hidden_size"] = np.log(df["hidden_size"].astype(float))
    df["log_num_hidden_layers"] = np.log(df["num_hidden_layers"].astype(float))
    df["log_bandwidth_gb_s"] = np.log(df["bandwidth_gb_s"].clip(lower=1e-10))

    # Simple bandwidth-based roofline: time = message_size / bandwidth
    # Only valid where bandwidth > 0
    df["roofline_bw_time_s"] = np.where(
        df["bandwidth_gb_s"] > 0,
        (df["message_size_bytes"].astype(float) / 1e9) / df["bandwidth_gb_s"],
        np.nan,
    )

    return df


# ---------------------------------------------------------------------------
# EDA Plots
# ---------------------------------------------------------------------------


def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate EDA plots for communication data."""
    output_dir = Path(output_dir)
    df = df.copy()
    df["model_short"] = df["model"].str.replace("Qwen_Qwen3-", "Q3-").str.replace(
        "zai-org_GLM-4.7-Flash", "GLM-4.7"
    )

    operations = sorted(df["operation"].unique())
    model_names = sorted(df["model_short"].unique())
    palette = sns.color_palette("tab10", len(model_names))

    # --- Plot 1: Latency vs message_size grid ---
    n_ops = len(operations)
    ncols = 4
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, op in enumerate(operations):
        ax = axes[idx // ncols, idx % ncols]
        sub_op = df[(df["operation"] == op) & (df["num_gpus"] == 2)]
        for mi, model in enumerate(model_names):
            sub = sub_op[sub_op["model_short"] == model]
            if len(sub) > 0:
                ax.scatter(sub["message_size_bytes"], sub["time_s"], label=model,
                           s=15, alpha=0.7, color=palette[mi])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Message Size (bytes)")
        ax.set_ylabel("Time (s)")
        ax.set_title(op, fontsize=9)
        ax.grid(True, alpha=0.3)
    for idx in range(n_ops, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.suptitle("Communication Latency vs Message Size (2 GPUs)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "eda_comm_latency_vs_msgsize.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_comm_latency_vs_msgsize.png")

    # --- Plot 2: Latency vs num_gpus ---
    fig, axes = plt.subplots(1, min(4, n_ops), figsize=(5 * min(4, n_ops), 5), squeeze=False)
    for idx, op in enumerate(operations[:4]):
        ax = axes[0, idx]
        for mi, model in enumerate(model_names):
            sub = df[(df["operation"] == op) & (df["model_short"] == model) &
                     (df["message_size_bytes"] == df["message_size_bytes"].max())]
            if len(sub) > 0:
                ax.scatter(sub["num_gpus"], sub["time_s"], label=model, s=25,
                           alpha=0.7, color=palette[mi])
        ax.set_xlabel("Num GPUs")
        ax.set_ylabel("Time (s)")
        ax.set_title(op, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=5)
    fig.suptitle("Comm Latency vs Num GPUs (max message size)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "eda_comm_latency_vs_gpus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eda_comm_latency_vs_gpus.png")


# ---------------------------------------------------------------------------
# CV Helpers
# ---------------------------------------------------------------------------


def _cv_evaluate(
    model, X: np.ndarray, y: np.ndarray, kf: KFold,
    *, collect_preds: bool = False,
) -> dict:
    maes, r2s, mapes = [], [], []
    all_true, all_pred = [], []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        pred = np.exp(model.predict(X[te]))
        true = np.exp(y[te])
        maes.append(mean_absolute_error(true, pred))
        r2s.append(r2_score(true, pred))
        mapes.append(mean_absolute_percentage_error(true, pred))
        if collect_preds:
            all_true.extend(true)
            all_pred.extend(pred)
    result = {
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "mape_mean": float(np.mean(mapes)), "mape_std": float(np.std(mapes)),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
    return result


def _make_estimator(approach: str):
    if approach == "A-Ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if approach == "B-GBR":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            learning_rate=0.1, random_state=42,
        )
    if approach == "C-BW-Residual":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if approach == "D-Poly2Ridge":
        return Pipeline([
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0)),
        ])
    if approach == "E-PowerLaw":
        return LinearRegression()
    if approach == "F-RF":
        return RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42,
        )
    raise ValueError(f"Unknown approach: {approach}")


def _get_feature_dict(op_name: str) -> dict[str, list[str]]:
    if op_name in MODEL_INDEPENDENT_OPS:
        return FEATURES_NETWORK
    return FEATURES_MODEL_DEP


# ---------------------------------------------------------------------------
# Per-Operator Training
# ---------------------------------------------------------------------------


def train_models_for_operator(df_op: pd.DataFrame, op_name: str) -> dict:
    """Fit all approaches for one communication operator."""
    n = len(df_op)
    n_splits = min(10, max(2, n // 3))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y = df_op["log_time_s"].values
    feat_dict = _get_feature_dict(op_name)
    results: dict = {}

    approach_names = ["A-Ridge", "B-GBR", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]

    # Add bandwidth-residual if bandwidth is available
    has_bw = (df_op["bandwidth_gb_s"] > 0).sum() > n // 2

    for approach in approach_names:
        feat_list = feat_dict.get(approach, [])
        avail = [f for f in feat_list if f in df_op.columns]
        if not avail:
            continue
        X = df_op[avail].values
        est = _make_estimator(approach)
        cv = _cv_evaluate(est, X, y, kf, collect_preds=True)
        est_final = _make_estimator(approach)
        est_final.fit(X, y)
        results[approach] = {"model": est_final, "features": avail, "cv": cv, "target": "log_time_s"}

    # Bandwidth-residual approach (C) — predict log(time / bw_roofline_time)
    if has_bw:
        valid = df_op["roofline_bw_time_s"].notna() & (df_op["roofline_bw_time_s"] > 0)
        df_c = df_op[valid]
        if len(df_c) >= 6:
            avail_c = [f for f in FEATURES_BW_RESIDUAL if f in df_c.columns]
            X_c = df_c[avail_c].values
            y_ratio = np.log((df_c["time_s"] / df_c["roofline_bw_time_s"]).clip(lower=1e-10).values)
            roofline_t = df_c["roofline_bw_time_s"].values
            actual_t = df_c["time_s"].values
            n_c = min(10, max(2, len(df_c) // 3))
            kf_c = KFold(n_splits=n_c, shuffle=True, random_state=42)

            # Manual CV for roofline residual
            maes, r2s, mapes = [], [], []
            for tr, te in kf_c.split(X_c):
                est_c = _make_estimator("C-BW-Residual")
                est_c.fit(X_c[tr], y_ratio[tr])
                pred_ratio = est_c.predict(X_c[te])
                pred_time = np.exp(pred_ratio) * roofline_t[te]
                maes.append(mean_absolute_error(actual_t[te], pred_time))
                r2s.append(r2_score(actual_t[te], pred_time))
                mapes.append(mean_absolute_percentage_error(actual_t[te], pred_time))

            cv_c = {
                "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
                "mape_mean": float(np.mean(mapes)), "mape_std": float(np.std(mapes)),
            }
            est_c_final = _make_estimator("C-BW-Residual")
            est_c_final.fit(X_c, y_ratio)
            results["C-BW-Residual"] = {
                "model": est_c_final, "features": avail_c,
                "cv": cv_c, "target": "log_ratio",
            }

    return results


def train_all_operators(df: pd.DataFrame) -> dict[str, dict]:
    """Train surrogates for all communication operators."""
    print(f"\n=== Per-Operator Cross-Validation ===")

    operations = sorted(df["operation"].unique())
    all_results: dict[str, dict] = {}
    approach_names = ["A-Ridge", "B-GBR", "C-BW-Residual", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]

    print(f"  {'Operation':<28s} {'N':>5s}", end="")
    for a in approach_names:
        print(f"  {a:>14s}", end="")
    print()
    print(f"  {'-'*28} {'-'*5}" + f"  {'-'*14}" * len(approach_names))

    for op in operations:
        df_op = df[df["operation"] == op].reset_index(drop=True)
        n = len(df_op)
        if n < 6:
            print(f"  {op:<28s} {n:>5d}  SKIPPED")
            continue

        results = train_models_for_operator(df_op, op)
        all_results[op] = results

        parts = [f"  {op:<28s} {n:>5d}"]
        for a in approach_names:
            if a in results:
                parts.append(f"  {results[a]['cv']['mape_mean']:>13.4f}")
            else:
                parts.append(f"  {'N/A':>14s}")
        print("".join(parts))

    return all_results


# ---------------------------------------------------------------------------
# Leave-One-Model-Out CV
# ---------------------------------------------------------------------------


def leave_one_model_out_cv(df: pd.DataFrame, output_dir: str | Path) -> dict:
    """LOMO CV per operator per approach."""
    output_dir = Path(output_dir)
    print(f"\n=== Leave-One-Model-Out Cross-Validation ===")

    operations = sorted(df["operation"].unique())
    logo = LeaveOneGroupOut()
    lomo_rows: list[dict] = []
    approach_names = ["A-Ridge", "B-GBR", "C-BW-Residual", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]

    for op in operations:
        df_op = df[df["operation"] == op].reset_index(drop=True)
        if len(df_op) < 6 or len(df_op["model"].unique()) < 2:
            continue

        feat_dict = _get_feature_dict(op)
        y = df_op["log_time_s"].values
        groups = df_op["model"].values
        has_bw = (df_op["bandwidth_gb_s"] > 0).sum() > len(df_op) // 2

        for approach in approach_names:
            if approach == "C-BW-Residual":
                if not has_bw:
                    continue
                valid = df_op["roofline_bw_time_s"].notna() & (df_op["roofline_bw_time_s"] > 0)
                df_c = df_op[valid].reset_index(drop=True)
                if len(df_c) < 6 or len(df_c["model"].unique()) < 2:
                    continue
                avail = [f for f in FEATURES_BW_RESIDUAL if f in df_c.columns]
                X = df_c[avail].values
                y_ratio = np.log((df_c["time_s"] / df_c["roofline_bw_time_s"]).clip(lower=1e-10).values)
                roofline_t = df_c["roofline_bw_time_s"].values
                actual_t = df_c["time_s"].values
                groups_c = df_c["model"].values

                for tr, te in logo.split(X, y_ratio, groups_c):
                    est = _make_estimator("C-BW-Residual")
                    est.fit(X[tr], y_ratio[tr])
                    pred = np.exp(est.predict(X[te])) * roofline_t[te]
                    true = actual_t[te]
                    lomo_rows.append({
                        "operator": op, "approach": approach,
                        "held_out": groups_c[te[0]], "n_test": len(te),
                        "mape": mean_absolute_percentage_error(true, pred),
                        "r2": r2_score(true, pred) if len(te) > 1 else float("nan"),
                        "mae": mean_absolute_error(true, pred),
                    })
                continue

            feat_list = feat_dict.get(approach, [])
            avail = [f for f in feat_list if f in df_op.columns]
            if not avail:
                continue
            X = df_op[avail].values

            for tr, te in logo.split(X, y, groups):
                est = _make_estimator(approach)
                est.fit(X[tr], y[tr])
                pred = np.exp(est.predict(X[te]))
                true = np.exp(y[te])
                lomo_rows.append({
                    "operator": op, "approach": approach,
                    "held_out": groups[te[0]], "n_test": len(te),
                    "mape": mean_absolute_percentage_error(true, pred),
                    "r2": r2_score(true, pred) if len(te) > 1 else float("nan"),
                    "mae": mean_absolute_error(true, pred),
                })

    lomo_df = pd.DataFrame(lomo_rows)
    if len(lomo_df) == 0:
        print("  No LOMO results.")
        return {"lomo_df": lomo_df}

    # Summary table
    print(f"\n  LOMO CV — Mean MAPE per Operator x Approach")
    header = f"  {'Operation':<28s}"
    for a in approach_names:
        header += f"  {a:>14s}"
    print(header)
    print(f"  {'-'*28}" + f"  {'-'*14}" * len(approach_names))
    for op in sorted(lomo_df["operator"].unique()):
        row = f"  {op:<28s}"
        for a in approach_names:
            sub = lomo_df[(lomo_df["operator"] == op) & (lomo_df["approach"] == a)]
            row += f"  {sub['mape'].mean():>14.4f}" if len(sub) > 0 else f"  {'N/A':>14s}"
        print(row)

    print(f"\n  LOMO CV — Overall Aggregate")
    print(f"  {'Approach':<16s} {'Mean MAPE':>10s} {'Mean R²':>10s}")
    print(f"  {'-'*16} {'-'*10} {'-'*10}")
    for a in approach_names:
        sub = lomo_df[lomo_df["approach"] == a]
        if len(sub) > 0:
            print(f"  {a:<16s} {sub['mape'].mean():>10.4f} {sub['r2'].mean():>10.4f}")

    # LOMO plots
    for op in sorted(lomo_df["operator"].unique()):
        sub_op = lomo_df[lomo_df["operator"] == op]
        best = sub_op.groupby("approach")["mape"].mean().idxmin()
        df_op = df[df["operation"] == op].reset_index(drop=True)
        feat_dict = _get_feature_dict(op)
        feat_list = feat_dict.get(best, [])
        avail = [f for f in feat_list if f in df_op.columns]
        if not avail:
            continue
        X = df_op[avail].values
        y_op = df_op["log_time_s"].values
        groups_op = df_op["model"].values

        all_t, all_p, all_m = [], [], []
        for tr, te in logo.split(X, y_op, groups_op):
            est = _make_estimator(best)
            est.fit(X[tr], y_op[tr])
            pred = np.exp(est.predict(X[te]))
            true = np.exp(y_op[te])
            all_t.extend(true)
            all_p.extend(pred)
            all_m.extend([groups_op[te[0]]] * len(te))

        if not all_t:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        t_arr, p_arr, m_arr = np.array(all_t), np.array(all_p), np.array(all_m)
        for ml in sorted(set(all_m)):
            mask = m_arr == ml
            short = ml.replace("Qwen_Qwen3-", "Q3-").replace("zai-org_GLM-4.7-Flash", "GLM-4.7")
            ax.scatter(t_arr[mask], p_arr[mask], label=short, s=15, alpha=0.7)
        lo, hi = min(t_arr.min(), p_arr.min()) * 0.5, max(t_arr.max(), p_arr.max()) * 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("Actual Time (s)")
        ax.set_ylabel("Predicted Time (s)")
        ax.set_title(f"LOMO: {op} ({best})")
        ax.legend(fontsize=6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"lomo_comm_{op}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved LOMO plots for {len(lomo_df['operator'].unique())} operators")
    return {"lomo_df": lomo_df}


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def feature_importance(all_results: dict[str, dict], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    gbr_rows: list[dict] = []
    for op, results in all_results.items():
        if "B-GBR" in results:
            features = results["B-GBR"]["features"]
            importances = results["B-GBR"]["model"].feature_importances_
            for f, imp in zip(features, importances):
                gbr_rows.append({"operator": op, "feature": f, "importance": float(imp)})

    gbr_df = pd.DataFrame(gbr_rows)
    print(f"\n=== Feature Importance (GBR) ===")
    for op in sorted(gbr_df["operator"].unique()):
        sub = gbr_df[gbr_df["operator"] == op].sort_values("importance", ascending=False)
        top3 = ", ".join(f"{r['feature']}({r['importance']:.3f})" for _, r in sub.head(3).iterrows())
        print(f"  {op:<28s}: {top3}")

    if len(gbr_df) > 0:
        pivot = gbr_df.pivot_table(index="feature", columns="operator", values="importance",
                                   aggfunc="mean").fillna(0)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title("GBR Feature Importance — Communication Operators")
        fig.tight_layout()
        fig.savefig(output_dir / "comm_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved comm_feature_importance.png")

    return {"gbr_importances": gbr_df}


# ---------------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------------


def save_artifacts(
    df: pd.DataFrame, all_results: dict, lomo_results: dict,
    importance_results: dict, output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    csv_cols = [
        "operation", "model", "gpu", "quant", "num_gpus", "message_size_bytes",
        "time_s", "bandwidth_gb_s",
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "num_experts", "num_experts_per_tok",
        "is_moe", "head_dim", "gqa_ratio",
        "log_time_s", "log_message_size_bytes", "log_num_gpus",
    ]
    avail = [c for c in csv_cols if c in df.columns]
    csv_path = output_dir / "communication_latency_data.csv"
    df[avail].to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path} ({len(df)} rows)")

    def _safe(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    op_summary = {}
    for op, res in all_results.items():
        op_summary[op] = {
            a: {k: _safe(v) for k, v in d["cv"].items() if k not in ("all_true", "all_pred")}
            for a, d in res.items() if "cv" in d
        }

    lomo_df = lomo_results.get("lomo_df", pd.DataFrame())
    summary = {
        "n_rows": len(df),
        "operations": sorted(df["operation"].unique().tolist()),
        "per_operator_cv": op_summary,
        "lomo_cv": [{k: _safe(v) for k, v in r.items()} for _, r in lomo_df.iterrows()] if len(lomo_df) > 0 else [],
    }
    json_path = output_dir / "communication_latency_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_safe)
    print(f"  Saved {json_path}")

    # Print summary
    approach_names = ["A-Ridge", "B-GBR", "C-BW-Residual", "D-Poly2Ridge", "E-PowerLaw", "F-RF"]
    print(f"\n{'='*80}")
    print(f"  Communication Latency Surrogate — Summary")
    print(f"{'='*80}")
    print(f"  Data: {len(df)} rows, {len(df['operation'].unique())} operations")
    print(f"\n  K-Fold CV (MAPE):")
    for op in sorted(all_results.keys()):
        print(f"  {op:<28s}", end="")
        for a in approach_names:
            if a in all_results[op]:
                print(f"  {all_results[op][a]['cv']['mape_mean']:>12.4f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()
    if len(lomo_df) > 0:
        print(f"\n  LOMO CV:")
        for a in approach_names:
            sub = lomo_df[lomo_df["approach"] == a]
            if len(sub) > 0:
                print(f"    {a:<16s}: mean MAPE={sub['mape'].mean():.4f}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_dir: str | Path, output_dir: str | Path | None = None,
    operators: list[str] | None = None, exclude_models: list[str] | None = None,
) -> dict:
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir.parent / "communication_latency_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  Communication Latency Surrogate Model")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")

    df = load_data(data_dir)
    if len(df) == 0:
        sys.exit(1)

    if exclude_models:
        for p in exclude_models:
            before = len(df)
            df = df[~df["model"].str.lower().str.contains(p.lower())]
            if before - len(df):
                print(f"  Filtered {before - len(df)} rows matching '{p}'")
        df = df.reset_index(drop=True)

    if operators:
        df = df[df["operation"].isin(operators)].reset_index(drop=True)
        print(f"  Filtered to: {sorted(df['operation'].unique())}")

    df = add_architecture_features(df)
    df = compute_derived_features(df)

    print(f"\n=== EDA Plots ===")
    generate_eda_plots(df, output_dir)

    all_results = train_all_operators(df)
    lomo_results = leave_one_model_out_cv(df, output_dir)
    importance_results = feature_importance(all_results, output_dir)
    save_artifacts(df, all_results, lomo_results, importance_results, output_dir)

    return {"df": df, "all_results": all_results, "lomo_results": lomo_results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Communication latency surrogate")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--operators", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-models", type=str, nargs="+", default=None)
    args = parser.parse_args(argv)
    run_pipeline(args.data_dir, args.output_dir, args.operators, args.exclude_models)
    print("\nDone.")


if __name__ == "__main__":
    main()
