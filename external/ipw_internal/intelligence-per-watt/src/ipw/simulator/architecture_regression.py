"""Surrogate Model Analysis: Architecture Config → Energy/Power.

Builds surrogate models that predict avg power (W) and avg energy (J)
from neural-network architecture configurations.  Compares Ridge,
Polynomial Ridge, Gaussian Process, and Gradient Boosting regressors,
extracts interpretable equations, and produces diagnostic plots.

Usage:
    python -m ipw.simulator.model evolutionary_results.json --target avg_power
    python -m ipw.simulator.model evolutionary_results.json --target avg_energy
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_FEATURES = ["d_model", "n_layers", "n_kv_heads", "weight_quant", "kv_quant"]
ENGINEERED_FEATURES = RAW_FEATURES + [
    "d_model_sq",
    "d_model_x_layers",
    "compute_proxy",
    "total_params_approx",
    "total_bw_proxy",
]

HEAD_DIM = 128
FFN_MULT = 3.5

# ---------------------------------------------------------------------------
# Step 1: Parse Data
# ---------------------------------------------------------------------------


def load_data(path: str | Path) -> pd.DataFrame:
    """Load *evolutionary_results.json* into a tidy DataFrame."""
    with open(path) as f:
        raw = json.load(f)

    rows: list[dict] = []
    for config_hash, entry in raw.items():
        arch = entry["architecture_config"]
        metrics = entry["metrics"]

        wq = 1 if arch["weight_quant"] == "8bit" else 0
        kq = 1 if arch["kv_quant"] == "fp16" else 0

        rows.append(
            {
                "config_hash": config_hash,
                "d_model": int(arch["d_model"]),
                "n_layers": int(arch["n_layers"]),
                "n_kv_heads": int(arch["n_kv_heads"]),
                "weight_quant": wq,
                "kv_quant": kq,
                "avg_power": float(metrics["power"]["avg"]),
                "avg_energy": float(metrics["energy"]["avg"]),
                "avg_latency": float(metrics["latency"]["avg"]),
            }
        )

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, target: str) -> None:
    """Print descriptive statistics for features and target."""
    cols = RAW_FEATURES + [target]
    if "avg_latency" in df.columns:
        cols.append("avg_latency")
    print("\n=== Dataset Summary ===")
    print(f"  N = {len(df)}")
    print(df[cols].describe().round(3).to_string())
    print()


# ---------------------------------------------------------------------------
# Step 2: Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns in-place and return *df*."""
    df = df.copy()
    df["d_model_sq"] = df["d_model"] ** 2
    df["d_model_x_layers"] = df["d_model"] * df["n_layers"]
    df["compute_proxy"] = df["d_model"] ** 2 * df["n_layers"]

    df["attn_params_per_layer"] = df["d_model"] * (
        df["d_model"] + 2 * df["n_kv_heads"] * HEAD_DIM + df["d_model"]
    )
    df["ffn_params_per_layer"] = 3 * df["d_model"] * (df["d_model"] * FFN_MULT).astype(int)
    df["total_params_approx"] = df["n_layers"] * (
        df["attn_params_per_layer"] + df["ffn_params_per_layer"]
    )

    df["weight_bw_mult"] = df["weight_quant"].map({0: 0.5, 1: 1.0})
    df["kv_bw_mult"] = df["kv_quant"].map({0: 0.5, 1: 1.0})
    df["total_bw_proxy"] = df["compute_proxy"] * df["weight_bw_mult"] * df["kv_bw_mult"]

    return df


# ---------------------------------------------------------------------------
# Step 3: Train & Compare Models (10-fold CV)
# ---------------------------------------------------------------------------


def _cv_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    *,
    collect_preds: bool = False,
) -> dict:
    """Run cross-validation and return metric dict."""
    maes, rmses, r2s = [], [], []
    all_true, all_pred, all_std = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)

        if hasattr(model, "predict") and "return_std" in model.predict.__code__.co_varnames:
            y_hat, y_std = model.predict(X_te, return_std=True)
        else:
            y_hat = model.predict(X_te)
            y_std = np.zeros_like(y_hat)

        maes.append(mean_absolute_error(y_te, y_hat))
        rmses.append(np.sqrt(mean_squared_error(y_te, y_hat)))
        r2s.append(r2_score(y_te, y_hat))

        if collect_preds:
            all_true.extend(y_te)
            all_pred.extend(y_hat)
            all_std.extend(y_std)

    result = {
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
        result["all_std"] = np.array(all_std)
    return result


def _is_gp(model) -> bool:
    """Check if model is (or wraps) a GaussianProcessRegressor."""
    if isinstance(model, GaussianProcessRegressor):
        return True
    if isinstance(model, Pipeline):
        return isinstance(model.steps[-1][1], GaussianProcessRegressor)
    return False


def _cv_evaluate_gp(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    *,
    collect_preds: bool = False,
) -> dict:
    """GP-aware cross-validation that calls predict(return_std=True) correctly."""
    maes, rmses, r2s = [], [], []
    all_true, all_pred, all_std = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)

        # For a Pipeline ending in GP, we must manually transform + predict
        if isinstance(pipeline, Pipeline):
            scaler = pipeline.named_steps["scaler"]
            gp = pipeline.named_steps["gp"]
            X_te_sc = scaler.transform(X_te)
            y_hat, y_std = gp.predict(X_te_sc, return_std=True)
        else:
            y_hat, y_std = pipeline.predict(X_te, return_std=True)

        maes.append(mean_absolute_error(y_te, y_hat))
        rmses.append(np.sqrt(mean_squared_error(y_te, y_hat)))
        r2s.append(r2_score(y_te, y_hat))

        if collect_preds:
            all_true.extend(y_te)
            all_pred.extend(y_hat)
            all_std.extend(y_std)

    result = {
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
    }
    if collect_preds:
        result["all_true"] = np.array(all_true)
        result["all_pred"] = np.array(all_pred)
        result["all_std"] = np.array(all_std)
    return result


def build_models(n_eng_features: int) -> dict:
    """Return {name: (model, feature_set_key)} for each model."""
    kernel = (
        ConstantKernel(1.0)
        * Matern(nu=2.5, length_scale=np.ones(n_eng_features))
        + WhiteKernel(noise_level=0.1)
    )

    return {
        "Ridge (raw)": (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            "raw",
        ),
        "Ridge (engineered)": (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            "engineered",
        ),
        "Ridge+Poly2 (raw)": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=10.0)),
            ]),
            "raw",
        ),
        "GP (Matérn)": (
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "gp",
                    GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=20,
                        normalize_y=True,
                        alpha=1e-6,
                        random_state=42,
                    ),
                ),
            ]),
            "engineered",
        ),
        "GBR": (
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=5,
                random_state=42,
            ),
            "engineered",
        ),
    }


def run_cv(
    models: dict,
    X_raw: np.ndarray,
    X_eng: np.ndarray,
    y: np.ndarray,
) -> dict:
    """10-fold CV for every model.  Returns {name: metrics_dict}."""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results: dict = {}

    for name, (model, feat_key) in models.items():
        X = X_raw if feat_key == "raw" else X_eng
        print(f"  CV: {name} …", end=" ", flush=True)
        if _is_gp(model):
            res = _cv_evaluate_gp(model, X, y, kf, collect_preds=(name == "GP (Matérn)"))
        else:
            res = _cv_evaluate(model, X, y, kf, collect_preds=False)
        results[name] = res
        print(
            f"R²={res['r2_mean']:.4f}±{res['r2_std']:.4f}  "
            f"MAE={res['mae_mean']:.3f}±{res['mae_std']:.3f}  "
            f"RMSE={res['rmse_mean']:.3f}±{res['rmse_std']:.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Step 4: Feature Importance
# ---------------------------------------------------------------------------


def ridge_importances(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Fit Ridge on scaled data; return standardized coefficients."""
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    coefs = pipe.named_steps["ridge"].coef_
    imp = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    imp["abs_coef"] = imp["coefficient"].abs()
    return imp.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def gbr_importances(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Fit GBR; return feature importances."""
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    )
    gbr.fit(X, y)
    imp = pd.DataFrame({"feature": feature_names, "importance": gbr.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def gp_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features: int,
) -> tuple[pd.DataFrame, dict]:
    """Fit GP on scaled data and extract kernel hyper-parameters."""
    kernel = (
        ConstantKernel(1.0)
        * Matern(nu=2.5, length_scale=np.ones(n_features))
        + WhiteKernel(noise_level=0.1)
    )
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        normalize_y=True,
        alpha=1e-6,
        random_state=42,
    )
    gp.fit(X_sc, y)

    k = gp.kernel_
    amplitude = k.k1.k1.constant_value
    length_scales = k.k1.k2.length_scale
    noise = k.k2.noise_level

    sensitivity = 1.0 / length_scales
    sens_df = pd.DataFrame({"feature": feature_names, "length_scale": length_scales, "sensitivity": sensitivity})
    sens_df = sens_df.sort_values("sensitivity", ascending=False).reset_index(drop=True)

    hypers = {"amplitude": amplitude, "noise": noise, "scaler": scaler, "gp": gp}
    return sens_df, hypers


# ---------------------------------------------------------------------------
# Step 5: Ridge + Poly2 Equation (unscaled)
# ---------------------------------------------------------------------------


def extract_poly_equation(
    X_raw: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_name: str,
) -> pd.DataFrame:
    """Fit Ridge+Poly2 WITHOUT scaling so coefficients are in original units."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    poly_names = poly.get_feature_names_out(feature_names)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_poly, y)

    # Compute swing = |coef| * (max - min) for each poly term
    term_ranges = X_poly.max(axis=0) - X_poly.min(axis=0)
    swings = np.abs(ridge.coef_) * term_ranges

    eq_df = pd.DataFrame({
        "term": poly_names,
        "coefficient": ridge.coef_,
        "swing": swings,
    }).sort_values("swing", ascending=False).reset_index(drop=True)

    print(f"\n=== Ridge+Poly2 Equation ({target_name}) ===")
    print(f"  {target_name} = {ridge.intercept_:.4f}")
    for _, row in eq_df.iterrows():
        sign = "+" if row["coefficient"] >= 0 else "-"
        print(f"    {sign} {abs(row['coefficient']):.6e} × {row['term']}  (swing={row['swing']:.3f})")
    print()

    return eq_df


# ---------------------------------------------------------------------------
# Step 6: GP Marginal Effects
# ---------------------------------------------------------------------------


def _build_engineered_row(
    d_model: int,
    n_layers: int,
    n_kv_heads: int,
    weight_quant: int,
    kv_quant: int,
) -> np.ndarray:
    """Construct a single engineered-feature row from raw config."""
    d_model_sq = d_model ** 2
    d_model_x_layers = d_model * n_layers
    compute_proxy = d_model ** 2 * n_layers

    attn = d_model * (d_model + 2 * n_kv_heads * HEAD_DIM + d_model)
    ffn = 3 * d_model * int(d_model * FFN_MULT)
    total_params = n_layers * (attn + ffn)

    wbw = 0.5 if weight_quant == 0 else 1.0
    kbw = 0.5 if kv_quant == 0 else 1.0
    total_bw = compute_proxy * wbw * kbw

    return np.array([
        d_model, n_layers, n_kv_heads, weight_quant, kv_quant,
        d_model_sq, d_model_x_layers, compute_proxy, total_params, total_bw,
    ], dtype=float)


def gp_predict_single(
    d_model: int,
    n_layers: int,
    n_kv_heads: int,
    weight_quant: int,
    kv_quant: int,
    scaler: StandardScaler,
    gp: GaussianProcessRegressor,
) -> tuple[float, float]:
    """Return (mean, std) from the fitted GP."""
    row = _build_engineered_row(d_model, n_layers, n_kv_heads, weight_quant, kv_quant)
    row_sc = scaler.transform(row.reshape(1, -1))
    mean, std = gp.predict(row_sc, return_std=True)
    return float(mean[0]), float(std[0])


def marginal_effects(
    scaler: StandardScaler,
    gp: GaussianProcessRegressor,
    target_name: str,
) -> pd.DataFrame:
    """Compute GP marginal effects from a mid-range baseline."""
    baseline = dict(d_model=3072, n_layers=28, n_kv_heads=4, weight_quant=0, kv_quant=0)
    base_mean, base_std = gp_predict_single(**baseline, scaler=scaler, gp=gp)

    changes = [
        ("d_model 3072→4096", dict(d_model=4096)),
        ("d_model 3072→5120", dict(d_model=5120)),
        ("n_layers 28→36", dict(n_layers=36)),
        ("n_layers 28→48", dict(n_layers=48)),
        ("weight_quant 4bit→8bit", dict(weight_quant=1)),
        ("kv_quant int8→fp16", dict(kv_quant=1)),
        ("n_kv_heads 4→8", dict(n_kv_heads=8)),
    ]

    rows = [
        {
            "change": "baseline",
            "predicted": base_mean,
            "std": base_std,
            "ci_lo": base_mean - 1.96 * base_std,
            "ci_hi": base_mean + 1.96 * base_std,
            "delta": 0.0,
        }
    ]
    for label, overrides in changes:
        cfg = {**baseline, **overrides}
        m, s = gp_predict_single(**cfg, scaler=scaler, gp=gp)
        rows.append(
            {
                "change": label,
                "predicted": m,
                "std": s,
                "ci_lo": m - 1.96 * s,
                "ci_hi": m + 1.96 * s,
                "delta": m - base_mean,
            }
        )

    me_df = pd.DataFrame(rows)

    print(f"\n=== GP Marginal Effects ({target_name}) ===")
    for _, r in me_df.iterrows():
        print(
            f"  {r['change']:30s}  "
            f"pred={r['predicted']:8.3f} ± {1.96*r['std']:.3f}  "
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
            f"Δ={r['delta']:+.3f}"
        )
    print()

    return me_df


# ---------------------------------------------------------------------------
# Step 7: Correlation Analysis (energy only)
# ---------------------------------------------------------------------------


def correlation_analysis(df: pd.DataFrame) -> None:
    """Print power / energy / latency correlations."""
    print("\n=== Correlation Analysis ===")
    for a, b in [
        ("avg_energy", "avg_power"),
        ("avg_energy", "avg_latency"),
        ("avg_power", "avg_latency"),
    ]:
        if a in df.columns and b in df.columns:
            r = df[a].corr(df[b])
            print(f"  corr({a}, {b}) = {r:.4f}")
    print()


# ---------------------------------------------------------------------------
# Step 8: Diagnostic Plots
# ---------------------------------------------------------------------------


def _bar_chart(ax, names, means, stds, title, ylabel, fmt=".4f"):
    """Draw a bar chart with error bars and value labels."""
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=sns.color_palette("muted", len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def generate_plots(
    df: pd.DataFrame,
    cv_results: dict,
    ridge_imp: pd.DataFrame,
    gbr_imp: pd.DataFrame,
    gp_cv_true: np.ndarray | None,
    gp_cv_pred: np.ndarray | None,
    gp_cv_std: np.ndarray | None,
    target: str,
    output_path: str | Path,
) -> None:
    """Create the 3×3 diagnostic plot grid."""
    unit = "W" if target == "avg_power" else "J"
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f"Surrogate Model Diagnostics — {target}", fontsize=14, y=0.98)

    model_names = list(cv_results.keys())

    # --- Row 1 ---
    # (1) R² bar chart
    r2_means = [cv_results[n]["r2_mean"] for n in model_names]
    r2_stds = [cv_results[n]["r2_std"] for n in model_names]
    _bar_chart(axes[0, 0], model_names, r2_means, r2_stds, "R² (10-fold CV)", "R²")

    # (2) MAE bar chart
    mae_means = [cv_results[n]["mae_mean"] for n in model_names]
    mae_stds = [cv_results[n]["mae_std"] for n in model_names]
    _bar_chart(axes[0, 1], model_names, mae_means, mae_stds, f"MAE (10-fold CV)", f"MAE ({unit})", fmt=".3f")

    # (3) GP Actual vs Predicted
    ax = axes[0, 2]
    if gp_cv_true is not None:
        ax.errorbar(
            gp_cv_true,
            gp_cv_pred,
            yerr=1.96 * gp_cv_std,
            fmt="o",
            markersize=3,
            alpha=0.6,
            elinewidth=0.5,
            label="GP pred ± 95% CI",
        )
        lo = min(gp_cv_true.min(), gp_cv_pred.min()) * 0.95
        hi = max(gp_cv_true.max(), gp_cv_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.set_xlabel(f"Actual {target} ({unit})")
        ax.set_ylabel(f"Predicted {target} ({unit})")
        ax.legend(fontsize=7)
    ax.set_title("GP: Actual vs Predicted")

    # --- Row 2 ---
    # (4) GP Residual plot
    ax = axes[1, 0]
    if gp_cv_true is not None:
        residuals = gp_cv_true - gp_cv_pred
        ax.scatter(gp_cv_pred, residuals, s=12, alpha=0.6)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel(f"Predicted {target} ({unit})")
        ax.set_ylabel(f"Residual ({unit})")
    ax.set_title("GP: Residuals")

    # (5) Ridge Feature Coefficients
    ax = axes[1, 1]
    colors = ["green" if c >= 0 else "red" for c in ridge_imp["coefficient"]]
    ax.barh(ridge_imp["feature"], ridge_imp["coefficient"], color=colors)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Ridge (engineered) Coefficients")
    ax.invert_yaxis()

    # (6) GBR Feature Importance
    ax = axes[1, 2]
    ax.barh(gbr_imp["feature"], gbr_imp["importance"], color=sns.color_palette("viridis", len(gbr_imp)))
    ax.set_xlabel("Feature Importance")
    ax.set_title("Gradient Boosting Importance")
    ax.invert_yaxis()

    # --- Row 3 ---
    # (7) Target vs d_model by weight_quant
    ax = axes[2, 0]
    for wq, label, marker in [(0, "4bit", "o"), (1, "8bit", "^")]:
        subset = df[df["weight_quant"] == wq]
        ax.scatter(subset["d_model"], subset[target], label=label, marker=marker, s=20, alpha=0.7)
    ax.set_xlabel("d_model")
    ax.set_ylabel(f"{target} ({unit})")
    ax.set_title(f"{target} vs d_model")
    ax.legend(title="weight_quant", fontsize=7)

    # (8) Target vs n_layers by kv_quant
    ax = axes[2, 1]
    for kq, label, marker in [(0, "int8", "o"), (1, "fp16", "^")]:
        subset = df[df["kv_quant"] == kq]
        ax.scatter(subset["n_layers"], subset[target], label=label, marker=marker, s=20, alpha=0.7)
    ax.set_xlabel("n_layers")
    ax.set_ylabel(f"{target} ({unit})")
    ax.set_title(f"{target} vs n_layers")
    ax.legend(title="kv_quant", fontsize=7)

    # (9) Conditional on target
    ax = axes[2, 2]
    if target == "avg_power":
        sc = ax.scatter(
            df["compute_proxy"],
            df[target],
            c=df["total_bw_proxy"],
            cmap="viridis",
            s=20,
            alpha=0.7,
        )
        ax.set_xlabel("Compute Proxy")
        ax.set_ylabel(f"{target} ({unit})")
        ax.set_title(f"{target} vs Compute Proxy")
        plt.colorbar(sc, ax=ax, label="Bandwidth Proxy")
    else:
        sc = ax.scatter(
            df["avg_power"],
            df[target],
            c=df["avg_latency"],
            cmap="plasma",
            s=20,
            alpha=0.7,
        )
        ax.set_xlabel("avg_power (W)")
        ax.set_ylabel(f"{target} ({unit})")
        ax.set_title(f"{target} vs Power (colored by Latency)")
        plt.colorbar(sc, ax=ax, label="Latency (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 9: Sample Efficiency / Learning Curves
# ---------------------------------------------------------------------------


def _diverse_train_test_split(
    X_raw: np.ndarray,
    X_eng: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split that guarantees the test set covers all feature values
    and the target-range extremes.

    Strategy:
      1. For each raw feature, ensure at least one sample with each unique
         value is placed in the test set.
      2. Include at least one sample from the bottom and top 5% of the target.
      3. Fill remaining test slots randomly from the leftover pool.

    Returns (X_raw_train, X_raw_test, X_eng_train, X_eng_test, y_train, y_test).
    """
    n = len(y)
    n_test = max(1, int(n * test_frac))
    rng = np.random.RandomState(random_state)

    test_idx_set: set[int] = set()

    # 1. Coverage: every unique value of every raw feature appears in test
    n_raw_feats = X_raw.shape[1]
    for col in range(n_raw_feats):
        vals = X_raw[:, col]
        for v in np.unique(vals):
            candidates = np.where(vals == v)[0]
            if not (set(candidates) & test_idx_set):
                test_idx_set.add(int(rng.choice(candidates)))

    # 2. Target-range extremes: bottom and top sample if not yet covered
    sorted_idx = np.argsort(y)
    for idx in sorted_idx[:3]:
        if int(idx) not in test_idx_set:
            test_idx_set.add(int(idx))
            break
    for idx in sorted_idx[-3:]:
        if int(idx) not in test_idx_set:
            test_idx_set.add(int(idx))
            break

    # 3. Fill remaining test slots randomly
    remaining = sorted(set(range(n)) - test_idx_set)
    n_more = n_test - len(test_idx_set)
    if n_more > 0:
        extra = rng.choice(remaining, size=n_more, replace=False)
        test_idx_set.update(int(i) for i in extra)

    test_idx = np.array(sorted(test_idx_set))
    train_idx = np.array(sorted(set(range(n)) - test_idx_set))

    return (
        X_raw[train_idx], X_raw[test_idx],
        X_eng[train_idx], X_eng[test_idx],
        y[train_idx], y[test_idx],
    )


def _find_knee(
    sizes: np.ndarray,
    r2_means: np.ndarray,
    full_r2: float,
    tol: float = 0.01,
) -> int | None:
    """Return smallest training size where R² >= full_r2 - tol."""
    threshold = full_r2 - tol
    mask = r2_means >= threshold
    if not mask.any():
        return None
    return int(sizes[mask.argmax()])


def plot_learning_curves(
    results_df: pd.DataFrame,
    target: str,
    output_path: str | Path,
) -> None:
    """Plot R² and MAE learning curves with ±1 std bands."""
    models = results_df["model"].unique()
    palette = sns.color_palette("tab10", len(models))

    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Sample Efficiency — {target}", fontsize=13)

    for i, model_name in enumerate(models):
        sub = results_df[results_df["model"] == model_name]
        agg = sub.groupby("size").agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        ).reset_index()

        sizes = agg["size"].values
        color = palette[i]

        # R² subplot
        ax_r2.plot(sizes, agg["r2_mean"], marker=".", label=model_name, color=color)
        ax_r2.fill_between(
            sizes,
            agg["r2_mean"] - agg["r2_std"],
            agg["r2_mean"] + agg["r2_std"],
            alpha=0.15,
            color=color,
        )

        # Mark knee (tol=0.01)
        full_r2 = agg["r2_mean"].iloc[-1]
        knee = _find_knee(sizes, agg["r2_mean"].values, full_r2, tol=0.01)
        if knee is not None and knee < sizes[-1]:
            ax_r2.axvline(knee, color=color, ls="--", lw=0.8, alpha=0.7)

        # MAE subplot
        ax_mae.plot(sizes, agg["mae_mean"], marker=".", label=model_name, color=color)
        ax_mae.fill_between(
            sizes,
            agg["mae_mean"] - agg["mae_std"],
            agg["mae_mean"] + agg["mae_std"],
            alpha=0.15,
            color=color,
        )

    ax_r2.set_xlabel("Training Set Size")
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("R² vs Training Size")
    ax_r2.legend(fontsize=7)
    ax_r2.grid(True, alpha=0.3)

    ax_mae.set_xlabel("Training Set Size")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("MAE vs Training Size")
    ax_mae.legend(fontsize=7)
    ax_mae.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Learning curve plot saved to {output_path}")
    plt.close(fig)


def run_sample_efficiency(
    X_raw: np.ndarray,
    X_eng: np.ndarray,
    y: np.ndarray,
    target: str,
    output_dir: str | Path,
    *,
    test_frac: float = 0.2,
    n_trials: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Learning-curve analysis: how many configs are needed for a good model?

    Holds out a fixed test set, then sweeps training-set sizes with repeated
    random subsampling.  Returns a DataFrame of per-trial metrics and prints
    a summary table with knee points.
    """
    output_dir = Path(output_dir)

    # Fixed hold-out split with coverage guarantee
    X_raw_train, X_raw_test, X_eng_train, X_eng_test, y_train, y_test = (
        _diverse_train_test_split(
            X_raw, X_eng, y,
            test_frac=test_frac,
            random_state=random_state,
        )
    )

    n_train = len(y_train)
    n_features = X_eng.shape[1]
    min_size = max(10, n_features + 2)
    n_steps = 18
    sizes = np.unique(
        np.linspace(min_size, n_train, n_steps).astype(int)
    )

    n_eng = X_eng.shape[1]
    gp_min_size = 20

    print(f"\n=== Sample Efficiency Analysis ({target}) ===")
    print(f"  Hold-out test set: {len(y_test)},  training pool: {n_train}")

    # Report test-set coverage
    raw_feat_names = RAW_FEATURES[:X_raw.shape[1]]
    all_covered = True
    for i, fname in enumerate(raw_feat_names):
        full_vals = set(np.unique(X_raw[:, i]))
        test_vals = set(np.unique(X_raw_test[:, i]))
        missing = full_vals - test_vals
        if missing:
            print(f"  WARNING: test set missing {fname} = {sorted(missing)}")
            all_covered = False
    if all_covered:
        print(f"  Test set covers all unique values for each raw feature.")
    print(f"  Target range — full: [{y.min():.2f}, {y.max():.2f}], "
          f"test: [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"  Sizes: {sizes.tolist()}")
    print(f"  Trials per size: {n_trials}")

    records: list[dict] = []
    rng = np.random.RandomState(random_state)

    for size in sizes:
        print(f"  N={size:4d} ", end="", flush=True)
        for trial in range(n_trials):
            idx = rng.choice(n_train, size=size, replace=False)
            Xr_sub = X_raw_train[idx]
            Xe_sub = X_eng_train[idx]
            y_sub = y_train[idx]

            models = build_models(n_eng)
            for name, (model, feat_key) in models.items():
                # Skip GP when N is too small
                if _is_gp(model) and size < gp_min_size:
                    records.append({
                        "model": name, "size": size, "trial": trial,
                        "r2": np.nan, "mae": np.nan, "rmse": np.nan,
                    })
                    continue

                X_sub = Xr_sub if feat_key == "raw" else Xe_sub
                X_te = X_raw_test if feat_key == "raw" else X_eng_test

                model.fit(X_sub, y_sub)
                y_hat = model.predict(X_te)

                records.append({
                    "model": name,
                    "size": size,
                    "trial": trial,
                    "r2": r2_score(y_test, y_hat),
                    "mae": mean_absolute_error(y_test, y_hat),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_hat)),
                })
        print(".", flush=True)

    results_df = pd.DataFrame(records)

    # --- Summary table ---
    print(f"\n{'='*72}")
    print(f"  Sample Efficiency Summary — {target}")
    print(f"{'='*72}")
    header = (
        f"  {'Model':<22s}  {'Full R²':>8s}  "
        f"{'N (99% R²)':>11s}  {'N (95% R²)':>11s}"
    )
    print(header)
    print(f"  {'-'*22}  {'-'*8}  {'-'*11}  {'-'*11}")

    model_names = results_df["model"].unique()
    for name in model_names:
        sub = results_df[results_df["model"] == name]
        agg = sub.groupby("size")["r2"].mean()
        full_r2 = agg.iloc[-1]
        knee_99 = _find_knee(agg.index.values, agg.values, full_r2, tol=0.01)
        knee_95 = _find_knee(agg.index.values, agg.values, full_r2, tol=0.05)
        k99 = str(knee_99) if knee_99 is not None else "—"
        k95 = str(knee_95) if knee_95 is not None else "—"
        r2_str = f"{full_r2:.4f}" if not np.isnan(full_r2) else "NaN"
        print(f"  {name:<22s}  {r2_str:>8s}  {k99:>11s}  {k95:>11s}")
    print()

    # --- Learning curve plot ---
    plot_path = output_dir / f"learning_curves_{target}.png"
    plot_learning_curves(results_df, target, plot_path)

    return results_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_path: str | Path,
    target: str,
    output_dir: str | Path | None = None,
    *,
    sample_efficiency: bool = False,
) -> dict:
    """Run the full surrogate-model analysis for a single target.

    Parameters
    ----------
    data_path : path to evolutionary_results.json
    target    : ``'avg_power'`` or ``'avg_energy'``
    output_dir: directory for saved plots (defaults to parent of *data_path*)
    sample_efficiency : if True, run the learning-curve analysis (Step 9)

    Returns
    -------
    dict with cv_results, ridge_importances, gbr_importances,
    gp_sensitivity, poly_equation, marginal_effects DataFrames.
    """
    data_path = Path(data_path)
    if output_dir is None:
        output_dir = data_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Surrogate Model Analysis — target = {target}")
    print(f"{'='*60}")

    # Step 1
    df = load_data(data_path)
    print_summary(df, target)

    # Step 2
    df = engineer_features(df)

    X_raw = df[RAW_FEATURES].values
    X_eng = df[ENGINEERED_FEATURES].values
    y = df[target].values

    # Step 3
    n_eng = X_eng.shape[1]
    models = build_models(n_eng)
    print("\n=== 10-Fold Cross-Validation ===")
    cv_results = run_cv(models, X_raw, X_eng, y)

    # Step 4
    print("\n=== Feature Importances ===")
    ridge_imp = ridge_importances(X_eng, y, ENGINEERED_FEATURES)
    print("\nRidge (engineered) — standardized coefficients:")
    print(ridge_imp.to_string(index=False))

    gbr_imp = gbr_importances(X_eng, y, ENGINEERED_FEATURES)
    print("\nGradient Boosting — feature importances:")
    print(gbr_imp.to_string(index=False))

    sens_df, gp_hypers = gp_sensitivity(X_eng, y, ENGINEERED_FEATURES, n_eng)
    print("\nGP — kernel sensitivity (1/length_scale):")
    print(sens_df.to_string(index=False))
    print(f"  amplitude = {gp_hypers['amplitude']:.4f},  noise = {gp_hypers['noise']:.6f}")

    # Step 5
    poly_eq = extract_poly_equation(X_raw, y, RAW_FEATURES, target)

    # Step 6
    me_df = marginal_effects(gp_hypers["scaler"], gp_hypers["gp"], target)

    # Step 7
    if target == "avg_energy":
        correlation_analysis(df)

    # Step 8
    # Collect GP CV predictions for plotting
    gp_cv = cv_results.get("GP (Matérn)", {})
    gp_true = gp_cv.get("all_true")
    gp_pred = gp_cv.get("all_pred")
    gp_std = gp_cv.get("all_std")

    plot_path = output_dir / f"surrogate_diagnostics_{target}.png"
    generate_plots(df, cv_results, ridge_imp, gbr_imp, gp_true, gp_pred, gp_std, target, plot_path)

    # Step 9: Sample efficiency (optional)
    sample_eff_df = None
    if sample_efficiency:
        sample_eff_df = run_sample_efficiency(X_raw, X_eng, y, target, output_dir)

    return {
        "cv_results": cv_results,
        "ridge_importances": ridge_imp,
        "gbr_importances": gbr_imp,
        "gp_sensitivity": sens_df,
        "gp_hypers": gp_hypers,
        "poly_equation": poly_eq,
        "marginal_effects": me_df,
        "sample_efficiency": sample_eff_df,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Surrogate model analysis: architecture config → energy/power",
    )
    parser.add_argument("data", type=str, help="Path to evolutionary_results.json")
    parser.add_argument(
        "--target",
        choices=["avg_power", "avg_energy", "both"],
        default="both",
        help="Which target to model (default: both)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for plots")
    parser.add_argument(
        "--sample-efficiency",
        action="store_true",
        help="Run sample-efficiency / learning-curve analysis (slower)",
    )
    args = parser.parse_args(argv)

    targets = ["avg_power", "avg_energy"] if args.target == "both" else [args.target]

    for t in targets:
        run_pipeline(args.data, t, args.output_dir, sample_efficiency=args.sample_efficiency)

    print("\nDone.")


if __name__ == "__main__":
    main()
