"""Step 5: fit scaling-law regressions and held-out validations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FitResult:
    intercept: float
    coefficients: dict[str, float]
    r2: float | None
    y_pred: np.ndarray


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.astype(float))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def _fit_linear(X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> FitResult:
    X_design = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coeffs
    return FitResult(
        intercept=float(coeffs[0]),
        coefficients={name: float(value) for name, value in zip(names, coeffs[1:])},
        r2=_r2(y, y_pred),
        y_pred=y_pred,
    )


def _kfold_r2(X: np.ndarray, y: np.ndarray, names: Sequence[str], k: int, seed: int) -> dict[str, Any]:
    n = len(X)
    if n < 2:
        return {"folds": 0, "mean": None, "std": None, "scores": []}
    k = max(2, min(k, n))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    scores: list[float] = []
    for test_idx in folds:
        if test_idx.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        if train_idx.size < 2:
            continue
        result = _fit_linear(X[train_idx], y[train_idx], names)
        y_test_pred = np.column_stack([np.ones(len(test_idx)), X[test_idx]]) @ np.array(
            [result.intercept] + [result.coefficients[name] for name in names]
        )
        score = _r2(y[test_idx], y_test_pred)
        if score is not None and np.isfinite(score):
            scores.append(float(score))

    if not scores:
        return {"folds": k, "mean": None, "std": None, "scores": []}
    return {
        "folds": k,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }


def _interaction_features(base: pd.DataFrame) -> pd.DataFrame:
    result = base.copy()
    columns = list(base.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            left, right = columns[i], columns[j]
            result[f"{left}:{right}"] = base[left] * base[right]
    return result


def _build_prefill_features(frame: pd.DataFrame, cross_terms: bool) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "logP": _safe_log(frame["active_params_b"]),
            "logB": _safe_log(frame["batch_size"]),
            "logSin": _safe_log(frame["seq_in"]),
            "logQ": _safe_log(frame["bytes_per_param"]),
        }
    )
    return _interaction_features(base) if cross_terms else base


def _build_decode_features(frame: pd.DataFrame, cross_terms: bool) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "logP": _safe_log(frame["active_params_b"]),
            "logB": _safe_log(frame["batch_size"]),
            "logSout": _safe_log(frame["seq_out"]),
            "logSin": _safe_log(frame["seq_in"]),
            "logQ": _safe_log(frame["bytes_per_param"]),
        }
    )
    return _interaction_features(base) if cross_terms else base


def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if y_true.size == 0:
        return {
            "n": 0,
            "rmse_log": None,
            "mae_log": None,
            "mape_pct": None,
            "xerr": None,
        }

    rmse_log = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_log = float(np.mean(np.abs(y_true - y_pred)))
    true_linear = np.exp(y_true)
    pred_linear = np.exp(y_pred)
    mape = float(np.mean(np.abs(true_linear - pred_linear) / true_linear) * 100.0)
    return {
        "n": int(y_true.size),
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "mape_pct": mape,
        "xerr": float(np.exp(rmse_log)),
    }


def _fit_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_builder,
    y_key: str,
    cross_terms: bool,
) -> dict[str, float | None]:
    if train_df.empty or test_df.empty:
        return {
            "n": int(len(test_df)),
            "rmse_log": None,
            "mae_log": None,
            "mape_pct": None,
            "xerr": None,
        }

    train_X_df = feature_builder(train_df, cross_terms=cross_terms)
    test_X_df = feature_builder(test_df, cross_terms=cross_terms)
    train_y = np.log(train_df[y_key].astype(float).to_numpy())
    test_y = np.log(test_df[y_key].astype(float).to_numpy())

    names = list(train_X_df.columns)
    fit = _fit_linear(train_X_df.to_numpy(), train_y, names)
    coeff = np.array([fit.intercept] + [fit.coefficients[name] for name in names])
    test_X = np.column_stack([np.ones(len(test_X_df)), test_X_df.to_numpy()])
    test_pred = test_X @ coeff
    return _error_metrics(test_y, test_pred)


def _make_holdout_splits(df: pd.DataFrame, heldout_model_b: float) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    interp_mask = df["batch_size"].isin([4, 64]) | df["seq_in"].isin([512, 4096])
    model_mask = np.isclose(df["active_params_b"], heldout_model_b, atol=0.5)
    seq_mask = df["seq_in"] == int(df["seq_in"].max())
    corner_mask = (df["batch_size"] >= 128) & (df["seq_in"] >= 4096)

    return {
        "interpolation": (df[~interp_mask], df[interp_mask]),
        "model_extrapolation": (df[~model_mask], df[model_mask]),
        "sequence_extrapolation": (df[~seq_mask], df[seq_mask]),
        "corner_extrapolation": (df[~corner_mask], df[corner_mask]),
    }


def _fit_family(
    df: pd.DataFrame,
    *,
    y_key: str,
    feature_builder,
    cv_folds: int,
    cv_seed: int,
    heldout_model_b: float,
) -> dict[str, Any]:
    if df.empty:
        return {"simple": None, "cross": None}

    payload: dict[str, Any] = {}
    splits = _make_holdout_splits(df, heldout_model_b=heldout_model_b)

    for cross_terms, label in ((False, "simple"), (True, "cross")):
        X_df = feature_builder(df, cross_terms=cross_terms)
        names = list(X_df.columns)
        y = np.log(df[y_key].astype(float).to_numpy())
        fit = _fit_linear(X_df.to_numpy(), y, names)
        cv = _kfold_r2(
            X_df.to_numpy(),
            y,
            names,
            k=cv_folds,
            seed=cv_seed,
        )

        holdouts: dict[str, Any] = {}
        for split_name, (train_df, test_df) in splits.items():
            holdouts[split_name] = _fit_and_eval(
                train_df=train_df,
                test_df=test_df,
                feature_builder=feature_builder,
                y_key=y_key,
                cross_terms=cross_terms,
            )

        payload[label] = {
            "r2": fit.r2,
            "intercept": fit.intercept,
            "coefficients": fit.coefficients,
            "cv_r2": cv,
            "holdouts": holdouts,
        }

    # Explicit simple-vs-cross comparison on held-out RMSE.
    comparisons: dict[str, Any] = {}
    simple_holdouts = payload["simple"]["holdouts"]
    cross_holdouts = payload["cross"]["holdouts"]
    for split_name in simple_holdouts.keys():
        simple_rmse = simple_holdouts[split_name].get("rmse_log")
        cross_rmse = cross_holdouts[split_name].get("rmse_log")
        preferred: str | None = None
        if simple_rmse is not None and cross_rmse is not None:
            preferred = "simple" if simple_rmse <= cross_rmse else "cross"
        comparisons[split_name] = {
            "simple_rmse_log": simple_rmse,
            "cross_rmse_log": cross_rmse,
            "preferred": preferred,
        }
    payload["simple_vs_cross"] = comparisons
    return payload


def fit_scaling_laws(
    frame: pd.DataFrame,
    *,
    gpu: str | None,
    cv_folds: int,
    cv_seed: int,
    heldout_model_b: float,
) -> dict[str, Any]:
    working = frame.copy()
    if gpu:
        working = working[working["gpu"] == gpu]

    # Keep only rows usable by both families.
    required_cols = [
        "active_params_b",
        "batch_size",
        "seq_in",
        "seq_out",
        "bytes_per_param",
        "E_prefill_j",
        "E_decode_j",
    ]
    working = working.dropna(subset=required_cols)
    working = working[
        (working["active_params_b"] > 0)
        & (working["batch_size"] > 0)
        & (working["seq_in"] > 0)
        & (working["seq_out"] > 0)
        & (working["bytes_per_param"] > 0)
        & (working["E_prefill_j"] > 0)
        & (working["E_decode_j"] > 0)
    ].copy()

    prefill_fit = _fit_family(
        working,
        y_key="E_prefill_j",
        feature_builder=_build_prefill_features,
        cv_folds=cv_folds,
        cv_seed=cv_seed,
        heldout_model_b=heldout_model_b,
    )
    decode_fit = _fit_family(
        working,
        y_key="E_decode_j",
        feature_builder=_build_decode_features,
        cv_folds=cv_folds,
        cv_seed=cv_seed,
        heldout_model_b=heldout_model_b,
    )

    return {
        "metadata": {
            "rows_input": int(len(frame)),
            "rows_fit": int(len(working)),
            "gpu_filter": gpu,
            "cv_folds": cv_folds,
            "cv_seed": cv_seed,
            "heldout_model_b": heldout_model_b,
        },
        "prefill": prefill_fit,
        "decode": decode_fit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit Step-5 scaling laws and held-out validation metrics."
    )
    parser.add_argument(
        "--input",
        default="scaling_law_data.parquet",
        help="Input parquet produced by aggregate.py",
    )
    parser.add_argument("--gpu", default=None, help="Optional GPU filter (e.g., h100)")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation fold count",
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=42,
        help="Cross-validation RNG seed",
    )
    parser.add_argument(
        "--heldout-model-b",
        type=float,
        default=14.0,
        help="Model size (in billions) for model-extrapolation holdout",
    )
    parser.add_argument(
        "--output-json",
        default="scaling_law_fit_report.json",
        help="Output report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    frame = pd.read_parquet(input_path)
    report = fit_scaling_laws(
        frame,
        gpu=args.gpu,
        cv_folds=args.cv_folds,
        cv_seed=args.cv_seed,
        heldout_model_b=args.heldout_model_b,
    )
    report["metadata"]["input_path"] = str(input_path)

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(
        f"Wrote fit report to {output_path} "
        f"(rows_fit={report['metadata']['rows_fit']})"
    )

    prefill_simple = (report.get("prefill") or {}).get("simple") or {}
    decode_simple = (report.get("decode") or {}).get("simple") or {}
    print(
        "Simple-model R2: "
        f"prefill={prefill_simple.get('r2')} "
        f"decode={decode_simple.get('r2')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

