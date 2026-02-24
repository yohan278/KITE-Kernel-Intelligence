"""Step 6: estimate cross-hardware normalization factors (k_HW)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_TERM_TO_COLUMN = {
    "logP": "active_params_b",
    "logB": "batch_size",
    "logSin": "seq_in",
    "logSout": "seq_out",
    "logQ": "bytes_per_param",
}

_PHASE_BASE_TERMS = {
    "prefill": ("logP", "logB", "logSin", "logQ"),
    "decode": ("logP", "logB", "logSout", "logSin", "logQ"),
}


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _deep_merge(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if (
            key in dst
            and isinstance(dst[key], Mapping)
            and isinstance(value, Mapping)
        ):
            nested = dict(dst[key])
            dst[key] = _deep_merge(nested, value)
        else:
            dst[key] = value
    return dst


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_benchmark_parameters(paths: Sequence[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    merged: dict[str, Any] = {}
    for path in paths:
        payload = _load_json(path)
        params = payload.get("parameters") if "parameters" in payload else payload
        if not isinstance(params, Mapping):
            raise ValueError(
                f"Benchmark JSON at {path} does not contain a valid parameters object"
            )
        _deep_merge(merged, params)
    return merged


def _lookup_dtype_metric(
    section: Mapping[str, Any] | None,
    *,
    dtype: str,
    key: str,
) -> float | None:
    if not isinstance(section, Mapping):
        return None
    dtype_norm = dtype.strip().lower()
    for dt_key, values in section.items():
        if str(dt_key).strip().lower() != dtype_norm:
            continue
        if isinstance(values, Mapping):
            return _first_float(values.get(key))
    return None


def _lookup_simple_dtype_metric(
    section: Mapping[str, Any] | None,
    *,
    dtype: str,
) -> float | None:
    if not isinstance(section, Mapping):
        return None
    dtype_norm = dtype.strip().lower()
    for dt_key, value in section.items():
        if str(dt_key).strip().lower() == dtype_norm:
            return _first_float(value)
    return None


def _ratio(target: float | None, source: float | None) -> float | None:
    if target is None or source is None or source == 0.0:
        return None
    return float(target / source)


def _extract_benchmark_epsilons(
    parameters: Mapping[str, Any] | None,
    *,
    dtype: str,
) -> dict[str, float | None]:
    if not isinstance(parameters, Mapping):
        return {
            "memory_total_pj_per_bit": None,
            "compute_total_pj_per_flop": None,
            "gemm_total_pj_per_flop": None,
            "inference_gemm_prefill_pj_per_flop": None,
            "inference_gemm_decode_pj_per_flop": None,
            "attention_pj_per_flop": None,
            "kv_read_pj_per_bit": None,
            "kv_write_pj_per_bit": None,
            "decode_batch_exponent": None,
        }

    memory = parameters.get("memory")
    compute = parameters.get("compute")
    gemm = parameters.get("gemm")
    inference = parameters.get("inference")
    inference = inference if isinstance(inference, Mapping) else {}

    return {
        "memory_total_pj_per_bit": _first_float(
            memory.get("total_pj_per_bit") if isinstance(memory, Mapping) else None
        ),
        "compute_total_pj_per_flop": _lookup_dtype_metric(
            compute if isinstance(compute, Mapping) else None,
            dtype=dtype,
            key="total_pj_per_flop",
        ),
        "gemm_total_pj_per_flop": _lookup_dtype_metric(
            gemm if isinstance(gemm, Mapping) else None,
            dtype=dtype,
            key="total_pj_per_flop",
        ),
        "inference_gemm_prefill_pj_per_flop": _lookup_simple_dtype_metric(
            inference.get("gemm_prefill") if isinstance(inference, Mapping) else None,
            dtype=dtype,
        ),
        "inference_gemm_decode_pj_per_flop": _lookup_simple_dtype_metric(
            inference.get("gemm_decode") if isinstance(inference, Mapping) else None,
            dtype=dtype,
        ),
        "attention_pj_per_flop": _first_float(inference.get("attention_pj_per_flop")),
        "kv_read_pj_per_bit": _first_float(inference.get("kv_read_pj_per_bit")),
        "kv_write_pj_per_bit": _first_float(inference.get("kv_write_pj_per_bit")),
        "decode_batch_exponent": _first_float(inference.get("decode_batch_exponent")),
    }


def _base_log_features(row: pd.Series, *, phase: str) -> dict[str, float] | None:
    base_terms = _PHASE_BASE_TERMS[phase]
    features: dict[str, float] = {}
    for term in base_terms:
        column = _TERM_TO_COLUMN[term]
        value = _first_float(row.get(column))
        if value is None or value <= 0.0:
            return None
        features[term] = math.log(value)
    return features


def _term_value(term: str, features: Mapping[str, float]) -> float:
    if term in features:
        return features[term]
    if ":" in term:
        left, right = term.split(":", maxsplit=1)
        if left in features and right in features:
            return features[left] * features[right]
    raise ValueError(f"Unsupported coefficient term '{term}'")


def _phase_model(report: Mapping[str, Any], *, phase: str, model_kind: str) -> Mapping[str, Any]:
    phase_blob = report.get(phase)
    if not isinstance(phase_blob, Mapping):
        raise ValueError(f"Fit report is missing phase '{phase}'")
    model_blob = phase_blob.get(model_kind)
    if not isinstance(model_blob, Mapping):
        available = sorted(
            key
            for key, value in phase_blob.items()
            if isinstance(value, Mapping) and value.get("coefficients") is not None
        )
        raise ValueError(
            f"Fit report is missing {phase}.{model_kind} model. Available: {available}"
        )
    return model_blob


def _predict_log_energy(
    frame: pd.DataFrame,
    *,
    phase: str,
    phase_model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    intercept = _first_float(phase_model.get("intercept"))
    coefficients = phase_model.get("coefficients")
    if intercept is None or not isinstance(coefficients, Mapping):
        raise ValueError(f"Invalid fit payload for phase '{phase}'")

    energy_col = "E_prefill_j" if phase == "prefill" else "E_decode_j"
    y_true: list[float] = []
    y_pred: list[float] = []

    for _, row in frame.iterrows():
        energy = _first_float(row.get(energy_col))
        if energy is None or energy <= 0.0:
            continue

        features = _base_log_features(row, phase=phase)
        if features is None:
            continue

        pred = intercept
        for term, coef_raw in coefficients.items():
            coef = _first_float(coef_raw)
            if coef is None:
                continue
            pred += coef * _term_value(str(term), features)

        y_true.append(math.log(energy))
        y_pred.append(pred)

    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def _phase_k_stats(
    frame: pd.DataFrame,
    *,
    phase: str,
    phase_model: Mapping[str, Any],
) -> dict[str, float | int | None]:
    y_true, y_pred = _predict_log_energy(frame, phase=phase, phase_model=phase_model)
    if y_true.size == 0:
        return {
            "n": 0,
            "log_ratio_mean": None,
            "log_ratio_std": None,
            "ratio_target_over_source": None,
            "rmse_log": None,
            "mae_log": None,
            "mape_pct": None,
            "xerr": None,
        }

    residual = y_true - y_pred
    rmse_log = float(np.sqrt(np.mean(residual ** 2)))
    mae_log = float(np.mean(np.abs(residual)))
    mape = float(np.mean(np.abs(np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true)) * 100.0)
    log_ratio_mean = float(np.mean(residual))

    return {
        "n": int(y_true.size),
        "log_ratio_mean": log_ratio_mean,
        "log_ratio_std": float(np.std(residual)),
        "ratio_target_over_source": float(np.exp(log_ratio_mean)),
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "mape_pct": mape,
        "xerr": float(np.exp(rmse_log)),
    }


def estimate_hardware_normalization(
    frame: pd.DataFrame,
    fit_report: Mapping[str, Any],
    *,
    target_gpu: str,
    model_kind: str,
    source_gpu: str | None = None,
    dtype: str = "fp16",
    source_benchmark_parameters: Mapping[str, Any] | None = None,
    target_benchmark_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = [
        "gpu",
        "active_params_b",
        "batch_size",
        "seq_in",
        "seq_out",
        "bytes_per_param",
        "E_prefill_j",
        "E_decode_j",
    ]
    working = frame.dropna(subset=required).copy()
    working = working[
        (working["active_params_b"] > 0)
        & (working["batch_size"] > 0)
        & (working["seq_in"] > 0)
        & (working["seq_out"] > 0)
        & (working["bytes_per_param"] > 0)
    ]

    target = working[working["gpu"] == target_gpu].copy()

    source_gpu_resolved = source_gpu
    if not source_gpu_resolved and isinstance(fit_report.get("metadata"), Mapping):
        maybe = fit_report["metadata"].get("gpu_filter")
        if isinstance(maybe, str) and maybe.strip():
            source_gpu_resolved = maybe.strip()

    prefill_model = _phase_model(fit_report, phase="prefill", model_kind=model_kind)
    decode_model = _phase_model(fit_report, phase="decode", model_kind=model_kind)

    prefill_stats = _phase_k_stats(target, phase="prefill", phase_model=prefill_model)
    decode_stats = _phase_k_stats(target, phase="decode", phase_model=decode_model)

    source_eps = _extract_benchmark_epsilons(source_benchmark_parameters, dtype=dtype)
    target_eps = _extract_benchmark_epsilons(target_benchmark_parameters, dtype=dtype)
    ratio_eps = {
        key: _ratio(target_eps.get(key), source_eps.get(key))
        for key in sorted(set(source_eps) | set(target_eps))
    }

    return {
        "metadata": {
            "target_gpu": target_gpu,
            "source_gpu": source_gpu_resolved,
            "model_kind": model_kind,
            "dtype": dtype,
            "rows_input": int(len(frame)),
            "rows_usable": int(len(working)),
            "rows_target": int(len(target)),
        },
        "k_hw": {
            "prefill": prefill_stats,
            "decode": decode_stats,
        },
        "benchmark_epsilons": {
            "source": source_eps,
            "target": target_eps,
            "ratio_target_over_source": ratio_eps,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 6: estimate k_HW(target)/k_HW(source) by applying source-GPU "
            "fit coefficients to target-GPU runs."
        )
    )
    parser.add_argument(
        "--input",
        default="scaling_law_data.parquet",
        help="Input parquet from aggregate.py",
    )
    parser.add_argument(
        "--fit-report",
        required=True,
        help="Fit report JSON produced by fit.py (typically source GPU)",
    )
    parser.add_argument(
        "--target-gpu",
        required=True,
        help="Target GPU label in parquet (e.g., a100)",
    )
    parser.add_argument(
        "--source-gpu",
        default=None,
        help="Optional source GPU label override (defaults to fit report metadata.gpu_filter)",
    )
    parser.add_argument(
        "--model-kind",
        choices=["simple", "cross"],
        default="cross",
        help="Use simple or cross-term coefficients from fit report",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        help="Dtype key for epsilon extraction from benchmark JSONs",
    )
    parser.add_argument(
        "--source-benchmark-json",
        action="append",
        default=[],
        metavar="PATH",
        help="Source-GPU benchmark JSON (repeatable; merged in order)",
    )
    parser.add_argument(
        "--target-benchmark-json",
        action="append",
        default=[],
        metavar="PATH",
        help="Target-GPU benchmark JSON (repeatable; merged in order)",
    )
    parser.add_argument(
        "--output-json",
        default="hardware_normalization_report.json",
        help="Output report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    fit_path = Path(args.fit_report).resolve()

    frame = pd.read_parquet(input_path)
    fit_report = _load_json(fit_path)

    source_benchmark = _load_benchmark_parameters(
        [Path(p).resolve() for p in args.source_benchmark_json]
    )
    target_benchmark = _load_benchmark_parameters(
        [Path(p).resolve() for p in args.target_benchmark_json]
    )

    report = estimate_hardware_normalization(
        frame,
        fit_report,
        target_gpu=args.target_gpu,
        source_gpu=args.source_gpu,
        model_kind=args.model_kind,
        dtype=args.dtype,
        source_benchmark_parameters=source_benchmark,
        target_benchmark_parameters=target_benchmark,
    )
    report["metadata"]["input_path"] = str(input_path)
    report["metadata"]["fit_report_path"] = str(fit_path)

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    prefill_ratio = report["k_hw"]["prefill"]["ratio_target_over_source"]
    decode_ratio = report["k_hw"]["decode"]["ratio_target_over_source"]
    print(
        f"Wrote hardware normalization report: {output_path}\n"
        f"k_HW ratio (target/source): prefill={prefill_ratio} decode={decode_ratio}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
