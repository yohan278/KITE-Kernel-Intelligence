"""Phase-aware regression analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ipw.analysis.base import AnalysisContext, AnalysisProvider, AnalysisResult
from ipw.analysis.helpers import iter_model_entries, load_metrics_dataset, resolve_model_name
from ipw.core.registry import AnalysisRegistry


@dataclass(slots=True)
class RegressionSample:
    x: float
    y: float


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _compute_regression(samples: Sequence[RegressionSample]) -> dict[str, Optional[float]]:
    if len(samples) < 2:
        return {"count": len(samples), "slope": None, "intercept": None, "r2": None}

    x = np.array([s.x for s in samples], dtype=np.float64)
    y = np.array([s.y for s in samples], dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2 or np.unique(x).size < 2:
        return {"count": int(len(x)), "slope": None, "intercept": None, "r2": None}

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return {"count": int(len(x)), "slope": None, "intercept": None, "r2": None}

    predictions = slope * x + intercept
    residuals = y - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return {
        "count": int(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def _compute_multivariate_regression(
    samples: Sequence[tuple[float, float, float]]
) -> dict[str, Optional[float]]:
    if len(samples) < 2:
        return {
            "count": len(samples),
            "input_slope": None,
            "output_slope": None,
            "intercept": None,
            "r2": None,
        }

    x = np.array([[s[0], s[1], 1.0] for s in samples], dtype=np.float64)
    y = np.array([s[2] for s in samples], dtype=np.float64)

    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {
            "count": int(len(x)),
            "input_slope": None,
            "output_slope": None,
            "intercept": None,
            "r2": None,
        }

    try:
        coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "count": int(len(x)),
            "input_slope": None,
            "output_slope": None,
            "intercept": None,
            "r2": None,
        }

    input_slope = float(coeffs[0])
    output_slope = float(coeffs[1])
    intercept = float(coeffs[2])

    predictions = x @ coeffs
    residuals = y - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return {
        "count": int(len(x)),
        "input_slope": input_slope,
        "output_slope": output_slope,
        "intercept": intercept,
        "r2": float(r2),
    }


@AnalysisRegistry.register("phase-regression")
class PhaseRegressionAnalysis(AnalysisProvider):
    """Regression analysis for prefill and decode energy."""

    analysis_id = "phase-regression"

    def run(self, context: AnalysisContext) -> AnalysisResult:
        results_dir = context.results_dir
        options = dict(context.options)
        requested_model = options.get("model")

        dataset = load_metrics_dataset(results_dir)
        active_model = resolve_model_name(dataset, requested_model, results_dir)

        entries = list(iter_model_entries(dataset, active_model))
        if not entries:
            raise RuntimeError(
                f"No usable metrics found for model '{active_model}' in dataset at '{results_dir}'."
            )

        prefill_samples: list[RegressionSample] = []
        decode_samples: list[RegressionSample] = []
        combined_samples: list[tuple[float, float, float]] = []

        flat_samples_found = False
        for row in dataset:
            if not isinstance(row, Mapping):
                continue
            input_tokens = _to_float(row.get("input_tokens"))
            output_tokens = _to_float(row.get("output_tokens"))
            prefill_energy = _to_float(row.get("prefill_energy_j"))
            decode_energy = _to_float(row.get("decode_energy_j"))

            if prefill_energy is not None or decode_energy is not None:
                flat_samples_found = True

            if input_tokens is not None and prefill_energy is not None:
                prefill_samples.append(RegressionSample(input_tokens, prefill_energy))
            if output_tokens is not None and decode_energy is not None:
                decode_samples.append(RegressionSample(output_tokens, decode_energy))
            if (
                input_tokens is not None
                and output_tokens is not None
                and prefill_energy is not None
                and decode_energy is not None
            ):
                combined_samples.append(
                    (input_tokens, output_tokens, prefill_energy + decode_energy)
                )

        if not flat_samples_found:
            prefill_samples.clear()
            decode_samples.clear()
            combined_samples.clear()
            for entry in entries:
                token_metrics = _get_mapping(entry.get("token_metrics"))
                phase_metrics = _get_mapping(entry.get("phase_metrics"))

                input_tokens = _to_float(token_metrics.get("input"))
                output_tokens = _to_float(token_metrics.get("output"))
                prefill_energy = _to_float(phase_metrics.get("prefill_energy_j"))
                decode_energy = _to_float(phase_metrics.get("decode_energy_j"))

                if input_tokens is not None and prefill_energy is not None:
                    prefill_samples.append(
                        RegressionSample(input_tokens, prefill_energy)
                    )
                if output_tokens is not None and decode_energy is not None:
                    decode_samples.append(
                        RegressionSample(output_tokens, decode_energy)
                    )

                if (
                    input_tokens is not None
                    and output_tokens is not None
                    and prefill_energy is not None
                    and decode_energy is not None
                ):
                    combined_samples.append(
                        (input_tokens, output_tokens, prefill_energy + decode_energy)
                    )

        if not prefill_samples and not decode_samples and not combined_samples:
            raise RuntimeError(
                f"No phase-aware metrics found for model '{active_model}' in dataset at '{results_dir}'. "
                "Run profiling with --phased to enable phase attribution."
            )

        prefill_regression = _compute_regression(prefill_samples)
        decode_regression = _compute_regression(decode_samples)
        combined_regression = _compute_multivariate_regression(combined_samples)

        summary_payload = {
            "prefill_samples": prefill_regression.get("count", 0),
            "decode_samples": decode_regression.get("count", 0),
            "combined_samples": combined_regression.get("count", 0),
        }
        data_payload = {
            "prefill_regression": prefill_regression,
            "decode_regression": decode_regression,
            "combined_regression": combined_regression,
        }

        artifact_payload = {
            "analysis": self.analysis_id,
            "summary": summary_payload,
            "data": data_payload,
        }

        artifact_dir = results_dir / "analysis"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "phase_regression.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))

        return AnalysisResult(
            analysis=self.analysis_id,
            summary=summary_payload,
            data=data_payload,
            artifacts={"report": artifact_path},
        )


__all__ = ["PhaseRegressionAnalysis"]
