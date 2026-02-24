"""Calibration layer: fit correction factors from E2E profiling data.

Provides two fitting pathways:
  1. From phase regression results (ipw analyze --analysis phase-regression)
  2. From grid_eval JSONL traces (QueryResult records)

Fitted CalibrationFactors are persisted as JSON and looked up per
(gpu_type, model_type) pair via CalibrationDB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ipw.simulator.hardware_specs import HardwareSpecs, get_hardware_specs
from ipw.simulator.types import CalibrationFactors

logger = logging.getLogger(__name__)


def fit_from_phase_regression(
    phase_regression_path: Path,
    gpu_type: str = "",
    model_type: str = "",
    hw: Optional[HardwareSpecs] = None,
) -> CalibrationFactors:
    """Fit calibration factors from a phase_regression.json artifact.

    Extracts energy-per-token slopes from the prefill/decode/combined
    regressions produced by PhaseRegressionAnalysis.

    If hardware specs are provided, derives eta_prefill and eta_decode
    from the regression slopes.

    Args:
        phase_regression_path: Path to phase_regression.json.
        gpu_type: GPU type for metadata.
        model_type: Model type for metadata.
        hw: Optional hardware specs for deriving efficiency factors.

    Returns:
        CalibrationFactors with fitted slopes and (optionally) efficiencies.
    """
    with open(phase_regression_path) as f:
        report = json.load(f)

    data = report.get("data", report)

    prefill_reg = data.get("prefill_regression", {})
    decode_reg = data.get("decode_regression", {})
    combined_reg = data.get("combined_regression", {})

    factors = CalibrationFactors(
        gpu_type=gpu_type,
        model_type=model_type,
    )

    # Use combined regression slopes if available (most accurate)
    if combined_reg.get("input_slope") is not None:
        factors.energy_per_input_token_j = combined_reg["input_slope"]
        factors.energy_per_output_token_j = combined_reg["output_slope"]
        factors.intercept_j = combined_reg.get("intercept", 0.0)
        factors.r_squared = combined_reg.get("r2")
        factors.sample_count = combined_reg.get("count", 0)
    else:
        # Fall back to individual phase regressions
        if prefill_reg.get("slope") is not None:
            factors.energy_per_input_token_j = prefill_reg["slope"]
        if decode_reg.get("slope") is not None:
            factors.energy_per_output_token_j = decode_reg["slope"]
        factors.sample_count = max(
            prefill_reg.get("count", 0),
            decode_reg.get("count", 0),
        )

    return factors


def fit_from_grid_eval(
    jsonl_path: Path,
    gpu_type: str = "",
    model_type: str = "",
) -> CalibrationFactors:
    """Fit calibration factors from grid_eval JSONL results.

    Reads QueryResult records, extracts per-query (input_tokens,
    output_tokens, energy) tuples, and fits a multivariate regression.

    Also derives alpha (power fraction) from average power / TDP.

    Args:
        jsonl_path: Path to grid_eval results JSONL file.
        gpu_type: Filter to this GPU type (empty = use all).
        model_type: Filter to this model type (empty = use all).

    Returns:
        CalibrationFactors with fitted regression and power fraction.
    """
    records = _load_matching_records(jsonl_path, gpu_type, model_type)

    if not records:
        logger.warning(
            "No matching records found in %s for gpu=%s model=%s",
            jsonl_path, gpu_type, model_type,
        )
        return CalibrationFactors(gpu_type=gpu_type, model_type=model_type)

    # Extract samples for regression
    # Each record needs: input_tokens, output_tokens, total energy
    samples: List[Tuple[float, float, float]] = []
    powers: List[float] = []
    latencies: List[float] = []

    for rec in records:
        energy = rec.get("avg_joules", 0.0)
        latency = rec.get("latency_seconds", 0.0)

        # Try to get token counts from action_breakdowns or top-level
        input_tokens = _extract_token_count(rec, "input")
        output_tokens = _extract_token_count(rec, "output")

        if input_tokens is not None and output_tokens is not None and energy > 0:
            samples.append((input_tokens, output_tokens, energy))
        if latency > 0 and energy > 0:
            powers.append(energy / latency)
            latencies.append(latency)

    factors = CalibrationFactors(
        gpu_type=gpu_type,
        model_type=model_type,
        sample_count=len(samples),
    )

    # Fit multivariate regression: energy = a*input + b*output + c
    if len(samples) >= 3:
        reg = _fit_multivariate(samples)
        if reg["input_slope"] is not None:
            factors.energy_per_input_token_j = reg["input_slope"]
            factors.energy_per_output_token_j = reg["output_slope"]
            factors.intercept_j = reg.get("intercept", 0.0)
            factors.r_squared = reg.get("r2")

    # Derive alpha from average power / TDP
    if powers and gpu_type:
        try:
            hw = get_hardware_specs(gpu_type)
            avg_power = float(np.mean(powers))
            if hw.tdp_watts > 0:
                factors.alpha = min(avg_power / hw.tdp_watts, 1.0)
        except KeyError:
            pass

    return factors


def _extract_token_count(record: Dict[str, Any], phase: str) -> Optional[float]:
    """Try to extract token counts from a QueryResult record.

    Looks in action_breakdowns for token metadata, or in top-level fields.
    """
    # Try action_breakdowns (per-action profiling)
    breakdowns = record.get("action_breakdowns")
    if breakdowns:
        total = 0.0
        for bd in breakdowns:
            meta = bd.get("metadata", {})
            if phase == "input":
                val = meta.get("prompt_tokens")
            else:
                val = meta.get("completion_tokens")
            if val is not None:
                total += float(val)
        if total > 0:
            return total

    # Try top-level fields (from bench per-action)
    if phase == "input":
        val = record.get("total_prompt_tokens")
    else:
        val = record.get("total_completion_tokens")
    if val is not None:
        return float(val)

    # Cannot determine token counts
    return None


def _load_matching_records(
    jsonl_path: Path,
    gpu_type: str,
    model_type: str,
) -> List[Dict[str, Any]]:
    """Load and filter JSONL records by gpu_type and model_type."""
    records: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if gpu_type and rec.get("gpu_type", "") != gpu_type:
                continue
            if model_type and rec.get("model", "") != model_type:
                continue

            records.append(rec)
    return records


def _fit_multivariate(
    samples: Sequence[Tuple[float, float, float]],
) -> Dict[str, Optional[float]]:
    """Fit energy = input_slope*x1 + output_slope*x2 + intercept."""
    if len(samples) < 2:
        return {
            "count": len(samples),
            "input_slope": None,
            "output_slope": None,
            "intercept": None,
            "r2": None,
        }

    X = np.array([[s[0], s[1], 1.0] for s in samples], dtype=np.float64)
    y = np.array([s[2] for s in samples], dtype=np.float64)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    if len(X) < 2:
        return {
            "count": int(len(X)),
            "input_slope": None,
            "output_slope": None,
            "intercept": None,
            "r2": None,
        }

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "count": int(len(X)),
            "input_slope": None,
            "output_slope": None,
            "intercept": None,
            "r2": None,
        }

    predictions = X @ coeffs
    residuals = y - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return {
        "count": int(len(X)),
        "input_slope": float(coeffs[0]),
        "output_slope": float(coeffs[1]),
        "intercept": float(coeffs[2]),
        "r2": float(r2),
    }


class CalibrationDB:
    """Lookup table of CalibrationFactors per (gpu_type, model_type).

    Supports loading from a JSON file and fallback interpolation
    when an exact match is not available.
    """

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], CalibrationFactors] = {}

    def add(self, factors: CalibrationFactors) -> None:
        """Add calibration factors for a (gpu, model) pair."""
        key = (factors.gpu_type, factors.model_type)
        self._entries[key] = factors

    def get(
        self,
        gpu_type: str,
        model_type: str,
    ) -> Optional[CalibrationFactors]:
        """Look up exact-match calibration factors."""
        return self._entries.get((gpu_type, model_type))

    def get_or_interpolate(
        self,
        gpu_type: str,
        model_type: str,
    ) -> Optional[CalibrationFactors]:
        """Look up calibration factors with fallback interpolation.

        Fallback order:
        1. Exact (gpu_type, model_type) match.
        2. Same gpu_type, any model_type -> average across models.
        3. Same model_type, any gpu_type -> average across hardware.
        4. None (use default roofline).
        """
        # 1. Exact match
        exact = self.get(gpu_type, model_type)
        if exact is not None:
            return exact

        # 2. Same GPU, any model
        same_gpu = [v for (g, m), v in self._entries.items() if g == gpu_type]
        if same_gpu:
            return _average_factors(same_gpu, gpu_type, model_type)

        # 3. Same model, any GPU
        same_model = [v for (g, m), v in self._entries.items() if m == model_type]
        if same_model:
            return _average_factors(same_model, gpu_type, model_type)

        return None

    def save(self, path: Path) -> None:
        """Persist calibration DB to JSON."""
        entries = []
        for (gpu, model), factors in self._entries.items():
            entries.append({
                "gpu_type": factors.gpu_type,
                "model_type": factors.model_type,
                "eta_prefill": factors.eta_prefill,
                "eta_decode": factors.eta_decode,
                "alpha": factors.alpha,
                "energy_per_input_token_j": factors.energy_per_input_token_j,
                "energy_per_output_token_j": factors.energy_per_output_token_j,
                "intercept_j": factors.intercept_j,
                "sample_count": factors.sample_count,
                "r_squared": factors.r_squared,
            })
        with open(path, "w") as f:
            json.dump({"calibration_entries": entries}, f, indent=2)

    def load(self, path: Path) -> None:
        """Load calibration DB from JSON."""
        with open(path) as f:
            data = json.load(f)

        for entry in data.get("calibration_entries", []):
            factors = CalibrationFactors(
                gpu_type=entry.get("gpu_type", ""),
                model_type=entry.get("model_type", ""),
                eta_prefill=entry.get("eta_prefill", 0.5),
                eta_decode=entry.get("eta_decode", 0.6),
                alpha=entry.get("alpha", 0.65),
                energy_per_input_token_j=entry.get("energy_per_input_token_j"),
                energy_per_output_token_j=entry.get("energy_per_output_token_j"),
                intercept_j=entry.get("intercept_j"),
                sample_count=entry.get("sample_count", 0),
                r_squared=entry.get("r_squared"),
            )
            self.add(factors)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self._entries


def _average_factors(
    factors_list: List[CalibrationFactors],
    gpu_type: str,
    model_type: str,
) -> CalibrationFactors:
    """Average multiple CalibrationFactors into a single interpolated result."""
    n = len(factors_list)
    if n == 0:
        return CalibrationFactors(gpu_type=gpu_type, model_type=model_type)

    avg_eta_prefill = sum(f.eta_prefill for f in factors_list) / n
    avg_eta_decode = sum(f.eta_decode for f in factors_list) / n
    avg_alpha = sum(f.alpha for f in factors_list) / n

    # Average token-level slopes if all entries have them
    input_slopes = [f.energy_per_input_token_j for f in factors_list if f.energy_per_input_token_j is not None]
    output_slopes = [f.energy_per_output_token_j for f in factors_list if f.energy_per_output_token_j is not None]
    intercepts = [f.intercept_j for f in factors_list if f.intercept_j is not None]

    return CalibrationFactors(
        eta_prefill=avg_eta_prefill,
        eta_decode=avg_eta_decode,
        alpha=avg_alpha,
        energy_per_input_token_j=sum(input_slopes) / len(input_slopes) if input_slopes else None,
        energy_per_output_token_j=sum(output_slopes) / len(output_slopes) if output_slopes else None,
        intercept_j=sum(intercepts) / len(intercepts) if intercepts else None,
        gpu_type=gpu_type,
        model_type=model_type,
        sample_count=sum(f.sample_count for f in factors_list),
    )


__all__ = [
    "CalibrationDB",
    "fit_from_grid_eval",
    "fit_from_phase_regression",
]
