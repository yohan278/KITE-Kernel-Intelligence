"""PPI-rectified estimator wrapper for bias-corrected predictions with CIs.

Wraps any BaseRuntimeEstimator and uses Prediction Powered Inference (PPI)
to produce bias-corrected point estimates and confidence intervals by
combining real measurements (small n) with ML predictions (large N).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement

try:
    from ppi_py import ppi_mean_ci, ppi_mean_pointestimate

    _HAS_PPI = True
except ImportError:
    _HAS_PPI = False


@dataclass(frozen=True)
class RectifiedResult:
    """Bias-corrected prediction with optional confidence interval."""

    time_s: float
    time_s_ci: tuple[float, float] | None = None
    energy_j: float | None = None
    energy_j_ci: tuple[float, float] | None = None
    power_w: float | None = None


class PPIRectifiedEstimator(BaseRuntimeEstimator):
    """Bias-corrects an estimator using PPI with real measurements.

    Composable: can wrap a ``PerOperatorEstimator``, and the result can be
    wrapped by ``PredictionCache``.

    If ``ppi-python`` is not installed, falls back to the unwrapped estimator.
    """

    def __init__(
        self,
        estimator: BaseRuntimeEstimator,
        measurements: Sequence[OperatorMeasurement],
        alpha: float = 0.1,
    ) -> None:
        self._estimator = estimator
        self._alpha = alpha
        self._rectifiers: Dict[OperatorCategory, Dict[str, Any]] = {}

        if not _HAS_PPI:
            return

        # Group measurements by category
        by_category: Dict[OperatorCategory, list[OperatorMeasurement]] = {}
        for m in measurements:
            by_category.setdefault(m.category, []).append(m)

        for category, cat_measurements in by_category.items():
            if len(cat_measurements) < 2:
                continue

            Y_time = np.array([m.time_s for m in cat_measurements])
            Yhat_time = np.array(
                [
                    self._estimator.estimate(
                        m.category, m.batch_size, m.seq_len
                    ).time_s
                    for m in cat_measurements
                ]
            )

            # Compute bias: rectified_mean - mean(Yhat)
            # When no separate unlabeled set, use Yhat as Yhat_unlabeled
            rect_arr = ppi_mean_pointestimate(
                Y_time.reshape(-1, 1),
                Yhat_time.reshape(-1, 1),
                Yhat_time.reshape(-1, 1),
            )
            rectified_mean = float(np.asarray(rect_arr).item())
            bias_time = rectified_mean - float(np.mean(Yhat_time))

            # Energy bias (if available)
            energy_vals = [
                m.energy_j for m in cat_measurements if m.energy_j is not None
            ]
            bias_energy = 0.0
            Y_energy = None
            Yhat_energy = None
            if len(energy_vals) >= 2:
                energy_measurements = [
                    m for m in cat_measurements if m.energy_j is not None
                ]
                Y_energy = np.array([m.energy_j for m in energy_measurements])
                Yhat_energy = np.array(
                    [
                        self._estimator.estimate(
                            m.category, m.batch_size, m.seq_len
                        ).energy_j
                        or 0.0
                        for m in energy_measurements
                    ]
                )
                rect_e_arr = ppi_mean_pointestimate(
                    Y_energy.reshape(-1, 1),
                    Yhat_energy.reshape(-1, 1),
                    Yhat_energy.reshape(-1, 1),
                )
                rect_energy = float(np.asarray(rect_e_arr).item())
                bias_energy = rect_energy - float(np.mean(Yhat_energy))

            self._rectifiers[category] = {
                "bias_time": bias_time,
                "bias_energy": bias_energy,
                "Y_time": Y_time,
                "Yhat_time": Yhat_time,
                "Y_energy": Y_energy,
                "Yhat_energy": Yhat_energy,
            }

    def is_fitted(self) -> bool:
        return self._estimator.is_fitted()

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        base_result = self._estimator.estimate(
            operator_category, batch_size, seq_len, **kwargs
        )

        if not _HAS_PPI:
            return base_result

        rect = self._rectifiers.get(operator_category)
        if rect is None:
            return base_result

        corrected_time = max(base_result.time_s + rect["bias_time"], 1e-12)

        corrected_energy = base_result.energy_j
        if corrected_energy is not None and rect["bias_energy"] != 0.0:
            corrected_energy = max(corrected_energy + rect["bias_energy"], 1e-12)

        return EstimatorResult(
            time_s=corrected_time,
            energy_j=corrected_energy,
            power_w=base_result.power_w,
        )

    def estimate_with_ci(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> RectifiedResult:
        """Return bias-corrected estimate with confidence interval.

        Uses stored labeled data per-category. Constructs Yhat_unlabeled
        from the base estimator at a small grid around (batch_size, seq_len)
        and calls ``ppi_mean_ci()`` for the CI.
        """
        base_result = self._estimator.estimate(
            operator_category, batch_size, seq_len, **kwargs
        )

        if not _HAS_PPI:
            return RectifiedResult(
                time_s=base_result.time_s,
                energy_j=base_result.energy_j,
                power_w=base_result.power_w,
            )

        rect = self._rectifiers.get(operator_category)
        if rect is None:
            return RectifiedResult(
                time_s=base_result.time_s,
                energy_j=base_result.energy_j,
                power_w=base_result.power_w,
            )

        # Build Yhat_unlabeled from a grid around the query point
        bs_grid = [max(1, batch_size // 2), batch_size, batch_size * 2]
        sl_grid = [max(1, seq_len // 2), seq_len, seq_len * 2]
        Yhat_unlabeled_list = []
        for bs in bs_grid:
            for sl in sl_grid:
                pred = self._estimator.estimate(
                    operator_category, bs, sl, **kwargs
                )
                Yhat_unlabeled_list.append(pred.time_s)
        Yhat_unlabeled = np.array(Yhat_unlabeled_list)

        Y = rect["Y_time"]
        Yhat = rect["Yhat_time"]

        # PPI point estimate
        rect_arr = ppi_mean_pointestimate(
            Y.reshape(-1, 1),
            Yhat.reshape(-1, 1),
            Yhat_unlabeled.reshape(-1, 1),
        )
        corrected_time = max(float(np.asarray(rect_arr).item()), 1e-12)

        # PPI CI
        ci = ppi_mean_ci(
            Y.reshape(-1, 1),
            Yhat.reshape(-1, 1),
            Yhat_unlabeled.reshape(-1, 1),
            alpha=self._alpha,
        )
        time_ci = (
            max(float(np.asarray(ci[0]).item()), 0.0),
            max(float(np.asarray(ci[1]).item()), 0.0),
        )

        # Energy CI (if available)
        energy_j = base_result.energy_j
        energy_ci = None
        if (
            rect["Y_energy"] is not None
            and rect["Yhat_energy"] is not None
            and energy_j is not None
        ):
            Yhat_energy_unlabeled = np.array(
                [
                    self._estimator.estimate(
                        operator_category, bs, sl, **kwargs
                    ).energy_j
                    or 0.0
                    for bs in bs_grid
                    for sl in sl_grid
                ]
            )
            rect_e_arr = ppi_mean_pointestimate(
                rect["Y_energy"].reshape(-1, 1),
                rect["Yhat_energy"].reshape(-1, 1),
                Yhat_energy_unlabeled.reshape(-1, 1),
            )
            energy_j = max(float(np.asarray(rect_e_arr).item()), 1e-12)
            eci = ppi_mean_ci(
                rect["Y_energy"].reshape(-1, 1),
                rect["Yhat_energy"].reshape(-1, 1),
                Yhat_energy_unlabeled.reshape(-1, 1),
                alpha=self._alpha,
            )
            energy_ci = (
                max(float(np.asarray(eci[0]).item()), 0.0),
                max(float(np.asarray(eci[1]).item()), 0.0),
            )

        return RectifiedResult(
            time_s=corrected_time,
            time_s_ci=time_ci,
            energy_j=energy_j,
            energy_j_ci=energy_ci,
            power_w=base_result.power_w,
        )

    def rectification_summary(self) -> Dict[str, Dict[str, float]]:
        """Return per-category diagnostics: bias magnitude and CI width."""
        summary: Dict[str, Dict[str, float]] = {}
        for category, rect in self._rectifiers.items():
            entry: Dict[str, float] = {
                "bias_time": rect["bias_time"],
                "bias_energy": rect["bias_energy"],
                "n_measurements": len(rect["Y_time"]),
            }

            if _HAS_PPI:
                Y = rect["Y_time"]
                Yhat = rect["Yhat_time"]
                ci = ppi_mean_ci(
                    Y.reshape(-1, 1),
                    Yhat.reshape(-1, 1),
                    Yhat.reshape(-1, 1),
                    alpha=self._alpha,
                )
                entry["ci_width"] = float(np.asarray(ci[1]).item()) - float(np.asarray(ci[0]).item())

            summary[category.value] = entry
        return summary


__all__ = ["PPIRectifiedEstimator", "RectifiedResult"]
