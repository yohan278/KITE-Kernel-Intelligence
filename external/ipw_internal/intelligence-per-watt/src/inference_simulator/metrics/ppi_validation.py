"""PPI-rectified simulation metrics for Pipeline #2 validation.

Combines a small set of real serving measurements with simulator predictions
to produce bias-corrected latency percentiles and confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ppi_py import (
        ppi_mean_ci,
        ppi_mean_pointestimate,
        ppi_quantile_ci,
        ppi_quantile_pointestimate,
    )

    _HAS_PPI = True
except ImportError:
    _HAS_PPI = False


@dataclass
class RealServingMeasurements:
    """Ground-truth latencies from real inference serving."""

    ttft_s: np.ndarray  # shape (n,)
    tbt_s: np.ndarray  # shape (n_decode_steps,)
    e2e_s: np.ndarray  # shape (n,)
    energy_j: Optional[np.ndarray] = None  # shape (n,) if available


@dataclass
class SimulatedLatencies:
    """Simulator-predicted latencies at the same request configs as real measurements."""

    ttft_s: np.ndarray  # shape (n,) — must match real_latencies
    tbt_s: np.ndarray  # shape (n_decode_steps,)
    e2e_s: np.ndarray  # shape (n,)
    energy_j: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RectifiedSimulationMetrics:
    """Simulation metrics rectified against real serving measurements."""

    # Rectified point estimates
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    tbt_p50: float = 0.0
    tbt_p90: float = 0.0
    tbt_p95: float = 0.0
    tbt_p99: float = 0.0

    e2e_p50: float = 0.0
    e2e_p90: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0

    # Confidence intervals
    ttft_p50_ci: Optional[tuple[float, float]] = None
    ttft_p90_ci: Optional[tuple[float, float]] = None
    ttft_p95_ci: Optional[tuple[float, float]] = None
    ttft_p99_ci: Optional[tuple[float, float]] = None

    tbt_p50_ci: Optional[tuple[float, float]] = None
    tbt_p90_ci: Optional[tuple[float, float]] = None
    tbt_p95_ci: Optional[tuple[float, float]] = None
    tbt_p99_ci: Optional[tuple[float, float]] = None

    e2e_p50_ci: Optional[tuple[float, float]] = None
    e2e_p90_ci: Optional[tuple[float, float]] = None
    e2e_p95_ci: Optional[tuple[float, float]] = None
    e2e_p99_ci: Optional[tuple[float, float]] = None

    # Diagnostic: bias magnitude per metric
    ttft_mean_bias: float = 0.0
    tbt_mean_bias: float = 0.0
    e2e_mean_bias: float = 0.0


def _rectify_quantile(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    q: float,
    alpha: float,
) -> tuple[float, Optional[tuple[float, float]]]:
    """Compute PPI-rectified quantile with CI.

    Note: ppi_quantile_* functions require 1D arrays, unlike ppi_mean_*
    which require 2D (n, 1) arrays.
    """
    Y_flat = Y.ravel()
    Yhat_flat = Yhat.ravel()
    Yhat_u_flat = Yhat_unlabeled.ravel()

    point = float(
        ppi_quantile_pointestimate(Y_flat, Yhat_flat, Yhat_u_flat, q)
    )
    point = max(point, 0.0)

    try:
        ci = ppi_quantile_ci(Y_flat, Yhat_flat, Yhat_u_flat, q, alpha=alpha)
        ci_tuple = (max(float(ci[0]), 0.0), max(float(ci[1]), 0.0))
    except Exception:
        ci_tuple = None

    return point, ci_tuple


def _compute_bias(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """Compute mean bias: positive = simulator overpredicts."""
    return float(np.mean(Yhat - Y))


def rectify_simulation_metrics(
    real_latencies: RealServingMeasurements,
    simulated_latencies: SimulatedLatencies,
    sim_unlabeled_ttft: np.ndarray,
    sim_unlabeled_tbt: np.ndarray,
    sim_unlabeled_e2e: np.ndarray,
    alpha: float = 0.1,
) -> RectifiedSimulationMetrics:
    """Rectify simulation metrics using PPI.

    Args:
        real_latencies: Ground-truth latencies from real serving (labeled Y).
        simulated_latencies: Simulator predictions at the same configs (labeled Yhat).
        sim_unlabeled_ttft: TTFT predictions for all N simulated requests (unlabeled).
        sim_unlabeled_tbt: TBT predictions for all N decode steps (unlabeled).
        sim_unlabeled_e2e: E2E predictions for all N simulated requests (unlabeled).
        alpha: CI error level (1-alpha = confidence).

    Returns:
        RectifiedSimulationMetrics with bias-corrected percentiles and CIs.
    """
    if not _HAS_PPI:
        # Fallback: compute unrectified percentiles from unlabeled data, no CIs
        return RectifiedSimulationMetrics(
            ttft_p50=float(np.percentile(sim_unlabeled_ttft, 50)),
            ttft_p90=float(np.percentile(sim_unlabeled_ttft, 90)),
            ttft_p95=float(np.percentile(sim_unlabeled_ttft, 95)),
            ttft_p99=float(np.percentile(sim_unlabeled_ttft, 99)),
            tbt_p50=float(np.percentile(sim_unlabeled_tbt, 50)),
            tbt_p90=float(np.percentile(sim_unlabeled_tbt, 90)),
            tbt_p95=float(np.percentile(sim_unlabeled_tbt, 95)),
            tbt_p99=float(np.percentile(sim_unlabeled_tbt, 99)),
            e2e_p50=float(np.percentile(sim_unlabeled_e2e, 50)),
            e2e_p90=float(np.percentile(sim_unlabeled_e2e, 90)),
            e2e_p95=float(np.percentile(sim_unlabeled_e2e, 95)),
            e2e_p99=float(np.percentile(sim_unlabeled_e2e, 99)),
            ttft_mean_bias=_compute_bias(real_latencies.ttft_s, simulated_latencies.ttft_s),
            tbt_mean_bias=_compute_bias(real_latencies.tbt_s, simulated_latencies.tbt_s),
            e2e_mean_bias=_compute_bias(real_latencies.e2e_s, simulated_latencies.e2e_s),
        )

    quantiles = [0.5, 0.9, 0.95, 0.99]
    results: dict[str, float | tuple[float, float] | None] = {}

    # Process each metric
    for metric_name, Y, Yhat, Yhat_unlabeled in [
        ("ttft", real_latencies.ttft_s, simulated_latencies.ttft_s, sim_unlabeled_ttft),
        ("tbt", real_latencies.tbt_s, simulated_latencies.tbt_s, sim_unlabeled_tbt),
        ("e2e", real_latencies.e2e_s, simulated_latencies.e2e_s, sim_unlabeled_e2e),
    ]:
        results[f"{metric_name}_mean_bias"] = _compute_bias(Y, Yhat)

        for q in quantiles:
            pct_label = f"p{int(q * 100)}"
            point, ci = _rectify_quantile(Y, Yhat, Yhat_unlabeled, q, alpha)
            results[f"{metric_name}_{pct_label}"] = point
            results[f"{metric_name}_{pct_label}_ci"] = ci

    return RectifiedSimulationMetrics(
        ttft_p50=results["ttft_p50"],
        ttft_p90=results["ttft_p90"],
        ttft_p95=results["ttft_p95"],
        ttft_p99=results["ttft_p99"],
        tbt_p50=results["tbt_p50"],
        tbt_p90=results["tbt_p90"],
        tbt_p95=results["tbt_p95"],
        tbt_p99=results["tbt_p99"],
        e2e_p50=results["e2e_p50"],
        e2e_p90=results["e2e_p90"],
        e2e_p95=results["e2e_p95"],
        e2e_p99=results["e2e_p99"],
        ttft_p50_ci=results["ttft_p50_ci"],
        ttft_p90_ci=results["ttft_p90_ci"],
        ttft_p95_ci=results["ttft_p95_ci"],
        ttft_p99_ci=results["ttft_p99_ci"],
        tbt_p50_ci=results["tbt_p50_ci"],
        tbt_p90_ci=results["tbt_p90_ci"],
        tbt_p95_ci=results["tbt_p95_ci"],
        tbt_p99_ci=results["tbt_p99_ci"],
        e2e_p50_ci=results["e2e_p50_ci"],
        e2e_p90_ci=results["e2e_p90_ci"],
        e2e_p95_ci=results["e2e_p95_ci"],
        e2e_p99_ci=results["e2e_p99_ci"],
        ttft_mean_bias=results["ttft_mean_bias"],
        tbt_mean_bias=results["tbt_mean_bias"],
        e2e_mean_bias=results["e2e_mean_bias"],
    )


__all__ = [
    "RealServingMeasurements",
    "RectifiedSimulationMetrics",
    "SimulatedLatencies",
    "rectify_simulation_metrics",
]
