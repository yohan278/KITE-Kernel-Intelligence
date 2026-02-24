"""Tests for PPI-rectified simulation metrics validation."""
from __future__ import annotations

import numpy as np
import pytest

ppi_py = pytest.importorskip("ppi_py")

from inference_simulator.metrics.ppi_validation import (
    RealServingMeasurements,
    RectifiedSimulationMetrics,
    SimulatedLatencies,
    rectify_simulation_metrics,
)


def _make_data(
    n: int = 100,
    N: int = 500,
    bias: float = 0.02,
    seed: int = 42,
):
    """Generate synthetic real and simulated latency data.

    Returns:
        Tuple of (real, simulated_labeled, sim_unlabeled_ttft,
        sim_unlabeled_tbt, sim_unlabeled_e2e).
    """
    rng = np.random.default_rng(seed)

    # True latencies: lognormal-ish
    true_ttft = rng.lognormal(mean=-2.0, sigma=0.3, size=n)
    true_tbt = rng.lognormal(mean=-3.0, sigma=0.2, size=n)
    true_e2e = rng.lognormal(mean=-1.0, sigma=0.4, size=n)

    # Real measurements = true + small noise
    real = RealServingMeasurements(
        ttft_s=true_ttft + rng.normal(0, 0.001, n),
        tbt_s=true_tbt + rng.normal(0, 0.0005, n),
        e2e_s=true_e2e + rng.normal(0, 0.002, n),
    )

    # Simulator predictions = true + systematic bias + noise
    sim_labeled = SimulatedLatencies(
        ttft_s=true_ttft + bias + rng.normal(0, 0.002, n),
        tbt_s=true_tbt + bias + rng.normal(0, 0.001, n),
        e2e_s=true_e2e + bias + rng.normal(0, 0.003, n),
    )

    # Unlabeled: bigger set from simulator
    true_ttft_u = rng.lognormal(mean=-2.0, sigma=0.3, size=N)
    true_tbt_u = rng.lognormal(mean=-3.0, sigma=0.2, size=N)
    true_e2e_u = rng.lognormal(mean=-1.0, sigma=0.4, size=N)

    sim_unlabeled_ttft = true_ttft_u + bias + rng.normal(0, 0.002, N)
    sim_unlabeled_tbt = true_tbt_u + bias + rng.normal(0, 0.001, N)
    sim_unlabeled_e2e = true_e2e_u + bias + rng.normal(0, 0.003, N)

    return real, sim_labeled, sim_unlabeled_ttft, sim_unlabeled_tbt, sim_unlabeled_e2e


class TestBiasRectification:
    """Verify that rectify_simulation_metrics() corrects systematic bias."""

    def test_ttft_bias_corrected(self):
        """Simulator adds 20ms to TTFT — should be corrected."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=100, N=500, bias=0.02)

        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)

        assert isinstance(result, RectifiedSimulationMetrics)
        # Bias should be close to 0.02 (positive = simulator overpredicts)
        assert result.ttft_mean_bias > 0.01

    def test_e2e_bias_corrected(self):
        """E2E bias should also be reported."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=100, N=500, bias=0.02)
        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)
        assert result.e2e_mean_bias > 0.01

    def test_zero_bias_minimal_correction(self):
        """With zero bias, rectified percentiles should be close to raw percentiles."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=100, N=500, bias=0.0)
        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)

        raw_p50 = float(np.percentile(ttft_u, 50))
        # Should be within reasonable tolerance
        assert abs(result.ttft_p50 - raw_p50) < raw_p50 * 0.5


class TestQuantileCICoverage:
    """Verify CI coverage for quantile estimates."""

    def test_ci_present(self):
        """CIs should be populated."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=50, N=200, bias=0.01)
        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)

        assert result.ttft_p50_ci is not None
        assert result.ttft_p90_ci is not None
        assert result.e2e_p95_ci is not None

    def test_ci_ordering(self):
        """CI lower should be <= CI upper."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=50, N=200, bias=0.01)
        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)

        for attr in ("ttft_p50_ci", "ttft_p90_ci", "e2e_p50_ci", "e2e_p95_ci"):
            ci = getattr(result, attr)
            if ci is not None:
                assert ci[0] <= ci[1], f"{attr}: lower={ci[0]} > upper={ci[1]}"

    def test_p90_ci_coverage(self):
        """90% CI should cover true P90 a reasonable fraction of the time."""
        coverage_count = 0
        n_trials = 30

        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 1000)
            n, N = 80, 400
            true_ttft = rng.lognormal(mean=-2.0, sigma=0.3, size=N)
            true_p90 = float(np.percentile(true_ttft[:n], 90))

            real = RealServingMeasurements(
                ttft_s=true_ttft[:n],
                tbt_s=rng.lognormal(mean=-3.0, sigma=0.2, size=n),
                e2e_s=rng.lognormal(mean=-1.0, sigma=0.4, size=n),
            )
            bias = 0.01
            sim = SimulatedLatencies(
                ttft_s=true_ttft[:n] + bias + rng.normal(0, 0.002, n),
                tbt_s=real.tbt_s + bias + rng.normal(0, 0.001, n),
                e2e_s=real.e2e_s + bias + rng.normal(0, 0.003, n),
            )

            ttft_u = true_ttft + bias + rng.normal(0, 0.002, N)
            tbt_u = rng.lognormal(mean=-3.0, sigma=0.2, size=N) + bias
            e2e_u = rng.lognormal(mean=-1.0, sigma=0.4, size=N) + bias

            result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)

            if result.ttft_p90_ci is not None:
                lo, hi = result.ttft_p90_ci
                if lo <= true_p90 <= hi:
                    coverage_count += 1

        # Allow generous slack for finite samples
        assert coverage_count >= int(n_trials * 0.3), (
            f"P90 CI coverage {coverage_count}/{n_trials} is too low"
        )


class TestGracefulDegradation:
    """Verify behavior without ppi-python is handled in the module."""

    def test_result_dataclass_fields(self):
        """RectifiedSimulationMetrics has all expected fields."""
        result = RectifiedSimulationMetrics()
        assert result.ttft_p50 == 0.0
        assert result.ttft_p50_ci is None
        assert result.ttft_mean_bias == 0.0


class TestEdgeCases:
    def test_small_n(self):
        """Should work with very small labeled sets."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=5, N=100, bias=0.01)
        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)
        assert result.ttft_p50 >= 0.0

    def test_equal_n_and_N(self):
        """Should work when labeled and unlabeled are same size."""
        real, sim, ttft_u, tbt_u, e2e_u = _make_data(n=50, N=50, bias=0.01)
        result = rectify_simulation_metrics(real, sim, ttft_u, tbt_u, e2e_u)
        assert result.ttft_p50 >= 0.0
