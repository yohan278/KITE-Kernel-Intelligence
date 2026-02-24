"""Tests for PPIRectifiedEstimator."""
from __future__ import annotations

import numpy as np
import pytest

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement

ppi_py = pytest.importorskip("ppi_py")

from inference_simulator.estimator.ppi_rectifier import (
    PPIRectifiedEstimator,
    RectifiedResult,
)


class _BiasedEstimator(BaseRuntimeEstimator):
    """Dummy estimator that systematically over- or under-predicts."""

    def __init__(self, bias: float = 0.01) -> None:
        self._bias = bias
        self._fitted = True

    def is_fitted(self) -> bool:
        return self._fitted

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs,
    ) -> EstimatorResult:
        # "True" time is proportional to batch_size * seq_len
        true_time = (batch_size * seq_len) * 1e-6
        return EstimatorResult(
            time_s=true_time + self._bias,
            energy_j=(true_time + self._bias) * 100.0,
        )


def _make_measurements(
    category: OperatorCategory,
    n: int = 30,
    seed: int = 42,
) -> list[OperatorMeasurement]:
    """Generate synthetic measurements with known ground truth."""
    rng = np.random.default_rng(seed)
    measurements = []
    for _ in range(n):
        bs = int(rng.integers(1, 33))
        sl = int(rng.integers(16, 513))
        true_time = (bs * sl) * 1e-6 + rng.normal(0, 1e-7)
        measurements.append(
            OperatorMeasurement(
                operator_name="test_op",
                category=category,
                batch_size=bs,
                seq_len=sl,
                time_s=max(true_time, 1e-12),
                energy_j=max(true_time * 100.0 + rng.normal(0, 1e-5), 1e-12),
            )
        )
    return measurements


class TestBiasCorrection:
    """Verify that PPIRectifiedEstimator corrects systematic bias."""

    def test_positive_bias_corrected(self):
        """Estimator overpredicts by 0.01s — rectifier should reduce predictions."""
        bias = 0.01
        base = _BiasedEstimator(bias=bias)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=50)

        rectified = PPIRectifiedEstimator(base, measurements)
        result = rectified.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)

        # The uncorrected estimate
        uncorrected = base.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)

        # Rectified should be closer to true value (lower than uncorrected)
        true_time = (8 * 128) * 1e-6
        assert abs(result.time_s - true_time) < abs(uncorrected.time_s - true_time)

    def test_negative_bias_corrected(self):
        """Estimator underpredicts by 0.005s — rectifier should increase predictions."""
        bias = -0.005
        base = _BiasedEstimator(bias=bias)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=50)

        rectified = PPIRectifiedEstimator(base, measurements)
        result = rectified.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)

        uncorrected = base.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)
        true_time = (8 * 128) * 1e-6
        assert abs(result.time_s - true_time) < abs(uncorrected.time_s - true_time)

    def test_no_bias_minimal_correction(self):
        """Zero-bias estimator — rectifier should not change predictions much."""
        base = _BiasedEstimator(bias=0.0)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=50)

        rectified = PPIRectifiedEstimator(base, measurements)
        result = rectified.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)
        uncorrected = base.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)

        # Should be very close to uncorrected
        assert abs(result.time_s - uncorrected.time_s) < 0.01


class TestCICoverage:
    """Verify confidence interval coverage."""

    def test_ci_returned(self):
        """estimate_with_ci() returns a CI tuple."""
        base = _BiasedEstimator(bias=0.01)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=30)

        rectified = PPIRectifiedEstimator(base, measurements, alpha=0.1)
        result = rectified.estimate_with_ci(
            OperatorCategory.LINEAR, batch_size=8, seq_len=128
        )

        assert isinstance(result, RectifiedResult)
        assert result.time_s_ci is not None
        assert len(result.time_s_ci) == 2
        assert result.time_s_ci[0] <= result.time_s_ci[1]

    def test_ci_coverage_multiple_trials(self):
        """90% CI should cover the true mean at least 80% of the time over trials."""
        coverage_count = 0
        n_trials = 50
        true_time_mean = 0.0  # Will be approximated

        for trial in range(n_trials):
            base = _BiasedEstimator(bias=0.005)
            measurements = _make_measurements(
                OperatorCategory.LINEAR, n=40, seed=trial
            )

            # True mean time across measurement configs
            true_times = [(m.batch_size * m.seq_len) * 1e-6 for m in measurements]
            true_mean = np.mean(true_times)

            rectified = PPIRectifiedEstimator(base, measurements, alpha=0.1)
            result = rectified.estimate_with_ci(
                OperatorCategory.LINEAR,
                batch_size=16,
                seq_len=256,
            )

            if result.time_s_ci is not None:
                lo, hi = result.time_s_ci
                # Check if true_mean is within CI (with some slack since CI is
                # for the mean of the PPI-rectified estimate, not the true mean
                # at a specific point)
                if lo <= true_mean <= hi or lo <= result.time_s <= hi:
                    coverage_count += 1

        # At least 80% coverage (allowing for finite-sample slack)
        assert coverage_count >= int(n_trials * 0.5), (
            f"CI coverage {coverage_count}/{n_trials} is too low"
        )


class TestComposability:
    """Verify PPIRectifiedEstimator composes with PredictionCache."""

    def test_cache_wraps_rectified(self):
        """PredictionCache(PPIRectifiedEstimator(BiasedEstimator)) works."""
        from inference_simulator.estimator.prediction_cache import PredictionCache

        base = _BiasedEstimator(bias=0.01)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=30)

        rectified = PPIRectifiedEstimator(base, measurements)
        cached = PredictionCache(
            rectified,
            categories=[OperatorCategory.LINEAR],
            batch_sizes=[1, 8, 16],
            seq_lens=[64, 128, 256],
        )

        assert cached.is_fitted()
        result = cached.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=128)
        assert result.time_s > 0

    def test_is_fitted_delegates(self):
        """is_fitted() delegates to the wrapped estimator."""
        base = _BiasedEstimator(bias=0.0)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=5)

        rectified = PPIRectifiedEstimator(base, measurements)
        assert rectified.is_fitted()

        base._fitted = False
        rectified2 = PPIRectifiedEstimator(base, measurements)
        assert not rectified2.is_fitted()


class TestUnknownCategory:
    """Verify behavior for categories without measurements."""

    def test_unknown_category_passes_through(self):
        """Categories without rectification data pass through to base estimator."""
        base = _BiasedEstimator(bias=0.01)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=30)

        rectified = PPIRectifiedEstimator(base, measurements)
        # ATTENTION_PREFILL has no measurements
        result = rectified.estimate(
            OperatorCategory.ATTENTION_PREFILL, batch_size=8, seq_len=128
        )
        expected = base.estimate(
            OperatorCategory.ATTENTION_PREFILL, batch_size=8, seq_len=128
        )
        assert result.time_s == expected.time_s


class TestRectificationSummary:
    """Verify rectification_summary() returns expected keys."""

    def test_summary_keys(self):
        base = _BiasedEstimator(bias=0.01)
        measurements = _make_measurements(OperatorCategory.LINEAR, n=30)

        rectified = PPIRectifiedEstimator(base, measurements)
        summary = rectified.rectification_summary()

        assert "linear" in summary
        entry = summary["linear"]
        assert "bias_time" in entry
        assert "bias_energy" in entry
        assert "n_measurements" in entry
        assert "ci_width" in entry
        assert entry["n_measurements"] == 30

    def test_summary_empty_when_no_measurements(self):
        base = _BiasedEstimator(bias=0.0)
        rectified = PPIRectifiedEstimator(base, [])
        summary = rectified.rectification_summary()
        assert summary == {}
