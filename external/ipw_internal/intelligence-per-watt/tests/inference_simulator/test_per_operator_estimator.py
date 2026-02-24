"""Tests for per-operator-type estimator."""
from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


@pytest.fixture
def synthetic_measurements():
    measurements = []
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [128, 256, 512, 1024]:
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6
            measurements.append(OperatorMeasurement(
                operator_name="linear_qkv", category=OperatorCategory.LINEAR,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time, energy_j=base_time * 400, power_w=400.0,
            ))
            measurements.append(OperatorMeasurement(
                operator_name="attention_prefill", category=OperatorCategory.ATTENTION_PREFILL,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time * 1.5 * (seq_len / 128),  # quadratic-ish
                energy_j=base_time * 1.5 * 400,
            ))
            measurements.append(OperatorMeasurement(
                operator_name="attention_decode", category=OperatorCategory.ATTENTION_DECODE,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time * 0.3, energy_j=base_time * 0.3 * 350,
            ))
    return measurements


class TestPerOperatorEstimator:
    @pytest.fixture(autouse=True)
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_fit_and_predict(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        est = PerOperatorEstimator()
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted()
        assert len(scores) > 0
        # At least LINEAR, ATTENTION_PREFILL, ATTENTION_DECODE should have models
        assert any("linear" in k for k in scores)
        assert any("attention_prefill" in k for k in scores)

    def test_estimate_per_category(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        for cat in [OperatorCategory.LINEAR, OperatorCategory.ATTENTION_PREFILL, OperatorCategory.ATTENTION_DECODE]:
            result = est.estimate(cat, batch_size=4, seq_len=256)
            assert result.time_s > 0

    def test_prefill_captures_quadratic(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        # Longer seq should take more time for attention prefill
        short = est.estimate(OperatorCategory.ATTENTION_PREFILL, batch_size=1, seq_len=128)
        long = est.estimate(OperatorCategory.ATTENTION_PREFILL, batch_size=1, seq_len=1024)
        assert long.time_s > short.time_s

    def test_energy_prediction(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)
        result = est.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)
        assert result.energy_j is not None
        assert result.energy_j > 0
