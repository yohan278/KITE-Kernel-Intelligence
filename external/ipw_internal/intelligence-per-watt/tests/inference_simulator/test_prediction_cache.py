"""Tests for prediction cache."""
from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


@pytest.fixture
def synthetic_measurements():
    measurements = []
    for bs in [1, 2, 4, 8]:
        for sl in [128, 256, 512]:
            tokens = bs * sl
            measurements.append(OperatorMeasurement(
                operator_name="linear", category=OperatorCategory.LINEAR,
                batch_size=bs, seq_len=sl, time_s=tokens * 1e-6,
            ))
    return measurements


class TestPredictionCache:
    @pytest.fixture(autouse=True)
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_cache_hit(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        from inference_simulator.estimator.prediction_cache import PredictionCache

        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        cache = PredictionCache(
            estimator=est,
            categories=[OperatorCategory.LINEAR],
            batch_sizes=[1, 2, 4, 8],
            seq_lens=[128, 256, 512],
        )
        assert cache.is_fitted()
        assert cache.cache_size > 0

        result = cache.estimate(OperatorCategory.LINEAR, batch_size=4, seq_len=256)
        assert result.time_s > 0

    def test_cache_miss_fallback(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        from inference_simulator.estimator.prediction_cache import PredictionCache

        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        cache = PredictionCache(
            estimator=est,
            categories=[OperatorCategory.LINEAR],
            batch_sizes=[1, 4],
            seq_lens=[128, 512],
        )

        # Query a point not in the cache grid
        result = cache.estimate(OperatorCategory.LINEAR, batch_size=2, seq_len=256)
        assert result.time_s > 0  # Should fall back to nearest or estimator

    def test_categories_tracked(self, synthetic_measurements):
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
        from inference_simulator.estimator.prediction_cache import PredictionCache

        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        cache = PredictionCache(
            estimator=est,
            categories=[OperatorCategory.LINEAR],
            batch_sizes=[1, 4],
            seq_lens=[128],
        )
        assert "linear" in cache.categories
