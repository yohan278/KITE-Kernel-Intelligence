"""Random forest runtime estimator trained on profiling data."""

from __future__ import annotations

from typing import Any, Optional

from inference_simulator.estimator.sklearn_base import (
    SklearnEstimatorBase,
    load_csv_measurements,
    _parse_opt,
    _parse_opt_int,
)


class RandomForestEstimator(SklearnEstimatorBase):
    """Runtime estimator using scikit-learn RandomForestRegressor.

    Trains separate models for time_s, energy_j, and power_w targets.
    Features: batch_size, seq_len, operator_category (one-hot), and
    optional model dimensions (hidden_dim, num_heads, etc.).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self._n_estimators = n_estimators
        self._max_depth = max_depth

    def _create_model(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
        )


# Re-export helpers for backward compatibility
_load_csv_measurements = load_csv_measurements
