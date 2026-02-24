"""LightGBM runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class LightGBMEstimator(SklearnEstimatorBase):
    """Runtime estimator using LGBMRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate

    def _create_model(self) -> Any:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            verbose=-1,
        )
