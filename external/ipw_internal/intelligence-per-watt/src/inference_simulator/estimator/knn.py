"""K-Nearest Neighbors runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class KNNEstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn KNeighborsRegressor."""

    def __init__(self, n_neighbors: int = 5, random_state: int = 42) -> None:
        super().__init__(random_state=random_state)
        self._n_neighbors = n_neighbors

    def _create_model(self) -> Any:
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(n_neighbors=self._n_neighbors)
