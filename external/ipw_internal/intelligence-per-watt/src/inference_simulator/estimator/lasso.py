"""Lasso regression runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class LassoRegressionEstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn Lasso regression."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42, poly_degree: int = 0) -> None:
        super().__init__(random_state=random_state, poly_degree=poly_degree)
        self._alpha = alpha

    def _create_model(self) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(alpha=self._alpha, random_state=self._random_state)
