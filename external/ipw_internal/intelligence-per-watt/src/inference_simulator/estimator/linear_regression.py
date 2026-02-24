"""Linear regression runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class LinearRegressionEstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn LinearRegression."""

    def __init__(self, random_state: int = 42) -> None:
        super().__init__(random_state=random_state)

    def _create_model(self) -> Any:
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
