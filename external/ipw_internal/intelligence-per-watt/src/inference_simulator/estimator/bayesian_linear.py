"""Bayesian linear regression runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class BayesianLinearEstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn BayesianRidge."""

    def __init__(self, random_state: int = 42) -> None:
        super().__init__(random_state=random_state)

    def _create_model(self) -> Any:
        from sklearn.linear_model import BayesianRidge

        return BayesianRidge()
