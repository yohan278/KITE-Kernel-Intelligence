"""Gaussian Process runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class GaussianProcessEstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn GaussianProcessRegressor with Matern kernel."""

    def __init__(self, random_state: int = 42) -> None:
        super().__init__(random_state=random_state)

    def _create_model(self) -> Any:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        return GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            random_state=self._random_state,
            normalize_y=True,
        )
