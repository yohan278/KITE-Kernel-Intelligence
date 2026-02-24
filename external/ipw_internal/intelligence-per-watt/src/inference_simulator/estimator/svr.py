"""Support Vector Regression runtime estimator."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class SVREstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn SVR."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self._kernel = kernel
        self._C = C

    def _create_model(self) -> Any:
        from sklearn.svm import SVR

        return SVR(kernel=self._kernel, C=self._C)
