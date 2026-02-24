"""MLP (Multi-Layer Perceptron) runtime estimator."""

from __future__ import annotations

from typing import Any, Tuple

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase


class MLPEstimator(SklearnEstimatorBase):
    """Runtime estimator using sklearn MLPRegressor."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 128),
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self._hidden_layer_sizes = hidden_layer_sizes
        self._max_iter = max_iter

    def _create_model(self) -> Any:
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=self._hidden_layer_sizes,
            max_iter=self._max_iter,
            random_state=self._random_state,
        )
