"""Multi-output estimator wrapper for joint (time, energy, power) prediction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


def _resolve_estimator(
    estimator: Union[SklearnEstimatorBase, str],
) -> SklearnEstimatorBase:
    """Resolve an estimator from a class instance or a string name."""
    if isinstance(estimator, SklearnEstimatorBase):
        return estimator

    name_map = _get_estimator_name_map()
    key = estimator.lower().replace("-", "_")
    if key not in name_map:
        raise ValueError(
            f"Unknown estimator type '{estimator}'. "
            f"Available: {sorted(name_map.keys())}"
        )
    return name_map[key]()


def _get_estimator_name_map() -> Dict[str, type]:
    """Lazy mapping from short name to estimator class."""
    from inference_simulator.estimator.random_forest import RandomForestEstimator
    from inference_simulator.estimator.linear_regression import LinearRegressionEstimator
    from inference_simulator.estimator.ridge import RidgeRegressionEstimator
    from inference_simulator.estimator.lasso import LassoRegressionEstimator
    from inference_simulator.estimator.knn import KNNEstimator
    from inference_simulator.estimator.bayesian_linear import BayesianLinearEstimator
    from inference_simulator.estimator.mlp import MLPEstimator
    from inference_simulator.estimator.svr import SVREstimator
    from inference_simulator.estimator.gaussian_process import GaussianProcessEstimator

    mapping: Dict[str, type] = {
        "random_forest": RandomForestEstimator,
        "linear_regression": LinearRegressionEstimator,
        "ridge": RidgeRegressionEstimator,
        "lasso": LassoRegressionEstimator,
        "knn": KNNEstimator,
        "bayesian_linear": BayesianLinearEstimator,
        "mlp": MLPEstimator,
        "svr": SVREstimator,
        "gaussian_process": GaussianProcessEstimator,
    }

    try:
        from inference_simulator.estimator.xgboost_estimator import XGBoostEstimator
        mapping["xgboost"] = XGBoostEstimator
    except ImportError:
        pass

    try:
        from inference_simulator.estimator.lightgbm_estimator import LightGBMEstimator
        mapping["lightgbm"] = LightGBMEstimator
    except ImportError:
        pass

    return mapping


class MultiOutputEstimatorWrapper(BaseRuntimeEstimator):
    """Wraps a SklearnEstimatorBase to predict (time_s, energy_j, power_w) jointly.

    Uses sklearn MultiOutputRegressor to train a single model that produces
    all three targets simultaneously, rather than training separate models.

    Args:
        base_estimator: An instance of SklearnEstimatorBase, or a string name
            (e.g., ``"random_forest"``, ``"ridge"``).
        base_estimator_type: Alias for base_estimator (string name).
    """

    def __init__(
        self,
        base_estimator: Union[SklearnEstimatorBase, str, None] = None,
        *,
        base_estimator_type: Optional[str] = None,
    ) -> None:
        spec = base_estimator_type or base_estimator
        if spec is None:
            raise ValueError("Must provide base_estimator or base_estimator_type")
        self._base = _resolve_estimator(spec)
        self._multi_model: Any = None
        self._fitted = False
        self._has_energy = False
        self._has_power = False
        self._train_score: Optional[float] = None
        self._val_score: Optional[float] = None

    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def train_score(self) -> Optional[float]:
        return self._train_score

    @property
    def val_score(self) -> Optional[float]:
        return self._val_score

    def fit(
        self,
        measurements: Sequence[OperatorMeasurement],
        model_dims: Optional[Dict[str, float]] = None,
        val_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """Train the multi-output model on profiling measurements.

        Rows with missing energy or power values use 0.0 as fill;
        the corresponding output columns are flagged so estimate()
        knows not to return them.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.multioutput import MultiOutputRegressor

        if len(measurements) < 2:
            raise ValueError("Need at least 2 measurements to train")

        X, y_time, y_energy, y_power = SklearnEstimatorBase._build_dataset(
            measurements, model_dims
        )

        # Check which optional targets have data
        energy_valid = ~np.isnan(y_energy)
        power_valid = ~np.isnan(y_power)
        self._has_energy = energy_valid.sum() >= 2
        self._has_power = power_valid.sum() >= 2

        # Build combined target matrix; fill NaNs with 0 for multi-output fitting
        y_energy_filled = np.where(np.isnan(y_energy), 0.0, y_energy)
        y_power_filled = np.where(np.isnan(y_power), 0.0, y_power)

        targets: List[np.ndarray] = [y_time]
        if self._has_energy:
            targets.append(y_energy_filled)
        if self._has_power:
            targets.append(y_power_filled)

        Y = np.column_stack(targets)

        # Train/val split
        if len(X) < 5:
            X_train, X_val, Y_train, Y_val = X, X, Y, Y
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X, Y, test_size=val_fraction, random_state=self._base._random_state
            )

        self._multi_model = MultiOutputRegressor(self._base._create_model())
        self._multi_model.fit(X_train, Y_train)

        scores: Dict[str, float] = {
            "multi_train_r2": self._multi_model.score(X_train, Y_train),
            "multi_val_r2": self._multi_model.score(X_val, Y_val),
        }
        self._train_score = scores["multi_train_r2"]
        self._val_score = scores["multi_val_r2"]
        self._fitted = True
        return scores

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        if not self._fitted:
            raise RuntimeError("MultiOutputEstimatorWrapper is not fitted")

        features = SklearnEstimatorBase._build_features(
            operator_category, batch_size, seq_len, kwargs.get("model_dims")
        )
        X = np.array([features])
        preds = self._multi_model.predict(X)[0]

        time_s = max(float(preds[0]), 0.0)

        idx = 1
        energy_j: Optional[float] = None
        if self._has_energy:
            energy_j = max(float(preds[idx]), 0.0)
            idx += 1

        power_w: Optional[float] = None
        if self._has_power:
            power_w = max(float(preds[idx]), 0.0)

        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=power_w)
