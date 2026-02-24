"""Abstract base class for sklearn-based runtime estimators."""

from __future__ import annotations

import csv
import math
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement

# All categories used for one-hot encoding
_CATEGORY_VALUES = [cat.value for cat in OperatorCategory]


class SklearnEstimatorBase(BaseRuntimeEstimator):
    """Abstract base for sklearn-backed runtime estimators.

    Provides shared logic for feature engineering, dataset construction,
    training, and evaluation. Subclasses only need to implement
    ``_create_model()`` to return a configured sklearn estimator.
    """

    # Number of numeric prefix features: [batch, seq, log2(batch), log2(seq), batch*seq]
    _NUM_NUMERIC_FEATURES = 5

    def __init__(self, random_state: int = 42, poly_degree: int = 0) -> None:
        self._random_state = random_state
        self._poly_degree = poly_degree
        self._poly_transformer: Any = None
        self._time_model: Any = None
        self._energy_model: Any = None
        self._power_model: Any = None
        self._fitted = False
        self._has_energy = False
        self._has_power = False
        self._train_score: Optional[float] = None
        self._val_score: Optional[float] = None

    def _fit_poly_transformer(self, X: np.ndarray) -> np.ndarray:
        """Apply polynomial expansion to numeric prefix features.

        Only expands the first _NUM_NUMERIC_FEATURES columns, preserving
        one-hot category and other features unchanged.
        """
        if self._poly_degree < 2:
            return X
        from sklearn.preprocessing import PolynomialFeatures

        n = self._NUM_NUMERIC_FEATURES
        X_numeric = X[:, :n]
        X_rest = X[:, n:]
        self._poly_transformer = PolynomialFeatures(
            degree=self._poly_degree, include_bias=False
        )
        X_numeric_poly = self._poly_transformer.fit_transform(X_numeric)
        return np.hstack([X_numeric_poly, X_rest])

    def _transform_poly(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted polynomial expansion at prediction time."""
        if self._poly_transformer is None or self._poly_degree < 2:
            return X
        n = self._NUM_NUMERIC_FEATURES
        X_numeric = X[:, :n]
        X_rest = X[:, n:]
        X_numeric_poly = self._poly_transformer.transform(X_numeric)
        return np.hstack([X_numeric_poly, X_rest])

    @abstractmethod
    def _create_model(self) -> Any:
        """Return a fresh, unfitted sklearn estimator instance.

        Implementations should use deferred imports so the package works
        without optional ML dependencies installed.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def train_score(self) -> Optional[float]:
        """R^2 score on training data (None if not fitted)."""
        return self._train_score

    @property
    def val_score(self) -> Optional[float]:
        """R^2 score on validation data (None if not fitted)."""
        return self._val_score

    def fit(
        self,
        measurements: Sequence[OperatorMeasurement],
        model_dims: Optional[Dict[str, float]] = None,
        val_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """Train the estimator on profiling measurements.

        Args:
            measurements: List of OperatorMeasurement from profiling.
            model_dims: Optional dict of model dimensions to include as features.
            val_fraction: Fraction of data to hold out for validation.

        Returns:
            Dict with train/val R^2 scores for each target.
        """
        from sklearn.model_selection import train_test_split

        if len(measurements) < 2:
            raise ValueError("Need at least 2 measurements to train")

        X, y_time, y_energy, y_power = self._build_dataset(measurements, model_dims)
        X = self._fit_poly_transformer(X)

        # Train/val split
        if len(X) < 5:
            X_train, X_val = X, X
            y_time_train, y_time_val = y_time, y_time
            idx_train: List[int] = list(range(len(X)))
        else:
            X_train, X_val, y_time_train, y_time_val, idx_train, _idx_val = (
                train_test_split(
                    X, y_time, list(range(len(X))),
                    test_size=val_fraction,
                    random_state=self._random_state,
                )
            )

        # Time model (always trained)
        self._time_model = self._create_model()
        self._time_model.fit(X_train, y_time_train)

        scores: Dict[str, float] = {
            "time_train_r2": self._time_model.score(X_train, y_time_train),
            "time_val_r2": self._time_model.score(X_val, y_time_val),
        }
        self._train_score = scores["time_train_r2"]
        self._val_score = scores["time_val_r2"]

        # Energy model (if energy data available)
        energy_mask = ~np.isnan(y_energy)
        if energy_mask.sum() >= 2:
            self._has_energy = True
            train_mask = energy_mask[idx_train] if len(idx_train) < len(energy_mask) else energy_mask
            X_energy_train = X_train[train_mask[: len(X_train)]]
            y_energy_train = (
                y_energy[idx_train][train_mask[: len(X_train)]]
                if len(idx_train) < len(y_energy)
                else y_energy[train_mask]
            )

            if len(X_energy_train) >= 2:
                self._energy_model = self._create_model()
                self._energy_model.fit(X_energy_train, y_energy_train)
                scores["energy_train_r2"] = self._energy_model.score(
                    X_energy_train, y_energy_train
                )

        # Power model (if power data available)
        power_mask = ~np.isnan(y_power)
        if power_mask.sum() >= 2:
            self._has_power = True
            train_mask = power_mask[idx_train] if len(idx_train) < len(power_mask) else power_mask
            X_power_train = X_train[train_mask[: len(X_train)]]
            y_power_train = (
                y_power[idx_train][train_mask[: len(X_train)]]
                if len(idx_train) < len(y_power)
                else y_power[train_mask]
            )

            if len(X_power_train) >= 2:
                self._power_model = self._create_model()
                self._power_model.fit(X_power_train, y_power_train)
                scores["power_train_r2"] = self._power_model.score(
                    X_power_train, y_power_train
                )

        self._fitted = True
        return scores

    def fit_from_csv(
        self,
        csv_paths: Sequence[Tuple[Path, OperatorCategory]],
        model_dims: Optional[Dict[str, float]] = None,
        val_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """Train from profiling CSV files.

        Args:
            csv_paths: List of (csv_path, category) tuples.
            model_dims: Optional model dimension features.
            val_fraction: Validation split fraction.

        Returns:
            Training scores dict.
        """
        measurements: List[OperatorMeasurement] = []
        for csv_path, category in csv_paths:
            measurements.extend(load_csv_measurements(csv_path, category))

        return self.fit(measurements, model_dims, val_fraction)

    def fit_cross_model(
        self,
        measurements_by_model: Dict[str, Sequence[OperatorMeasurement]],
        holdout_model: str,
        model_dims_by_model: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Leave-one-model-out training.

        Trains on all models except ``holdout_model``, evaluates on the held-out model.

        Args:
            measurements_by_model: {model_name: measurements} mapping.
            holdout_model: Name of the model to hold out for evaluation.
            model_dims_by_model: Optional per-model dimension dicts.

        Returns:
            Dict with R^2 scores on the holdout model.
        """
        train_measurements: List[OperatorMeasurement] = []
        holdout_measurements: List[OperatorMeasurement] = []

        for model_name, ms in measurements_by_model.items():
            if model_name == holdout_model:
                holdout_measurements.extend(ms)
            else:
                train_measurements.extend(ms)

        if len(train_measurements) < 2:
            raise ValueError("Need at least 2 training measurements")
        if len(holdout_measurements) == 0:
            raise ValueError(f"No measurements for holdout model '{holdout_model}'")

        # Train on non-holdout data (no val split — the holdout IS validation)
        train_dims = (
            model_dims_by_model.get(list(measurements_by_model.keys())[0])
            if model_dims_by_model
            else None
        )
        self.fit(train_measurements, model_dims=train_dims, val_fraction=0.0)

        # Evaluate on holdout
        holdout_dims = (
            model_dims_by_model.get(holdout_model) if model_dims_by_model else None
        )
        X_holdout, y_time_holdout, y_energy_holdout, y_power_holdout = (
            self._build_dataset(holdout_measurements, holdout_dims)
        )
        X_holdout = self._transform_poly(X_holdout)

        scores: Dict[str, float] = {
            "holdout_model": holdout_model,  # type: ignore[dict-item]
            "time_holdout_r2": self._time_model.score(X_holdout, y_time_holdout),
        }
        if self._energy_model is not None:
            energy_mask = ~np.isnan(y_energy_holdout)
            if energy_mask.sum() >= 1:
                scores["energy_holdout_r2"] = self._energy_model.score(
                    X_holdout[energy_mask], y_energy_holdout[energy_mask]
                )
        if self._power_model is not None:
            power_mask = ~np.isnan(y_power_holdout)
            if power_mask.sum() >= 1:
                scores["power_holdout_r2"] = self._power_model.score(
                    X_holdout[power_mask], y_power_holdout[power_mask]
                )

        return scores

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Predict runtime using the trained model."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted")

        features = self._build_features(
            operator_category, batch_size, seq_len, kwargs.get("model_dims")
        )
        X = np.array([features])
        X = self._transform_poly(X)

        # Floor predictions at a small epsilon to avoid returning exactly 0.0
        # for linear models that may predict slightly negative values
        _EPS = 1e-12
        time_s = max(float(self._time_model.predict(X)[0]), _EPS)

        energy_j = None
        if self._energy_model is not None:
            energy_j = max(float(self._energy_model.predict(X)[0]), _EPS)

        power_w = None
        if self._power_model is not None:
            power_w = max(float(self._power_model.predict(X)[0]), _EPS)

        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=power_w)

    # ------------------------------------------------------------------
    # Feature engineering (shared by all sklearn estimators)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features(
        category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        model_dims: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """Build feature vector for a single data point.

        Features: [batch_size, seq_len, log2(batch_size), log2(seq_len),
                   batch_size*seq_len, <one-hot category>, <model_dims>]
        """
        features: List[float] = [
            float(batch_size),
            float(seq_len),
            math.log2(max(batch_size, 1)),
            math.log2(max(seq_len, 1)),
            float(batch_size * seq_len),
        ]

        # One-hot encode category
        cat_onehot = [0.0] * len(_CATEGORY_VALUES)
        try:
            idx = _CATEGORY_VALUES.index(category.value)
            cat_onehot[idx] = 1.0
        except ValueError:
            pass
        features.extend(cat_onehot)

        # Category-specific features (Vidur-inspired: O(n^2) attention cost)
        if category == OperatorCategory.ATTENTION_PREFILL:
            features.append(float(seq_len) ** 2)  # quadratic attention cost
        else:
            features.append(0.0)

        # Model dimensions (optional)
        if model_dims:
            for key in sorted(model_dims.keys()):
                features.append(float(model_dims[key]))

        return features

    @staticmethod
    def _build_dataset(
        measurements: Sequence[OperatorMeasurement],
        model_dims: Optional[Dict[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix and target arrays from measurements."""
        X_list: List[List[float]] = []
        y_time: List[float] = []
        y_energy: List[float] = []
        y_power: List[float] = []

        for m in measurements:
            features = SklearnEstimatorBase._build_features(
                m.category, m.batch_size, m.seq_len, model_dims
            )
            X_list.append(features)
            y_time.append(m.time_s)
            y_energy.append(m.energy_j if m.energy_j is not None else float("nan"))
            y_power.append(m.power_w if m.power_w is not None else float("nan"))

        return (
            np.array(X_list),
            np.array(y_time),
            np.array(y_energy),
            np.array(y_power),
        )


# ------------------------------------------------------------------
# CSV loading helpers (shared by all estimators)
# ------------------------------------------------------------------


def load_csv_measurements(
    csv_path: Path, category: OperatorCategory
) -> List[OperatorMeasurement]:
    """Load OperatorMeasurements from a profiling CSV."""
    csv_path = Path(csv_path)
    measurements: List[OperatorMeasurement] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = row.get("batch_size")
            sl = row.get("seq_len")
            ts = row.get("time_s")
            if bs is None or sl is None or ts is None:
                continue  # Skip rows from CSVs lacking required columns
            energy_j = _parse_opt(row.get("energy_j"))
            power_w = _parse_opt(row.get("power_w"))
            flops = _parse_opt_int(row.get("flops"))

            measurements.append(
                OperatorMeasurement(
                    operator_name=row.get("operator_name", "unknown"),
                    category=category,
                    batch_size=int(bs),
                    seq_len=int(sl),
                    time_s=float(ts),
                    energy_j=energy_j,
                    power_w=power_w,
                    flops=flops,
                )
            )

    return measurements


def load_csv_measurements_auto_category(
    csv_path: Path,
    operator_name_to_category: Dict[str, OperatorCategory],
) -> List[OperatorMeasurement]:
    """Load OperatorMeasurements from a combined CSV, inferring category per row."""
    csv_path = Path(csv_path)
    measurements: List[OperatorMeasurement] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = row.get("operator_name", "unknown")
            category = operator_name_to_category.get(op_name)
            if category is None:
                continue  # Skip unknown operators

            energy_j = _parse_opt(row.get("energy_j"))
            power_w = _parse_opt(row.get("power_w"))
            flops = _parse_opt_int(row.get("flops"))

            measurements.append(
                OperatorMeasurement(
                    operator_name=op_name,
                    category=category,
                    batch_size=int(row["batch_size"]),
                    seq_len=int(row["seq_len"]),
                    time_s=float(row["time_s"]),
                    energy_j=energy_j,
                    power_w=power_w,
                    flops=flops,
                )
            )

    return measurements


def _parse_opt(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_opt_int(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None
