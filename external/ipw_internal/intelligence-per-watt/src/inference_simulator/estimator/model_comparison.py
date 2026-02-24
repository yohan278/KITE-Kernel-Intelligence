"""Utilities for comparing estimator accuracy across models and targets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type, Union

import numpy as np

from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


def _resolve_classes(
    estimator_classes: Sequence[Union[Type[SklearnEstimatorBase], str]],
) -> List[Type[SklearnEstimatorBase]]:
    """Resolve a mix of class types and string names to class types."""
    from inference_simulator.estimator.multi_output import _get_estimator_name_map

    name_map = _get_estimator_name_map()
    resolved: List[Type[SklearnEstimatorBase]] = []
    for item in estimator_classes:
        if isinstance(item, str):
            key = item.lower().replace("-", "_")
            if key not in name_map:
                raise ValueError(f"Unknown estimator type '{item}'")
            resolved.append(name_map[key])
        else:
            resolved.append(item)
    return resolved


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute R2, MAE, RMSE for a single target."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def compare_estimators(
    measurements: Sequence[OperatorMeasurement],
    model_dims: Optional[Dict[str, float]],
    estimator_classes: Sequence[Union[Type[SklearnEstimatorBase], str]],
    val_fraction: float = 0.2,
) -> List[Dict[str, Any]]:
    """Train and evaluate multiple estimator classes on the same dataset.

    Args:
        measurements: Profiling measurements.
        model_dims: Optional model dimension features.
        estimator_classes: List of SklearnEstimatorBase subclass types or string
            names (e.g., ``"random_forest"``, ``"ridge"``).
        val_fraction: Validation split fraction.

    Returns:
        List of dicts, each containing ``"estimator"`` (class name) and
        per-target metrics (``time_r2``, ``time_mae``, ``time_rmse``, etc.).
    """
    from sklearn.model_selection import train_test_split

    resolved = _resolve_classes(estimator_classes)

    X, y_time, y_energy, y_power = SklearnEstimatorBase._build_dataset(
        measurements, model_dims
    )

    if len(X) < 5:
        val_idx = list(range(len(X)))
    else:
        indices = list(range(len(X)))
        _train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, random_state=42
        )

    y_time_val = y_time[val_idx]
    y_energy_val = y_energy[val_idx]
    y_power_val = y_power[val_idx]

    results: List[Dict[str, Any]] = []

    for cls in resolved:
        name = cls.__name__
        entry: Dict[str, Any] = {"estimator": name}
        try:
            est = cls()
            est.fit(measurements, model_dims, val_fraction)

            # Time metrics on validation set
            val_ms = [measurements[i] for i in val_idx]
            y_pred_time_val = np.array(
                [est.estimate(m.category, m.batch_size, m.seq_len, model_dims=model_dims).time_s
                 for m in val_ms]
            )
            time_metrics = _compute_metrics(y_time_val, y_pred_time_val)
            for k, v in time_metrics.items():
                entry[f"time_{k}"] = v

            # Energy metrics (if available)
            energy_mask_val = ~np.isnan(y_energy_val)
            if energy_mask_val.sum() >= 1 and est._has_energy:
                val_ms_energy = [measurements[i] for i in val_idx if not np.isnan(y_energy[i])]
                y_pred_energy_val = np.array(
                    [est.estimate(m.category, m.batch_size, m.seq_len, model_dims=model_dims).energy_j or 0.0
                     for m in val_ms_energy]
                )
                y_energy_val_clean = y_energy_val[energy_mask_val]
                if len(y_pred_energy_val) == len(y_energy_val_clean):
                    energy_metrics = _compute_metrics(y_energy_val_clean, y_pred_energy_val)
                    for k, v in energy_metrics.items():
                        entry[f"energy_{k}"] = v

            # Power metrics (if available)
            power_mask_val = ~np.isnan(y_power_val)
            if power_mask_val.sum() >= 1 and est._has_power:
                val_ms_power = [measurements[i] for i in val_idx if not np.isnan(y_power[i])]
                y_pred_power_val = np.array(
                    [est.estimate(m.category, m.batch_size, m.seq_len, model_dims=model_dims).power_w or 0.0
                     for m in val_ms_power]
                )
                y_power_val_clean = y_power_val[power_mask_val]
                if len(y_pred_power_val) == len(y_power_val_clean):
                    power_metrics = _compute_metrics(y_power_val_clean, y_pred_power_val)
                    for k, v in power_metrics.items():
                        entry[f"power_{k}"] = v

        except Exception as e:
            entry["error"] = str(e)

        results.append(entry)

    return results


def cross_model_evaluation(
    measurements_by_model: Dict[str, Sequence[OperatorMeasurement]],
    estimator_classes: Sequence[Union[Type[SklearnEstimatorBase], str]],
    model_dims_by_model: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Leave-one-model-out evaluation across multiple estimator types.

    Args:
        measurements_by_model: {model_name: measurements}.
        estimator_classes: Estimator classes or string names to evaluate.
        model_dims_by_model: Optional per-model dimension dicts.

    Returns:
        {estimator_class_name: {holdout_model: scores_dict}}
    """
    resolved = _resolve_classes(estimator_classes)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for cls in resolved:
        name = cls.__name__
        model_results: Dict[str, Dict[str, float]] = {}
        for holdout in measurements_by_model.keys():
            try:
                est = cls()
                scores = est.fit_cross_model(
                    measurements_by_model,
                    holdout,
                    model_dims_by_model,
                )
                model_results[holdout] = scores
            except Exception as e:
                model_results[holdout] = {"error": str(e)}  # type: ignore[dict-item]
        results[name] = model_results

    return results


def pick_best_estimator(
    comparison_results: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    target_metric: str = "time_r2",
) -> str:
    """Select the estimator class with the best score on a given metric.

    Args:
        comparison_results: Output of ``compare_estimators()`` (list or dict form).
        target_metric: Metric key to optimize (default: "time_r2").

    Returns:
        Class name of the best estimator.
    """
    best_name = ""
    best_score = float("-inf")

    # Support both list and dict formats
    if isinstance(comparison_results, list):
        items = [(entry.get("estimator", ""), entry) for entry in comparison_results]
    else:
        items = list(comparison_results.items())

    for name, metrics in items:
        if "error" in metrics:
            continue
        score = metrics.get(target_metric)
        if score is not None and score > best_score:
            best_score = score
            best_name = name

    if not best_name:
        raise ValueError(
            f"No estimator had metric '{target_metric}' without errors"
        )

    return best_name


def compare_estimators_by_category(
    measurements: Sequence[OperatorMeasurement],
    model_dims: Optional[Dict[str, float]],
    estimator_classes: Sequence[Union[Type[SklearnEstimatorBase], str]],
    val_fraction: float = 0.2,
    include_per_operator: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compare estimators per operator category.

    Groups measurements by OperatorCategory, trains each estimator on each
    category's data, and reports per-category metrics.

    Args:
        measurements: All profiling measurements.
        model_dims: Optional model dimension features.
        estimator_classes: SklearnEstimatorBase subclasses or string names.
        val_fraction: Validation split fraction.
        include_per_operator: Whether to include PerOperatorEstimator.

    Returns:
        ``{category_name: {estimator_name: {"time_r2": ..., "time_mae": ..., "time_rmse": ...}}}``
    """
    from sklearn.model_selection import train_test_split

    resolved = _resolve_classes(estimator_classes)

    # Group measurements by category
    by_category: Dict[OperatorCategory, List[OperatorMeasurement]] = {}
    for m in measurements:
        by_category.setdefault(m.category, []).append(m)

    # Optionally fit PerOperatorEstimator on ALL measurements once
    per_op_est = None
    if include_per_operator:
        try:
            from inference_simulator.estimator.per_operator_estimator import (
                PerOperatorEstimator,
            )

            per_op_est = PerOperatorEstimator()
            per_op_est.fit(list(measurements), model_dims, val_fraction)
        except (ImportError, Exception):
            per_op_est = None

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for category, cat_measurements in by_category.items():
        if len(cat_measurements) < 5:
            continue

        cat_name = category.value
        results[cat_name] = {}

        # Split category measurements for validation
        indices = list(range(len(cat_measurements)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, random_state=42
        )
        train_ms = [cat_measurements[i] for i in train_idx]
        val_ms = [cat_measurements[i] for i in val_idx]
        y_time_val = np.array([cat_measurements[i].time_s for i in val_idx])

        # Evaluate each SklearnEstimatorBase on this category's data
        for cls in resolved:
            name = cls.__name__
            try:
                est = cls()
                est.fit(train_ms, model_dims, val_fraction=0.0)

                y_pred = np.array(
                    [
                        est.estimate(
                            m.category, m.batch_size, m.seq_len, model_dims=model_dims
                        ).time_s
                        for m in val_ms
                    ]
                )
                metrics = _compute_metrics(y_time_val, y_pred)
                results[cat_name][name] = {
                    "time_r2": metrics["r2"],
                    "time_mae": metrics["mae"],
                    "time_rmse": metrics["rmse"],
                }
            except Exception as e:
                results[cat_name][name] = {"error": str(e)}  # type: ignore[dict-item]

        # Evaluate PerOperatorEstimator (trained on all data, evaluated per category)
        if per_op_est is not None:
            try:
                y_pred = np.array(
                    [
                        per_op_est.estimate(
                            m.category, m.batch_size, m.seq_len
                        ).time_s
                        for m in val_ms
                    ]
                )
                metrics = _compute_metrics(y_time_val, y_pred)
                results[cat_name]["PerOperatorEstimator"] = {
                    "time_r2": metrics["r2"],
                    "time_mae": metrics["mae"],
                    "time_rmse": metrics["rmse"],
                }
            except Exception as e:
                results[cat_name]["PerOperatorEstimator"] = {"error": str(e)}  # type: ignore[dict-item]

    return results
