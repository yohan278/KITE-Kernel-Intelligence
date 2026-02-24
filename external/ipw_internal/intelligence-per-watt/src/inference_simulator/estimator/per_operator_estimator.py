"""Per-operator-type estimator inspired by Vidur (arxiv:2405.05465).

Trains separate sklearn models per operator category with tailored feature
engineering, following Vidur's approach of per-operator ML models.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


# Category-specific feature extractors
def _linear_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """LINEAR/MLP: compute-bound, features = [num_tokens, log(num_tokens)]."""
    num_tokens = batch_size * seq_len
    return [float(num_tokens), math.log2(max(num_tokens, 1))]


def _attention_prefill_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """ATTENTION_PREFILL: O(n^2), features = [kv_cache_size, seq_len^2, batch_size, log(seq_len)]."""
    kv_cache_size = kwargs.get("kv_cache_size", seq_len)
    return [
        float(kv_cache_size),
        float(seq_len) ** 2,  # captures quadratic attention cost
        float(batch_size),
        math.log2(max(seq_len, 1)),
    ]


def _attention_decode_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """ATTENTION_DECODE: memory-bound, features = [batch_size, kv_cache_size, log(kv_cache)]."""
    kv_cache_size = kwargs.get("kv_cache_size", seq_len)
    return [
        float(batch_size),
        float(kv_cache_size),
        math.log2(max(kv_cache_size, 1)),
    ]


def _simple_token_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """NORMALIZATION/ACTIVATION/EMBEDDING: features = [num_tokens, log(num_tokens)]."""
    num_tokens = batch_size * seq_len
    return [float(num_tokens), math.log2(max(num_tokens, 1))]


def _communication_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """COMMUNICATION: features = [num_tokens, num_workers]."""
    num_tokens = batch_size * seq_len
    num_workers = kwargs.get("num_workers", 1)
    return [float(num_tokens), float(num_workers)]


def _cpu_host_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """CPU_HOST: features = [batch_size, log(batch_size)]."""
    return [float(batch_size), math.log2(max(batch_size, 1))]


def _sampling_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """SAMPLING: features = [batch_size, vocab_size_log]."""
    vocab_size = kwargs.get("vocab_size", 151936)
    return [float(batch_size), math.log2(max(vocab_size, 1))]


def _fused_prefill_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """FUSED_PREFILL: full prefill pass, features = [batch_size, seq_len, seq_len^2, log(seq_len)]."""
    return [
        float(batch_size),
        float(seq_len),
        float(seq_len) ** 2,
        math.log2(max(seq_len, 1)),
    ]


def _fused_decode_step_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """FUSED_DECODE_STEP: full decode step, features = [batch_size, kv_cache_size, log(kv_cache)]."""
    kv_cache_size = kwargs.get("kv_cache_size", seq_len)
    return [
        float(batch_size),
        float(kv_cache_size),
        math.log2(max(kv_cache_size, 1)),
    ]


def _fused_token_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """FUSED_MLP / FUSED_NORM_ATTN: features = [num_tokens, log(num_tokens)]."""
    num_tokens = batch_size * seq_len
    return [float(num_tokens), math.log2(max(num_tokens, 1))]


def _default_features(batch_size: int, seq_len: int, **kwargs: Any) -> List[float]:
    """Default: features = [batch_size, seq_len, num_tokens]."""
    return [float(batch_size), float(seq_len), float(batch_size * seq_len)]


# Map categories to their feature extractor
_FEATURE_EXTRACTORS = {
    OperatorCategory.LINEAR: _linear_features,
    OperatorCategory.ATTENTION_PREFILL: _attention_prefill_features,
    OperatorCategory.ATTENTION_DECODE: _attention_decode_features,
    OperatorCategory.NORMALIZATION: _simple_token_features,
    OperatorCategory.ACTIVATION: _simple_token_features,
    OperatorCategory.EMBEDDING: _simple_token_features,
    OperatorCategory.MOE_ROUTING: _simple_token_features,
    OperatorCategory.MOE_EXPERT: _linear_features,
    OperatorCategory.SSM_SCAN: _simple_token_features,
    OperatorCategory.COMMUNICATION: _communication_features,
    OperatorCategory.CPU_HOST: _cpu_host_features,
    OperatorCategory.SAMPLING: _sampling_features,
    OperatorCategory.MTP: _simple_token_features,
    OperatorCategory.KV_CACHE: _simple_token_features,
    OperatorCategory.AGENTIC_TOOL: _default_features,
    OperatorCategory.FUSED_PREFILL: _fused_prefill_features,
    OperatorCategory.FUSED_DECODE_STEP: _fused_decode_step_features,
    OperatorCategory.FUSED_ATTENTION: _attention_prefill_features,
    OperatorCategory.FUSED_MLP: _fused_token_features,
    OperatorCategory.FUSED_NORM_ATTN: _fused_token_features,
}


class PerOperatorEstimator(BaseRuntimeEstimator):
    """Trains separate sklearn models per operator category with tailored features.

    Inspired by Vidur's approach where each operator type gets its own ML model
    with category-specific feature engineering, rather than a single unified model
    with one-hot category encoding.
    """

    def __init__(self, random_state: int = 42, normalize_loss: bool = False) -> None:
        self._random_state = random_state
        self._normalize_loss = normalize_loss
        self._time_models: Dict[OperatorCategory, Any] = {}
        self._energy_models: Dict[OperatorCategory, Any] = {}
        self._power_models: Dict[OperatorCategory, Any] = {}
        self._fitted = False

    def _create_model(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=100, random_state=self._random_state, n_jobs=-1
        )

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        measurements: Sequence[OperatorMeasurement],
        model_dims: Optional[Dict[str, float]] = None,
        val_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """Train separate models per operator category."""
        # Group measurements by category
        by_category: Dict[OperatorCategory, List[OperatorMeasurement]] = {}
        for m in measurements:
            by_category.setdefault(m.category, []).append(m)

        scores: Dict[str, float] = {}

        for category, cat_measurements in by_category.items():
            if len(cat_measurements) < 2:
                continue

            feature_fn = _FEATURE_EXTRACTORS.get(category, _default_features)

            # Build feature matrix
            X_list = []
            y_time_list = []
            y_energy_list = []

            for m in cat_measurements:
                features = feature_fn(m.batch_size, m.seq_len)
                X_list.append(features)
                y_time_list.append(m.time_s)
                y_energy_list.append(
                    m.energy_j if m.energy_j is not None else float("nan")
                )

            X = np.array(X_list)
            y_time = np.array(y_time_list)
            y_energy = np.array(y_energy_list)

            # Train time model
            time_model = self._create_model()
            if self._normalize_loss and len(y_time) > 0:
                # 1/y weighting: sklearn RF loss squares residuals, so w*(pred-y)^2
                # with w=1/y gives (pred-y)^2/y, a relative error objective
                sample_weight = 1.0 / np.maximum(y_time, 1e-12)
                sample_weight *= len(y_time) / sample_weight.sum()
                time_model.fit(X, y_time, sample_weight=sample_weight)
            else:
                time_model.fit(X, y_time)
            self._time_models[category] = time_model
            scores[f"{category.value}_time_train_r2"] = time_model.score(X, y_time)

            # Train energy model if data available
            energy_mask = ~np.isnan(y_energy)
            if energy_mask.sum() >= 2:
                energy_model = self._create_model()
                X_en = X[energy_mask]
                y_en = y_energy[energy_mask]
                if self._normalize_loss and len(y_en) > 0:
                    sw = 1.0 / np.maximum(y_en, 1e-12)
                    sw *= len(y_en) / sw.sum()
                    energy_model.fit(X_en, y_en, sample_weight=sw)
                else:
                    energy_model.fit(X_en, y_en)
                self._energy_models[category] = energy_model

        self._fitted = len(self._time_models) > 0
        return scores

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        if not self._fitted:
            raise RuntimeError("PerOperatorEstimator is not fitted")

        feature_fn = _FEATURE_EXTRACTORS.get(operator_category, _default_features)
        features = feature_fn(batch_size, seq_len, **kwargs)
        X = np.array([features])

        _EPS = 1e-12

        # Get time prediction
        if operator_category in self._time_models:
            time_s = max(
                float(self._time_models[operator_category].predict(X)[0]), _EPS
            )
        else:
            # Fallback: use first available model with its own features
            if self._time_models:
                first_cat = next(iter(self._time_models))
                first_fn = _FEATURE_EXTRACTORS.get(first_cat, _default_features)
                fb_features = first_fn(batch_size, seq_len, **kwargs)
                time_s = max(
                    float(
                        self._time_models[first_cat].predict(
                            np.array([fb_features])
                        )[0]
                    ),
                    _EPS,
                )
            else:
                time_s = _EPS

        # Get energy prediction
        energy_j = None
        if operator_category in self._energy_models:
            energy_j = max(
                float(self._energy_models[operator_category].predict(X)[0]), _EPS
            )

        return EstimatorResult(time_s=time_s, energy_j=energy_j)
