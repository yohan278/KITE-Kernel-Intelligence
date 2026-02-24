"""Pre-computed prediction cache for O(1) lookup during simulation.

Follows Vidur's pattern of pre-computing all predictions at init time,
storing them in a dict keyed by input tuple for instant lookup.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory


class PredictionCache(BaseRuntimeEstimator):
    """O(1) prediction lookup via pre-computed dict.

    After training an estimator, pre-computes predictions for the full
    operating range and caches them for instant lookup during simulation.
    """

    def __init__(
        self,
        estimator: BaseRuntimeEstimator,
        categories: Sequence[OperatorCategory],
        batch_sizes: Sequence[int],
        seq_lens: Sequence[int],
    ) -> None:
        self._estimator = estimator
        self._cache: Dict[Tuple[str, int, int], EstimatorResult] = {}
        self._fitted = False
        self._precompute(categories, batch_sizes, seq_lens)

    def _precompute(
        self,
        categories: Sequence[OperatorCategory],
        batch_sizes: Sequence[int],
        seq_lens: Sequence[int],
    ) -> None:
        """Pre-compute predictions for the full grid."""
        if not self._estimator.is_fitted():
            return

        for cat in categories:
            for bs in batch_sizes:
                for sl in seq_lens:
                    key = (cat.value, bs, sl)
                    try:
                        result = self._estimator.estimate(cat, bs, sl)
                        self._cache[key] = result
                    except Exception:
                        pass

        self._fitted = len(self._cache) > 0

    def is_fitted(self) -> bool:
        return self._fitted

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """O(1) lookup in pre-computed cache.

        Falls back to the underlying estimator for cache misses.
        """
        key = (operator_category.value, batch_size, seq_len)
        if key in self._cache:
            return self._cache[key]

        # Cache miss: snap to nearest cached point
        nearest = self._find_nearest(operator_category, batch_size, seq_len)
        if nearest is not None:
            return nearest

        # Final fallback: call estimator directly
        return self._estimator.estimate(
            operator_category, batch_size, seq_len, **kwargs
        )

    def _find_nearest(
        self, category: OperatorCategory, batch_size: int, seq_len: int
    ) -> Optional[EstimatorResult]:
        """Find the nearest cached point by Euclidean distance in log space."""
        best_dist = float("inf")
        best_result = None
        log_bs = math.log2(max(batch_size, 1))
        log_sl = math.log2(max(seq_len, 1))

        for (cat_val, cached_bs, cached_sl), result in self._cache.items():
            if cat_val != category.value:
                continue
            dist = (math.log2(max(cached_bs, 1)) - log_bs) ** 2 + (
                math.log2(max(cached_sl, 1)) - log_sl
            ) ** 2
            if dist < best_dist:
                best_dist = dist
                best_result = result

        return best_result

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def categories(self) -> List[str]:
        return list(set(k[0] for k in self._cache))
