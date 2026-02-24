"""Lookup-table runtime estimator with interpolation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


class LookupTableEstimator(BaseRuntimeEstimator):
    """Runtime estimator using measured data with nearest-neighbor interpolation.

    Stores profiling measurements in a lookup table keyed by
    (operator_category, batch_size, seq_len). For exact matches, returns
    the measured value. For misses, interpolates between the nearest
    measured points.
    """

    def __init__(self) -> None:
        # Key: (category, batch_size, seq_len) → EstimatorResult
        self._table: Dict[Tuple[str, int, int], EstimatorResult] = {}
        # Sorted keys per category for interpolation
        self._category_keys: Dict[str, List[Tuple[int, int]]] = {}
        self._fitted = False

    def is_fitted(self) -> bool:
        return self._fitted

    def load_from_measurements(
        self, measurements: Sequence[OperatorMeasurement]
    ) -> None:
        """Load measurements directly into the lookup table."""
        for m in measurements:
            key = (m.category.value, m.batch_size, m.seq_len)
            self._table[key] = EstimatorResult(
                time_s=m.time_s,
                energy_j=m.energy_j,
                power_w=m.power_w,
            )

        self._rebuild_category_keys()
        self._fitted = len(self._table) > 0

    def load_from_csv(self, csv_path: Path, category: OperatorCategory) -> None:
        """Load measurements from a profiling CSV file.

        Args:
            csv_path: Path to CSV with columns: operator_name, batch_size,
                seq_len, time_s, energy_j, power_w, ...
            category: The OperatorCategory for all rows in this CSV.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch_size = int(row["batch_size"])
                seq_len = int(row["seq_len"])
                time_s = float(row["time_s"])
                energy_j = _parse_optional_float(row.get("energy_j"))
                power_w = _parse_optional_float(row.get("power_w"))

                key = (category.value, batch_size, seq_len)
                self._table[key] = EstimatorResult(
                    time_s=time_s, energy_j=energy_j, power_w=power_w
                )

        self._rebuild_category_keys()
        self._fitted = len(self._table) > 0

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Look up or interpolate runtime for given dimensions."""
        if not self._fitted:
            raise RuntimeError("LookupTableEstimator has no data loaded")

        cat_key = operator_category.value
        exact_key = (cat_key, batch_size, seq_len)

        # Exact match
        if exact_key in self._table:
            return self._table[exact_key]

        # Interpolation: find nearest neighbors
        return self._interpolate(cat_key, batch_size, seq_len)

    def _interpolate(
        self, cat_key: str, batch_size: int, seq_len: int
    ) -> EstimatorResult:
        """Bilinear interpolation between nearest measured points."""
        keys = self._category_keys.get(cat_key, [])
        if not keys:
            return EstimatorResult(time_s=0.0)

        # Find nearest point by Manhattan distance in log space
        import math

        def log_dist(b: int, s: int) -> float:
            db = abs(math.log2(max(b, 1)) - math.log2(max(batch_size, 1)))
            ds = abs(math.log2(max(s, 1)) - math.log2(max(seq_len, 1)))
            return db + ds

        sorted_keys = sorted(keys, key=lambda k: log_dist(k[0], k[1]))

        if len(sorted_keys) == 1:
            b, s = sorted_keys[0]
            result = self._table[(cat_key, b, s)]
            # Scale linearly by token count ratio
            scale = (batch_size * seq_len) / max(b * s, 1)
            return EstimatorResult(
                time_s=result.time_s * scale,
                energy_j=result.energy_j * scale if result.energy_j is not None else None,
                power_w=result.power_w,
            )

        # Weighted average of 2 nearest neighbors (inverse distance)
        results = []
        weights = []
        for b, s in sorted_keys[:2]:
            d = log_dist(b, s)
            w = 1.0 / max(d, 0.01)
            results.append(self._table[(cat_key, b, s)])
            weights.append(w)

        total_w = sum(weights)
        time_s = sum(r.time_s * w for r, w in zip(results, weights)) / total_w

        energy_vals = [r.energy_j for r in results if r.energy_j is not None]
        energy_j = (
            sum(e * w for e, w in zip(energy_vals, weights[: len(energy_vals)])) / sum(weights[: len(energy_vals)])
            if energy_vals
            else None
        )

        power_vals = [r.power_w for r in results if r.power_w is not None]
        power_w = (
            sum(p * w for p, w in zip(power_vals, weights[: len(power_vals)])) / sum(weights[: len(power_vals)])
            if power_vals
            else None
        )

        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=power_w)

    def _rebuild_category_keys(self) -> None:
        """Rebuild sorted category key index."""
        self._category_keys.clear()
        for cat_key, batch_size, seq_len in self._table:
            if cat_key not in self._category_keys:
                self._category_keys[cat_key] = []
            self._category_keys[cat_key].append((batch_size, seq_len))


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    """Parse a CSV field to Optional[float]."""
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
