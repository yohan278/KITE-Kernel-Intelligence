"""Per-operator power prediction model for inference simulation."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement

logger = logging.getLogger(__name__)


@dataclass
class OperatorEvent:
    """A single operator execution event during simulation."""

    category: OperatorCategory
    duration_ns: int
    batch_size: int
    seq_len: int
    start_time_ns: int = 0
    layer_idx: Optional[int] = None
    flops: Optional[int] = None
    bytes_accessed: Optional[int] = None


@dataclass(frozen=True)
class EnergyBreakdown:
    """Per-category and per-layer energy decomposition (IrEne-inspired).

    Provides fine-grained visibility into which operators and layers
    dominate energy consumption.

    Attributes:
        total_energy_j: Total energy in joules (matches compute_energy).
        energy_by_category: {category_name: joules} aggregated across all layers.
        energy_by_layer_category: {layer_idx: {category_name: joules}}.
            Events with layer_idx=None are stored under key -1.
        energy_fraction_by_category: {category_name: fraction} (sums to ~1.0).
        idle_energy_j: Energy consumed during idle gaps between events.
    """

    total_energy_j: float = 0.0
    energy_by_category: Dict[str, float] = field(default_factory=dict)
    energy_by_layer_category: Dict[int, Dict[str, float]] = field(default_factory=dict)
    energy_fraction_by_category: Dict[str, float] = field(default_factory=dict)
    idle_energy_j: float = 0.0


class PerOperatorPowerModel:
    """Predicts per-operator power draw from profiling measurements.

    Trains per-category random forest regressors from profiled energy data,
    then predicts power draw for operator events during simulation.
    """

    def __init__(self) -> None:
        self._power_models: Dict[OperatorCategory, Any] = {}
        self._idle_power_w: float = 0.0
        self._fitted = False
        # Fallback: per-category mean power for categories with too few samples
        self._mean_power: Dict[OperatorCategory, float] = {}
        # Cache: (category, batch_size, seq_len) -> predicted watts
        self._predict_cache: Dict[Tuple[OperatorCategory, int, int], float] = {}

    def fit(self, measurements: List[OperatorMeasurement]) -> None:
        """Train per-category power models from profiled energy data.

        Groups measurements with non-null power_w by category, then trains
        a random forest regressor mapping (batch_size, seq_len) -> watts
        for each category. Categories with fewer than 5 samples use the
        mean power as a constant predictor.

        Args:
            measurements: Profiling measurements with power_w data.
        """
        # Group measurements with non-null power by category
        by_category: Dict[OperatorCategory, List[OperatorMeasurement]] = defaultdict(list)
        all_power_values: List[float] = []

        for m in measurements:
            if m.power_w is not None and m.power_w > 0:
                by_category[m.category].append(m)
                all_power_values.append(m.power_w)

        if not all_power_values:
            logger.warning("No measurements with power_w data; model will use defaults")
            self._idle_power_w = 0.0
            self._fitted = True
            return

        # Estimate idle power as the 5th percentile of all power readings
        self._idle_power_w = float(np.percentile(all_power_values, 5))

        for category, cat_measurements in by_category.items():
            power_values = [m.power_w for m in cat_measurements]
            self._mean_power[category] = float(np.mean(power_values))

            if len(cat_measurements) < 5:
                # Too few samples for regression; use mean
                continue

            try:
                from sklearn.ensemble import RandomForestRegressor

                X = np.array([[m.batch_size, m.seq_len] for m in cat_measurements])
                y = np.array([m.power_w for m in cat_measurements])

                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=42, n_jobs=1
                )
                rf.fit(X, y)
                self._power_models[category] = rf
            except ImportError:
                logger.info(
                    "sklearn not available; using mean power for category %s",
                    category.value,
                )

        self._fitted = True
        logger.info(
            "Power model fitted: %d categories with RF, %d with mean fallback, "
            "idle_power=%.1fW",
            len(self._power_models),
            len(self._mean_power) - len(self._power_models),
            self._idle_power_w,
        )

    def predict_power(
        self,
        category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        flops: Optional[int] = None,
        bytes_accessed: Optional[int] = None,
    ) -> float:
        """Predict power draw in watts for an operator invocation.

        When flops and bytes_accessed are provided, includes arithmetic
        intensity as a feature (Power Roofline insight: compute-bound ops
        draw more power than memory-bound ops).

        Args:
            category: The operator category.
            batch_size: Batch size for the operation.
            seq_len: Sequence length for the operation.
            flops: Optional FLOPs count for arithmetic intensity.
            bytes_accessed: Optional bytes accessed for arithmetic intensity.

        Returns:
            Predicted power draw in watts.
        """
        cache_key = (category, batch_size, seq_len)
        cached = self._predict_cache.get(cache_key)
        if cached is not None:
            return cached

        rf = self._power_models.get(category)
        if rf is not None:
            X = np.array([[batch_size, seq_len]])
            result = float(max(0.0, rf.predict(X)[0]))
        elif category in self._mean_power:
            result = self._mean_power[category]
        else:
            result = self._idle_power_w

        self._predict_cache[cache_key] = result
        return result

    def compute_energy(self, operator_events: List[OperatorEvent]) -> float:
        """Compute total energy from operator events.

        Sums power_w * duration_s for each event, plus idle power during
        gaps between events.

        Args:
            operator_events: List of operator events sorted by start_time_ns.

        Returns:
            Total energy in joules.
        """
        if not operator_events:
            return 0.0

        total_energy_j = 0.0

        # Sort events by start time
        sorted_events = sorted(operator_events, key=lambda e: e.start_time_ns)

        prev_end_ns = sorted_events[0].start_time_ns

        for event in sorted_events:
            # Account for idle gap before this event
            gap_ns = max(0, event.start_time_ns - prev_end_ns)
            if gap_ns > 0:
                idle_duration_s = gap_ns / 1e9
                total_energy_j += self._idle_power_w * idle_duration_s

            # Energy for this operator event
            duration_s = event.duration_ns / 1e9
            power_w = self.predict_power(
                event.category, event.batch_size, event.seq_len,
                flops=event.flops, bytes_accessed=event.bytes_accessed,
            )
            total_energy_j += power_w * duration_s

            prev_end_ns = event.start_time_ns + event.duration_ns

        return total_energy_j

    def compute_energy_breakdown(
        self, operator_events: List[OperatorEvent]
    ) -> EnergyBreakdown:
        """Compute per-category and per-layer energy breakdown.

        Same integration logic as compute_energy(), but accumulates
        energy into per-category and per-layer-category dicts for
        fine-grained analysis (IrEne-inspired tree decomposition).

        Events with layer_idx=None are assigned to layer key -1.

        Args:
            operator_events: List of operator events.

        Returns:
            EnergyBreakdown with per-category and per-layer decomposition.
        """
        if not operator_events:
            return EnergyBreakdown()

        energy_by_cat: Dict[str, float] = defaultdict(float)
        energy_by_layer_cat: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        total_energy_j = 0.0
        idle_energy_j = 0.0

        sorted_events = sorted(operator_events, key=lambda e: e.start_time_ns)
        prev_end_ns = sorted_events[0].start_time_ns

        for event in sorted_events:
            # Idle gap energy
            gap_ns = max(0, event.start_time_ns - prev_end_ns)
            if gap_ns > 0:
                idle_j = self._idle_power_w * (gap_ns / 1e9)
                idle_energy_j += idle_j
                total_energy_j += idle_j

            # Event energy
            duration_s = event.duration_ns / 1e9
            power_w = self.predict_power(
                event.category, event.batch_size, event.seq_len,
                flops=event.flops, bytes_accessed=event.bytes_accessed,
            )
            event_energy = power_w * duration_s
            total_energy_j += event_energy

            cat_name = event.category.value
            energy_by_cat[cat_name] += event_energy

            layer_key = event.layer_idx if event.layer_idx is not None else -1
            energy_by_layer_cat[layer_key][cat_name] += event_energy

            prev_end_ns = event.start_time_ns + event.duration_ns

        # Compute fractions
        fraction_by_cat: Dict[str, float] = {}
        if total_energy_j > 0:
            for cat_name, energy in energy_by_cat.items():
                fraction_by_cat[cat_name] = energy / total_energy_j

        # Convert defaultdicts to regular dicts for frozen dataclass
        layer_cat_dict = {
            k: dict(v) for k, v in energy_by_layer_cat.items()
        }

        return EnergyBreakdown(
            total_energy_j=total_energy_j,
            energy_by_category=dict(energy_by_cat),
            energy_by_layer_category=layer_cat_dict,
            energy_fraction_by_category=fraction_by_cat,
            idle_energy_j=idle_energy_j,
        )

    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted
