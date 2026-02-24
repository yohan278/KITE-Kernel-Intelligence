"""Profiling result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.operators import OperatorMeasurement


@dataclass
class ProfilingResult:
    """Aggregate result from a complete profiling run.

    Contains all operator measurements across token ops, attention,
    and agentic profilers for a given model × hardware × precision.
    """

    model_spec: ModelSpec
    hardware_spec: HardwareSpec
    precision: str
    timestamp: str
    measurements: List[OperatorMeasurement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_measurements(self) -> int:
        return len(self.measurements)

    def filter_by_category(self, category) -> List[OperatorMeasurement]:
        """Return measurements matching a specific OperatorCategory."""
        return [m for m in self.measurements if m.category == category]
