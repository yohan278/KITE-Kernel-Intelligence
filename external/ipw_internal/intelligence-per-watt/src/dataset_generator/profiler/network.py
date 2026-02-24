"""Network operator profiler (stub)."""

from __future__ import annotations

from typing import List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig


class NetworkProfiler(BaseOperatorProfiler):
    """Profiles network communication operations (stub)."""

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.COMMUNICATION

    def get_sweep_dimensions(self) -> List[str]:
        return ["message_sizes_bytes", "gpu_topologies"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
    ) -> List[OperatorMeasurement]:
        raise NotImplementedError("Network profiler not yet implemented")
