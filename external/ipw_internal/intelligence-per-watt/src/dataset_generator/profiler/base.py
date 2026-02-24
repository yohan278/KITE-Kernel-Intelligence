"""Abstract base class for operator profilers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.sweep import SweepConfig


class BaseOperatorProfiler(ABC):
    """Abstract base for all operator profilers."""

    @property
    @abstractmethod
    def category(self) -> OperatorCategory:
        """The operator category this profiler measures."""
        raise NotImplementedError

    @abstractmethod
    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Run profiling across sweep dimensions.

        Args:
            model_spec: Model architecture specification.
            hw_spec: Hardware specification.
            sweep_config: Sweep dimension configuration.
            precision: Numeric precision ("fp16", "bf16", "fp8").

        Returns:
            List of measurements across all sweep points.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sweep_dimensions(self) -> List[str]:
        """Return the SweepConfig field names this profiler sweeps over."""
        raise NotImplementedError
