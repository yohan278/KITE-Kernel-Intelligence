"""Extended hardware specification for inference simulation.

Extends the base HardwareSpecs from ipw.simulator with additional fields
needed for cross-package profiling and cost analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ipw.simulator.hardware_specs import HardwareSpecs, get_hardware_specs


@dataclass(frozen=True)
class HardwareSpec:
    """Extended hardware specification with interconnect, CPU, and pricing info.

    Wraps the core accelerator specs from ipw.simulator.HardwareSpecs and
    adds fields needed by the dataset generator and inference search packages.
    """

    name: str
    vendor: str
    memory_gb: float
    tdp_watts: float
    peak_fp16_tflops: float
    peak_fp8_tflops: float = 0.0
    peak_bf16_tflops: float = 0.0
    hbm_bandwidth_gb_s: float = 0.0
    nvlink_bandwidth_gb_s: float = 0.0
    bytes_per_param_fp16: float = 2.0
    bytes_per_param_fp8: float = 1.0
    interconnect_type: str = ""
    interconnect_bandwidth_gb_s: float = 0.0
    cpu_model: str = ""
    price_per_hour_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def peak_tflops(self) -> float:
        """Best available tensor-core TFLOPS (prefers FP8 if available)."""
        if self.peak_fp8_tflops > 0:
            return self.peak_fp8_tflops
        return self.peak_fp16_tflops

    @property
    def bytes_per_param(self) -> float:
        """Bytes per parameter for the best quantization available."""
        if self.peak_fp8_tflops > 0:
            return self.bytes_per_param_fp8
        return self.bytes_per_param_fp16

    @classmethod
    def from_ipw_specs(cls, hw: HardwareSpecs, **extra) -> HardwareSpec:
        """Create a HardwareSpec from an existing ipw HardwareSpecs instance.

        Args:
            hw: Base hardware specs from ipw.simulator.
            **extra: Additional fields (interconnect_type, cpu_model, price_per_hour_usd).

        Returns:
            Extended HardwareSpec instance.
        """
        return cls(
            name=hw.name,
            vendor=hw.vendor,
            memory_gb=hw.memory_gb,
            tdp_watts=hw.tdp_watts,
            peak_fp16_tflops=hw.peak_fp16_tflops,
            peak_fp8_tflops=hw.peak_fp8_tflops,
            peak_bf16_tflops=hw.peak_bf16_tflops,
            hbm_bandwidth_gb_s=hw.hbm_bandwidth_gb_s,
            nvlink_bandwidth_gb_s=hw.nvlink_bandwidth_gb_s,
            bytes_per_param_fp16=hw.bytes_per_param_fp16,
            bytes_per_param_fp8=hw.bytes_per_param_fp8,
            **extra,
        )

    def peak_tflops_for_precision(self, precision: str) -> float:
        """Return peak TFLOPS for the given precision string.

        Args:
            precision: One of "fp16", "bf16", "fp8".

        Returns:
            Peak TFLOPS for the precision, falling back to fp16.
        """
        if precision == "fp8" and self.peak_fp8_tflops > 0:
            return self.peak_fp8_tflops
        if precision == "bf16" and self.peak_bf16_tflops > 0:
            return self.peak_bf16_tflops
        return self.peak_fp16_tflops

    def bytes_per_param_for_precision(self, precision: str) -> float:
        """Return bytes per parameter for the given precision.

        Args:
            precision: One of "fp16", "bf16", "fp8".

        Returns:
            Bytes per parameter (2.0 for fp16/bf16, 1.0 for fp8).
        """
        if precision == "fp8":
            return self.bytes_per_param_fp8
        return self.bytes_per_param_fp16

    @classmethod
    def from_registry(cls, gpu_type: str, **extra) -> HardwareSpec:
        """Look up hardware specs from the IPW registry and wrap as HardwareSpec.

        Args:
            gpu_type: GPU type key (e.g., "a100_80gb").
            **extra: Additional fields.

        Returns:
            Extended HardwareSpec instance.
        """
        hw = get_hardware_specs(gpu_type)
        return cls.from_ipw_specs(hw, **extra)
