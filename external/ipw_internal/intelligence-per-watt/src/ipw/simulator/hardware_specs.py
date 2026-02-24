"""Extended hardware specifications for inference simulation.

Augments GPU_TYPE_REGISTRY from grid_eval/config.py with compute-relevant
specs (peak TFLOPS, memory bandwidth, NVLink bandwidth) sourced from
publicly available spec sheets.

These specs are used by the roofline-based inference model to compute
theoretical prefill/decode latency and energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(slots=True)
class HardwareSpecs:
    """Compute-relevant hardware specifications for a single accelerator.

    All values are per-accelerator (not aggregate across multi-GPU).

    Attributes:
        name: Human-readable name.
        vendor: Hardware vendor (nvidia, amd, apple).
        memory_gb: Total HBM/unified memory in GB.
        tdp_watts: Thermal design power in watts.
        peak_fp16_tflops: Peak FP16 tensor-core TFLOPS.
        peak_fp8_tflops: Peak FP8 tensor-core TFLOPS (0 if unsupported).
        peak_bf16_tflops: Peak BF16 tensor-core TFLOPS.
        hbm_bandwidth_gb_s: Peak HBM bandwidth in GB/s.
        nvlink_bandwidth_gb_s: Bidirectional NVLink bandwidth in GB/s (0 if N/A).
        bytes_per_param_fp16: Bytes per parameter in FP16 (2).
        bytes_per_param_fp8: Bytes per parameter in FP8 (1).
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


# Extended hardware specifications per GPU type.
# Keys match GpuType enum values from grid_eval/config.py.
# Sources: NVIDIA/AMD/Apple product data sheets, MLPerf submissions.
HARDWARE_SPECS_REGISTRY: Dict[str, HardwareSpecs] = {
    # =========================================================================
    # NVIDIA GPUs
    # =========================================================================
    "a100_80gb": HardwareSpecs(
        name="NVIDIA A100 80GB SXM",
        vendor="nvidia",
        memory_gb=80,
        tdp_watts=400,
        peak_fp16_tflops=312.0,       # Tensor core FP16
        peak_fp8_tflops=0.0,          # A100 does not support FP8
        peak_bf16_tflops=312.0,       # Tensor core BF16
        hbm_bandwidth_gb_s=2039.0,    # HBM2e
        nvlink_bandwidth_gb_s=600.0,  # NVLink 3.0: 12 links x 50 GB/s
    ),
    "h100_80gb": HardwareSpecs(
        name="NVIDIA H100 80GB SXM",
        vendor="nvidia",
        memory_gb=80,
        tdp_watts=700,
        peak_fp16_tflops=989.4,       # Tensor core FP16
        peak_fp8_tflops=1978.9,       # Tensor core FP8
        peak_bf16_tflops=989.4,       # Tensor core BF16
        hbm_bandwidth_gb_s=3352.0,    # HBM3
        nvlink_bandwidth_gb_s=900.0,  # NVLink 4.0: 18 links x 50 GB/s
    ),
    "h200": HardwareSpecs(
        name="NVIDIA H200 141GB",
        vendor="nvidia",
        memory_gb=141,
        tdp_watts=700,
        peak_fp16_tflops=989.4,       # Same compute as H100
        peak_fp8_tflops=1978.9,
        peak_bf16_tflops=989.4,
        hbm_bandwidth_gb_s=4800.0,    # HBM3e (major upgrade over H100)
        nvlink_bandwidth_gb_s=900.0,
    ),
    "gh200": HardwareSpecs(
        name="NVIDIA GH200 Grace Hopper",
        vendor="nvidia",
        memory_gb=96,  # GPU HBM3e (+ 480GB LPDDR5X on Grace CPU)
        tdp_watts=900,
        peak_fp16_tflops=989.4,
        peak_fp8_tflops=1978.9,
        peak_bf16_tflops=989.4,
        hbm_bandwidth_gb_s=4000.0,    # HBM3e
        nvlink_bandwidth_gb_s=900.0,  # NVLink-C2C between Grace and Hopper
    ),
    "b200": HardwareSpecs(
        name="NVIDIA B200",
        vendor="nvidia",
        memory_gb=192,
        tdp_watts=1000,
        peak_fp16_tflops=2250.0,      # Tensor core FP16
        peak_fp8_tflops=4500.0,       # Tensor core FP8
        peak_bf16_tflops=2250.0,      # Tensor core BF16
        hbm_bandwidth_gb_s=8000.0,    # HBM3e
        nvlink_bandwidth_gb_s=1800.0, # NVLink 5.0
    ),
    # =========================================================================
    # AMD GPUs
    # =========================================================================
    "mi300x": HardwareSpecs(
        name="AMD Instinct MI300X",
        vendor="amd",
        memory_gb=192,
        tdp_watts=750,
        peak_fp16_tflops=1307.4,      # Matrix core FP16
        peak_fp8_tflops=2614.9,       # Matrix core FP8
        peak_bf16_tflops=1307.4,      # Matrix core BF16
        hbm_bandwidth_gb_s=5300.0,    # HBM3 (8 stacks)
        nvlink_bandwidth_gb_s=0.0,    # Uses Infinity Fabric, not NVLink
    ),
    # =========================================================================
    # Apple Silicon
    # =========================================================================
    "m4_max": HardwareSpecs(
        name="Apple M4 Max",
        vendor="apple",
        memory_gb=128,
        tdp_watts=40,
        peak_fp16_tflops=54.6,        # 40 GPU cores @ FP16
        peak_fp8_tflops=0.0,          # No FP8 tensor cores
        peak_bf16_tflops=54.6,
        hbm_bandwidth_gb_s=546.0,     # Unified memory (LPDDR5X)
    ),
    "m4_pro": HardwareSpecs(
        name="Apple M4 Pro",
        vendor="apple",
        memory_gb=64,
        tdp_watts=30,
        peak_fp16_tflops=27.3,        # 20 GPU cores @ FP16
        peak_fp8_tflops=0.0,
        peak_bf16_tflops=27.3,
        hbm_bandwidth_gb_s=273.0,     # Unified memory (LPDDR5X)
    ),
    "m3_max": HardwareSpecs(
        name="Apple M3 Max",
        vendor="apple",
        memory_gb=128,
        tdp_watts=40,
        peak_fp16_tflops=44.8,        # 40 GPU cores @ FP16
        peak_fp8_tflops=0.0,
        peak_bf16_tflops=44.8,
        hbm_bandwidth_gb_s=400.0,     # Unified memory (LPDDR5)
    ),
    "m3_pro": HardwareSpecs(
        name="Apple M3 Pro",
        vendor="apple",
        memory_gb=36,
        tdp_watts=30,
        peak_fp16_tflops=20.2,        # 18 GPU cores @ FP16
        peak_fp8_tflops=0.0,
        peak_bf16_tflops=20.2,
        hbm_bandwidth_gb_s=150.0,     # Unified memory (LPDDR5)
    ),
}


def get_hardware_specs(gpu_type: str) -> HardwareSpecs:
    """Look up hardware specs by GPU type string.

    Args:
        gpu_type: GPU type identifier (must match GpuType enum value).

    Returns:
        HardwareSpecs for the given GPU type.

    Raises:
        KeyError: If gpu_type is not in the registry.
    """
    key = gpu_type.lower()
    if key not in HARDWARE_SPECS_REGISTRY:
        available = ", ".join(sorted(HARDWARE_SPECS_REGISTRY.keys()))
        raise KeyError(
            f"Unknown GPU type '{gpu_type}'. Available: {available}"
        )
    return HARDWARE_SPECS_REGISTRY[key]


def get_model_specs(model_type: str) -> Dict[str, Any]:
    """Look up model specs from MODEL_REGISTRY by model type string.

    Imports grid_eval config to avoid duplicating the registry.

    Args:
        model_type: Model type identifier (must match ModelType enum value).

    Returns:
        Model configuration dictionary from MODEL_REGISTRY.

    Raises:
        KeyError: If model_type is not in the registry.
    """
    # Import lazily to avoid circular deps and keep simulator self-contained
    from grid_eval.config import MODEL_REGISTRY, ModelType

    # Try direct enum lookup
    try:
        model_enum = ModelType(model_type)
    except ValueError:
        available = ", ".join(m.value for m in ModelType)
        raise KeyError(
            f"Unknown model type '{model_type}'. Available: {available}"
        )

    if model_enum not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_type}' not found in MODEL_REGISTRY.")

    return dict(MODEL_REGISTRY[model_enum])


__all__ = [
    "HARDWARE_SPECS_REGISTRY",
    "HardwareSpecs",
    "get_hardware_specs",
    "get_model_specs",
]
