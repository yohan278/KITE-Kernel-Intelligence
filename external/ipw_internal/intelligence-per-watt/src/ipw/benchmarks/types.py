"""Data types for energy characterization benchmarks.

This module defines the core data structures used by the benchmark system
to represent workload configurations, results, and extracted energy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Platform(Enum):
    """Supported hardware platforms."""

    MACOS = "macos"
    NVIDIA = "nvidia"
    ROCM = "rocm"


class DataType(Enum):
    """Numeric data types for benchmarks."""

    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT32 = "int32"
    INT8 = "int8"


class WorkloadType(Enum):
    """Types of benchmark workloads."""

    MEMORY_BANDWIDTH = "memory_bandwidth"  # Combined/HBM memory bandwidth
    COMPUTE_BOUND = "compute_bound"
    GEMM = "gemm"
    # Cache-level specific workloads
    CACHE_L1 = "cache_l1"  # L1 cache bandwidth
    CACHE_L2 = "cache_l2"  # L2 cache bandwidth
    HBM_BANDWIDTH = "hbm_bandwidth"  # HBM/DRAM bandwidth
    # Inference-level workloads (for energy scaling laws)
    INFERENCE_GEMM = "inference_gemm"  # Rectangular GEMM (prefill/decode shapes)
    ATTENTION = "attention"  # Scaled dot-product attention
    KV_CACHE_IO = "kv_cache_io"  # KV cache read/write patterns
    NCCL_COLLECTIVE = "nccl_collective"  # Multi-GPU collective communication
    BATCHED_DECODE = "batched_decode"  # Batched decode token generation


@dataclass(slots=True)
class WorkloadConfig:
    """Configuration for a single workload run.

    Attributes:
        workload_type: Type of workload to run.
        duration_seconds: How long to run the workload.
        use_zeros: If True, use zero-initialized data (measures control energy).
                   If False, use random data (measures control + datapath energy).
        data_type: Numeric precision for the workload.
        params: Workload-specific parameters (e.g., array_size_mb, arithmetic_intensity).
    """

    workload_type: WorkloadType
    duration_seconds: float = 10.0
    use_zeros: bool = False
    use_cuda_graphs: bool = False
    data_type: DataType = DataType.FP32
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkloadResult:
    """Result from a single workload execution.

    Attributes:
        workload_type: Type of workload that was run.
        config: Configuration used for this run.
        throughput: Primary performance metric (GB/s for memory, TFLOP/s for compute).
        throughput_unit: Unit of the throughput metric.
        bytes_transferred: Total bytes moved (for memory workloads).
        flops_executed: Total FLOPs computed (for compute workloads).
        duration_seconds: Actual duration of the workload.
    """

    workload_type: WorkloadType
    config: WorkloadConfig
    throughput: float
    throughput_unit: str
    bytes_transferred: Optional[int] = None
    flops_executed: Optional[int] = None
    duration_seconds: float = 0.0


@dataclass(slots=True)
class EnergyMeasurement:
    """Energy and power measurements during a workload.

    Attributes:
        cpu_energy_joules: CPU energy consumed during workload.
        gpu_energy_joules: GPU energy consumed during workload.
        ane_energy_joules: ANE (Apple Neural Engine) energy consumed (macOS only).
        avg_cpu_power_watts: Average CPU power during workload.
        avg_gpu_power_watts: Average GPU power during workload.
        avg_ane_power_watts: Average ANE power during workload (macOS only).
        max_cpu_power_watts: Maximum CPU power observed.
        max_gpu_power_watts: Maximum GPU power observed.
        max_ane_power_watts: Maximum ANE power observed (macOS only).
        duration_seconds: Duration over which energy was measured.
        sample_count: Number of power samples collected.
    """

    cpu_energy_joules: float
    gpu_energy_joules: float
    ane_energy_joules: float
    avg_cpu_power_watts: float
    avg_gpu_power_watts: float
    avg_ane_power_watts: float
    max_cpu_power_watts: float
    max_gpu_power_watts: float
    max_ane_power_watts: float
    duration_seconds: float
    sample_count: int

    @property
    def total_energy_joules(self) -> float:
        """Total energy (CPU + GPU + ANE) in joules."""
        return self.cpu_energy_joules + self.gpu_energy_joules + self.ane_energy_joules

    @property
    def avg_total_power_watts(self) -> float:
        """Average total power (CPU + GPU + ANE) in watts."""
        return self.avg_cpu_power_watts + self.avg_gpu_power_watts + self.avg_ane_power_watts


@dataclass(slots=True)
class BenchmarkResult:
    """Combined workload execution and energy measurement result.

    This is the primary output of running a benchmark - it pairs the
    workload's performance metrics with the energy consumed.
    """

    workload: WorkloadResult
    energy: EnergyMeasurement

    @property
    def energy_per_byte_pj(self) -> Optional[float]:
        """Energy per byte transferred in picojoules (for memory workloads)."""
        if self.workload.bytes_transferred and self.workload.bytes_transferred > 0:
            joules = self.energy.total_energy_joules
            return (joules / self.workload.bytes_transferred) * 1e12
        return None

    @property
    def energy_per_bit_pj(self) -> Optional[float]:
        """Energy per bit transferred in picojoules (for memory workloads)."""
        epb = self.energy_per_byte_pj
        return epb / 8.0 if epb is not None else None

    @property
    def energy_per_flop_pj(self) -> Optional[float]:
        """Energy per FLOP in picojoules (for compute workloads)."""
        if self.workload.flops_executed and self.workload.flops_executed > 0:
            joules = self.energy.total_energy_joules
            return (joules / self.workload.flops_executed) * 1e12
        return None


@dataclass(slots=True)
class EnergyParameters:
    """Extracted energy parameters for a platform.

    These parameters can be used to model and predict power consumption
    for arbitrary workloads on the characterized hardware.

    Following the methodology from:
    "Benchmark-driven Models for Energy Analysis and Attribution of
    GPU-Accelerated Supercomputing" (SC '25)

    Attributes:
        platform: Hardware platform these parameters apply to.
        hardware_name: Specific hardware identifier (e.g., "Apple M2 Pro").
        e_memory_pj_per_bit: Total energy per bit for memory access.
        e_memory_control_pj_per_bit: Control plane energy (from zero data).
        e_memory_datapath_pj_per_bit: Datapath energy (random - zero).
        e_compute_pj_per_flop: Total energy per FLOP by data type.
        e_compute_control_pj_per_flop: Control energy by data type.
        e_compute_datapath_pj_per_flop: Datapath energy by data type.
        e_gemm_pj_per_flop: Matrix multiplication energy by data type.
        p_idle_cpu_watts: CPU idle power.
        p_idle_gpu_watts: GPU idle power.
        tdp_watts: Thermal design power (if known).
    """

    platform: Platform
    hardware_name: str

    # Memory hierarchy (pJ/bit) - combined/legacy
    e_memory_pj_per_bit: float = 0.0
    e_memory_control_pj_per_bit: float = 0.0
    e_memory_datapath_pj_per_bit: float = 0.0

    # Cache-level memory energy (pJ/bit)
    # Based on SC'25 Table 3 methodology:
    # - L1: 1-2 pJ/bit (control: 1-2, datapath: 0-0.3)
    # - L2: 1-5 pJ/bit (control: 1-3, datapath: 0-2)
    # - HBM: 8-15 pJ/bit (control: 8-13, datapath: 1-5)

    # L1 Cache
    e_l1_pj_per_bit: float = 0.0
    e_l1_control_pj_per_bit: float = 0.0
    e_l1_datapath_pj_per_bit: float = 0.0

    # L2 Cache
    e_l2_pj_per_bit: float = 0.0
    e_l2_control_pj_per_bit: float = 0.0
    e_l2_datapath_pj_per_bit: float = 0.0

    # HBM/DRAM
    e_hbm_pj_per_bit: float = 0.0
    e_hbm_control_pj_per_bit: float = 0.0
    e_hbm_datapath_pj_per_bit: float = 0.0

    # Compute by data type (pJ/FLOP for FP types)
    e_compute_pj_per_flop: Dict[DataType, float] = field(default_factory=dict)
    e_compute_control_pj_per_flop: Dict[DataType, float] = field(default_factory=dict)
    e_compute_datapath_pj_per_flop: Dict[DataType, float] = field(default_factory=dict)

    # Integer compute (pJ/Op) - separate from FP compute
    # Important for address calculations, loop counters (~10-20% of app power)
    e_int32_pj_per_op: float = 0.0
    e_int32_control_pj_per_op: float = 0.0
    e_int32_datapath_pj_per_op: float = 0.0

    # GEMM/Matrix (pJ/FLOP) - separate from vector compute
    e_gemm_pj_per_flop: Dict[DataType, float] = field(default_factory=dict)
    e_gemm_control_pj_per_flop: Dict[DataType, float] = field(default_factory=dict)
    e_gemm_datapath_pj_per_flop: Dict[DataType, float] = field(default_factory=dict)

    # Inference-level energy parameters (for E_call scaling law)
    # Rectangular GEMM energy by phase (pJ/FLOP)
    e_inference_gemm_prefill_pj_per_flop: Dict[DataType, float] = field(
        default_factory=dict
    )
    e_inference_gemm_decode_pj_per_flop: Dict[DataType, float] = field(
        default_factory=dict
    )
    # Attention energy (pJ/FLOP)
    e_attention_pj_per_flop: float = 0.0
    # KV cache energy (pJ/bit)
    e_kv_read_pj_per_bit: float = 0.0
    e_kv_write_pj_per_bit: float = 0.0
    # Communication energy (pJ/bit)
    e_comm_pj_per_bit: float = 0.0
    # Batched decode exponent: E_decode ∝ B^β
    e_decode_batch_exponent: float = 1.0

    # Raw (CUDA graph) inference energy parameters — theoretical hardware floor
    # These measure the same quantities as above but with kernel launch overhead
    # eliminated via CUDA graph replay, isolating the true hardware cost.
    e_inference_gemm_prefill_raw_pj_per_flop: Dict[DataType, float] = field(
        default_factory=dict
    )
    e_inference_gemm_decode_raw_pj_per_flop: Dict[DataType, float] = field(
        default_factory=dict
    )
    e_attention_raw_pj_per_flop: float = 0.0
    e_kv_read_raw_pj_per_bit: float = 0.0
    e_kv_write_raw_pj_per_bit: float = 0.0
    e_decode_batch_raw_exponent: float = 1.0

    # Idle power
    p_idle_cpu_watts: float = 0.0
    p_idle_gpu_watts: float = 0.0

    # TDP (for power capping model)
    tdp_watts: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnergyParameters":
        """Load EnergyParameters from a dictionary (inverse of to_dict).

        Args:
            data: Dictionary as produced by to_dict() or loaded from JSON.

        Returns:
            EnergyParameters instance.
        """
        memory = data.get("memory", {})
        cache = data.get("cache_levels", {})
        compute_raw = data.get("compute", {})
        int32_raw = data.get("int32", {})
        gemm_raw = data.get("gemm", {})
        inference = data.get("inference", {})
        inference_raw = data.get("inference_raw", {})
        idle = data.get("idle_power", {})

        # Parse DataType-keyed dicts
        def _parse_dt_dict(raw: Dict[str, Dict[str, float]], key: str) -> Dict[DataType, float]:
            out: Dict[DataType, float] = {}
            for dt_str, vals in raw.items():
                try:
                    dt = DataType(dt_str)
                except ValueError:
                    continue
                out[dt] = vals.get(key, 0.0)
            return out

        def _parse_simple_dt_dict(raw: Dict[str, float]) -> Dict[DataType, float]:
            out: Dict[DataType, float] = {}
            for dt_str, val in raw.items():
                try:
                    dt = DataType(dt_str)
                except ValueError:
                    continue
                out[dt] = val
            return out

        return cls(
            platform=Platform(data.get("platform", "macos")),
            hardware_name=data.get("hardware_name", ""),
            e_memory_pj_per_bit=memory.get("total_pj_per_bit", 0.0),
            e_memory_control_pj_per_bit=memory.get("control_pj_per_bit", 0.0),
            e_memory_datapath_pj_per_bit=memory.get("datapath_pj_per_bit", 0.0),
            e_l1_pj_per_bit=cache.get("l1", {}).get("total_pj_per_bit", 0.0),
            e_l1_control_pj_per_bit=cache.get("l1", {}).get("control_pj_per_bit", 0.0),
            e_l1_datapath_pj_per_bit=cache.get("l1", {}).get("datapath_pj_per_bit", 0.0),
            e_l2_pj_per_bit=cache.get("l2", {}).get("total_pj_per_bit", 0.0),
            e_l2_control_pj_per_bit=cache.get("l2", {}).get("control_pj_per_bit", 0.0),
            e_l2_datapath_pj_per_bit=cache.get("l2", {}).get("datapath_pj_per_bit", 0.0),
            e_hbm_pj_per_bit=cache.get("hbm", {}).get("total_pj_per_bit", 0.0),
            e_hbm_control_pj_per_bit=cache.get("hbm", {}).get("control_pj_per_bit", 0.0),
            e_hbm_datapath_pj_per_bit=cache.get("hbm", {}).get("datapath_pj_per_bit", 0.0),
            e_compute_pj_per_flop=_parse_dt_dict(compute_raw, "total_pj_per_flop"),
            e_compute_control_pj_per_flop=_parse_dt_dict(compute_raw, "control_pj_per_flop"),
            e_compute_datapath_pj_per_flop=_parse_dt_dict(compute_raw, "datapath_pj_per_flop"),
            e_int32_pj_per_op=int32_raw.get("total_pj_per_op", 0.0),
            e_int32_control_pj_per_op=int32_raw.get("control_pj_per_op", 0.0),
            e_int32_datapath_pj_per_op=int32_raw.get("datapath_pj_per_op", 0.0),
            e_gemm_pj_per_flop=_parse_dt_dict(gemm_raw, "total_pj_per_flop"),
            e_gemm_control_pj_per_flop=_parse_dt_dict(gemm_raw, "control_pj_per_flop"),
            e_gemm_datapath_pj_per_flop=_parse_dt_dict(gemm_raw, "datapath_pj_per_flop"),
            e_inference_gemm_prefill_pj_per_flop=_parse_simple_dt_dict(
                inference.get("gemm_prefill", {})
            ),
            e_inference_gemm_decode_pj_per_flop=_parse_simple_dt_dict(
                inference.get("gemm_decode", {})
            ),
            e_attention_pj_per_flop=inference.get("attention_pj_per_flop", 0.0),
            e_kv_read_pj_per_bit=inference.get("kv_read_pj_per_bit", 0.0),
            e_kv_write_pj_per_bit=inference.get("kv_write_pj_per_bit", 0.0),
            e_comm_pj_per_bit=inference.get("comm_pj_per_bit", 0.0),
            e_decode_batch_exponent=inference.get("decode_batch_exponent", 1.0),
            e_inference_gemm_prefill_raw_pj_per_flop=_parse_simple_dt_dict(
                inference_raw.get("gemm_prefill", {})
            ),
            e_inference_gemm_decode_raw_pj_per_flop=_parse_simple_dt_dict(
                inference_raw.get("gemm_decode", {})
            ),
            e_attention_raw_pj_per_flop=inference_raw.get("attention_pj_per_flop", 0.0),
            e_kv_read_raw_pj_per_bit=inference_raw.get("kv_read_pj_per_bit", 0.0),
            e_kv_write_raw_pj_per_bit=inference_raw.get("kv_write_pj_per_bit", 0.0),
            e_decode_batch_raw_exponent=inference_raw.get("decode_batch_exponent", 1.0),
            p_idle_cpu_watts=idle.get("cpu_watts", 0.0),
            p_idle_gpu_watts=idle.get("gpu_watts", 0.0),
            tdp_watts=data.get("tdp_watts"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "platform": self.platform.value,
            "hardware_name": self.hardware_name,
            "memory": {
                "total_pj_per_bit": self.e_memory_pj_per_bit,
                "control_pj_per_bit": self.e_memory_control_pj_per_bit,
                "datapath_pj_per_bit": self.e_memory_datapath_pj_per_bit,
            },
            "cache_levels": {
                "l1": {
                    "total_pj_per_bit": self.e_l1_pj_per_bit,
                    "control_pj_per_bit": self.e_l1_control_pj_per_bit,
                    "datapath_pj_per_bit": self.e_l1_datapath_pj_per_bit,
                },
                "l2": {
                    "total_pj_per_bit": self.e_l2_pj_per_bit,
                    "control_pj_per_bit": self.e_l2_control_pj_per_bit,
                    "datapath_pj_per_bit": self.e_l2_datapath_pj_per_bit,
                },
                "hbm": {
                    "total_pj_per_bit": self.e_hbm_pj_per_bit,
                    "control_pj_per_bit": self.e_hbm_control_pj_per_bit,
                    "datapath_pj_per_bit": self.e_hbm_datapath_pj_per_bit,
                },
            },
            "compute": {
                dt.value: {
                    "total_pj_per_flop": self.e_compute_pj_per_flop.get(dt, 0.0),
                    "control_pj_per_flop": self.e_compute_control_pj_per_flop.get(dt, 0.0),
                    "datapath_pj_per_flop": self.e_compute_datapath_pj_per_flop.get(
                        dt, 0.0
                    ),
                }
                for dt in self.e_compute_pj_per_flop.keys()
            },
            "int32": {
                "total_pj_per_op": self.e_int32_pj_per_op,
                "control_pj_per_op": self.e_int32_control_pj_per_op,
                "datapath_pj_per_op": self.e_int32_datapath_pj_per_op,
            },
            "gemm": {
                dt.value: {
                    "total_pj_per_flop": self.e_gemm_pj_per_flop.get(dt, 0.0),
                    "control_pj_per_flop": self.e_gemm_control_pj_per_flop.get(dt, 0.0),
                    "datapath_pj_per_flop": self.e_gemm_datapath_pj_per_flop.get(dt, 0.0),
                }
                for dt in self.e_gemm_pj_per_flop.keys()
            },
            "inference": {
                "gemm_prefill": {
                    dt.value: pj
                    for dt, pj in self.e_inference_gemm_prefill_pj_per_flop.items()
                },
                "gemm_decode": {
                    dt.value: pj
                    for dt, pj in self.e_inference_gemm_decode_pj_per_flop.items()
                },
                "attention_pj_per_flop": self.e_attention_pj_per_flop,
                "kv_read_pj_per_bit": self.e_kv_read_pj_per_bit,
                "kv_write_pj_per_bit": self.e_kv_write_pj_per_bit,
                "comm_pj_per_bit": self.e_comm_pj_per_bit,
                "decode_batch_exponent": self.e_decode_batch_exponent,
            },
            "inference_raw": {
                "gemm_prefill": {
                    dt.value: pj
                    for dt, pj in self.e_inference_gemm_prefill_raw_pj_per_flop.items()
                },
                "gemm_decode": {
                    dt.value: pj
                    for dt, pj in self.e_inference_gemm_decode_raw_pj_per_flop.items()
                },
                "attention_pj_per_flop": self.e_attention_raw_pj_per_flop,
                "kv_read_pj_per_bit": self.e_kv_read_raw_pj_per_bit,
                "kv_write_pj_per_bit": self.e_kv_write_raw_pj_per_bit,
                "decode_batch_exponent": self.e_decode_batch_raw_exponent,
            },
            "idle_power": {
                "cpu_watts": self.p_idle_cpu_watts,
                "gpu_watts": self.p_idle_gpu_watts,
            },
            "tdp_watts": self.tdp_watts,
        }


__all__ = [
    "Platform",
    "DataType",
    "WorkloadType",
    "WorkloadConfig",
    "WorkloadResult",
    "EnergyMeasurement",
    "BenchmarkResult",
    "EnergyParameters",
]
