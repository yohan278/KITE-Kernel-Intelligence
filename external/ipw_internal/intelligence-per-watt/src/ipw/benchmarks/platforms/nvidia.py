"""NVIDIA CUDA benchmark suite implementation.

This module provides concrete workload implementations for NVIDIA GPUs
using PyTorch CUDA backend. Implements the full memory hierarchy model
from the SC'25 paper:

    "Benchmark-driven Models for Energy Analysis and Attribution of
    GPU-Accelerated Supercomputing" (Antepara et al., SC '25)
    DOI: https://doi.org/10.1145/3712285.3712359

Methodology Overview
====================

The paper extracts energy parameters by controlling working set size:
- L1 workload: Working set fits in L1 cache (~192KB per SM, ~14MB total on A100)
- L2 workload: Working set fits in L2 but not L1 (e.g., 10-30MB)
- HBM workload: Large streaming arrays that don't fit in caches (>100MB)

Energy Model (Equation 5 from paper):
    P = min(TDP, P_const + ε_L1×BW_L1 + ε_L2×BW_L2 + ε_HBM×BW_HBM + ε_FPU×PERF)

Cache Energy Extraction (Equations 6-8):
    ε_L2+L1 × BW = P - P_const - ε_VFPU × PERF_VFPU  (Eq. 6)
    ε_L1 × BW = P - P_const - ε_VFPU × PERF_VFPU     (Eq. 7)
    ε_L2 = ε_L2+L1 - ε_L1                             (Eq. 8)

IMPORTANT: Cache workloads use .add_(const) which performs FLOPs. The FPU
contribution is subtracted during analysis per Equations 6-7.

GEMM Energy Extraction (Equation 9):
    ε_MFPU × PERF = P - P_const - ε_L1×BW_L1 - ε_L2×BW_L2 - ε_HBM×BW_HBM

Note: Our implementation currently only subtracts estimated HBM contribution.
L1/L2 subtraction requires profiler integration (NSight Compute) to measure
actual cache bandwidths during GEMM execution. Our GEMM energy values will
be ~10-30% higher than the paper's Table 3 as a result.

Reference Values (Table 3 from paper, A100 GPU):
    - HBM: 13.11 pJ/bit (total), 8.47 control, 4.64 datapath
    - L2:  4.71 pJ/bit (total), 3.11 control, 1.60 datapath
    - L1:  1.59 pJ/bit (total), 1.26 control, 0.33 datapath
    - V-FP64: 28.50 pJ/FLOP (vector/CUDA cores)
    - M-FP64: 13.35 pJ/FLOP (tensor cores)
    - M-FP16: 0.70 pJ/FLOP (tensor cores)

Known Limitations:
    1. No profiler integration for L1/L2 bandwidth during GEMM
    2. Tensor core detection is implicit (cuBLAS handles this)
    3. Cache residency depends on hardware (SM count, L2 size)
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import List, Optional

from ipw.benchmarks.base import BenchmarkSuite
from ipw.benchmarks.types import (
    DataType,
    Platform,
    WorkloadConfig,
    WorkloadResult,
    WorkloadType,
)
from ipw.benchmarks.workloads.base import (
    AttentionWorkload,
    BatchedDecodeWorkload,
    ComputeWorkload,
    GEMMWorkload,
    InferenceGEMMWorkload,
    KVCacheWorkload,
    MemoryWorkload,
    NCCLCollectiveWorkload,
)
from ipw.core.registry import BenchmarkRegistry

logger = logging.getLogger(__name__)


def _check_cuda_available() -> bool:
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_gpu_properties():
    """Get GPU properties via PyTorch CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return props
    except Exception:
        return None


def _get_sm_count() -> int:
    """Get the number of SMs (Streaming Multiprocessors) on the GPU."""
    props = _get_gpu_properties()
    if props:
        return props.multi_processor_count
    return 108  # Default for A100


def _get_l2_cache_size_bytes() -> int:
    """Get L2 cache size in bytes.

    Returns L2 cache size from device properties if available,
    otherwise returns a conservative default.
    """
    props = _get_gpu_properties()
    if props and hasattr(props, 'l2_cache_size'):
        return props.l2_cache_size
    # Defaults for common GPUs:
    # A100: 40MB, H100: 50MB, V100: 6MB, RTX 4090: 72MB
    return 40 * 1024 * 1024  # 40MB default (A100)


class NVIDIACacheL1Workload(MemoryWorkload):
    """L1 cache bandwidth workload using PyTorch CUDA.

    Measures energy characteristics of L1 cache accesses by keeping the
    working set small enough to fit entirely in L1 cache across all SMs.

    On A100: ~192KB L1 per SM, 108 SMs = ~20MB theoretical
    We use ~128KB per SM to ensure L1 residency, giving ~14MB working set.

    Uses repeated access to the same data to keep it hot in L1.
    """

    workload_name = "CUDA L1 Cache Streaming"

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        """L1 workload supports FP32 and FP16."""
        return [DataType.FP32, DataType.FP16]

    def _calculate_l1_array_size(self, dtype_size: int) -> int:
        """Calculate array size to fit in L1 cache.

        Args:
            dtype_size: Bytes per element

        Returns:
            Number of elements for L1-resident working set
        """
        # ~128KB per SM to stay within L1 (conservatively under 192KB)
        l1_per_sm_bytes = 128 * 1024
        sm_count = _get_sm_count()
        total_l1_bytes = l1_per_sm_bytes * sm_count

        # Use 80% to ensure L1 residency
        target_bytes = int(total_l1_bytes * 0.8)
        return target_bytes // dtype_size

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup CUDA by running a few L1 cache iterations."""
        import torch

        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")
        element_size = 4 if dtype == torch.float32 else 2

        n_elements = self._calculate_l1_array_size(element_size)
        arr = torch.zeros(n_elements, dtype=dtype, device=device)

        # Multiple passes to warm up L1 cache
        for _ in range(10):
            arr.add_(0.0)
        torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Stream through L1-resident data, measuring bandwidth.

        Uses repeated access pattern to maximize L1 cache hits.
        The working set is sized to fit in L1 across all SMs.

        Args:
            config: Configuration with data_type and duration.

        Returns:
            WorkloadResult with bandwidth in GB/s.
        """
        import torch

        self.validate_config(config)

        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")
        element_size = 4 if dtype == torch.float32 else 2

        # Size array to fit in L1 cache
        n_elements = self._calculate_l1_array_size(element_size)
        actual_size_mb = (n_elements * element_size) / (1024 * 1024)

        # Initialize tensor - zeros for control energy, random for full energy
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=dtype, device=device)
            const = 0.0
        else:
            # Use bounded random values (paper: 1.0 to 2.0, const = -2.0)
            arr = torch.empty(n_elements, dtype=dtype, device=device).uniform_(1.0, 2.0)
            const = -2.0

        bytes_transferred = 0
        iterations = 0

        # Synchronize before timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            # In-place add: read + write = 2x array size in bytes transferred
            # Repeated access to same small array keeps data in L1
            arr.add_(const)
            bytes_transferred += 2 * n_elements * element_size
            iterations += 1
            # Periodic sync to get accurate timing
            if iterations % 200 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9

        result = WorkloadResult(
            workload_type=WorkloadType.CACHE_L1,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

        # Store working set size in config params for analysis
        config.params["actual_array_size_mb"] = actual_size_mb
        config.params["cache_level"] = "L1"

        return result

    def _to_torch_dtype(self, dt: DataType):
        """Convert DataType enum to torch dtype."""
        import torch
        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
        }
        if dt not in mapping:
            raise ValueError(
                f"Unsupported data type {dt.value} for {self.workload_name}. "
                f"Supported: {[d.value for d in mapping.keys()]}"
            )
        return mapping[dt]


class NVIDIACacheL2Workload(MemoryWorkload):
    """L2 cache bandwidth workload using PyTorch CUDA.

    Measures energy characteristics of L2 cache accesses by using a
    working set that fits in L2 but exceeds L1 capacity.

    On A100: 40MB L2 cache
    Working set: 20-35MB (larger than all L1s combined, fits in L2)
    """

    workload_name = "CUDA L2 Cache Streaming"

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        """L2 workload supports FP32 and FP16."""
        return [DataType.FP32, DataType.FP16]

    def _calculate_l2_array_size(self, dtype_size: int) -> int:
        """Calculate array size to fit in L2 but not L1.

        Args:
            dtype_size: Bytes per element

        Returns:
            Number of elements for L2-resident working set
        """
        l2_cache_bytes = _get_l2_cache_size_bytes()

        # Use 60% of L2 to ensure L2 residency but stay within bounds
        # This should be larger than total L1 but fit in L2
        target_bytes = int(l2_cache_bytes * 0.6)

        # Ensure minimum size that exceeds L1
        l1_per_sm = 192 * 1024
        sm_count = _get_sm_count()
        total_l1 = l1_per_sm * sm_count

        if target_bytes < total_l1 * 1.5:
            target_bytes = int(total_l1 * 1.5)

        return target_bytes // dtype_size

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup CUDA by running a few L2 cache iterations."""
        import torch

        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")
        element_size = 4 if dtype == torch.float32 else 2

        n_elements = self._calculate_l2_array_size(element_size)
        arr = torch.zeros(n_elements, dtype=dtype, device=device)

        # Multiple passes to warm up L2 cache
        for _ in range(5):
            arr.add_(0.0)
        torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Stream through L2-resident data, measuring bandwidth.

        Args:
            config: Configuration with data_type and duration.

        Returns:
            WorkloadResult with bandwidth in GB/s.
        """
        import torch

        self.validate_config(config)

        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")
        element_size = 4 if dtype == torch.float32 else 2

        # Size array to fit in L2 but not L1
        n_elements = self._calculate_l2_array_size(element_size)
        actual_size_mb = (n_elements * element_size) / (1024 * 1024)

        # Initialize tensor
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=dtype, device=device)
            const = 0.0
        else:
            arr = torch.empty(n_elements, dtype=dtype, device=device).uniform_(1.0, 2.0)
            const = -2.0

        bytes_transferred = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            arr.add_(const)
            bytes_transferred += 2 * n_elements * element_size
            iterations += 1
            if iterations % 100 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9

        result = WorkloadResult(
            workload_type=WorkloadType.CACHE_L2,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

        config.params["actual_array_size_mb"] = actual_size_mb
        config.params["cache_level"] = "L2"

        return result

    def _to_torch_dtype(self, dt: DataType):
        """Convert DataType enum to torch dtype."""
        import torch
        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
        }
        if dt not in mapping:
            raise ValueError(
                f"Unsupported data type {dt.value} for {self.workload_name}. "
                f"Supported: {[d.value for d in mapping.keys()]}"
            )
        return mapping[dt]


class NVIDIAHBMWorkload(MemoryWorkload):
    """HBM (High Bandwidth Memory) streaming workload using PyTorch CUDA.

    Measures energy characteristics of HBM accesses by using large
    streaming arrays that don't fit in any cache level.

    Working set: >100MB (exceeds L2 cache, forces HBM traffic)
    Uses simple add operations to maximize memory throughput.
    """

    workload_name = "CUDA HBM Streaming"

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        """HBM workload supports FP32 and FP16."""
        return [DataType.FP32, DataType.FP16]

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup CUDA by running a few HBM streaming iterations."""
        import torch

        array_size_mb = config.params.get("array_size_mb", 256)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        element_size = 4 if dtype == torch.float32 else 2
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        arr = torch.zeros(n_elements, dtype=dtype, device=device)
        for _ in range(3):
            arr.add_(0.0)
        torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Stream through HBM, measuring bandwidth.

        Large arrays ensure data comes from HBM rather than cache.
        Simple add operations minimize compute overhead.

        Args:
            config: Must include params["array_size_mb"] (default 256MB).

        Returns:
            WorkloadResult with bandwidth in GB/s.
        """
        import torch

        self.validate_config(config)

        # Default to 256MB which exceeds typical L2 cache sizes
        array_size_mb = config.params.get("array_size_mb", 256)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        element_size = 4 if dtype == torch.float32 else 2
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        # Initialize tensor
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=dtype, device=device)
            const = 0.0
        else:
            arr = torch.empty(n_elements, dtype=dtype, device=device).uniform_(1.0, 2.0)
            const = -2.0

        bytes_transferred = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            # In-place add: read + write = 2x array size
            arr.add_(const)
            bytes_transferred += 2 * n_elements * element_size
            iterations += 1
            if iterations % 50 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9

        result = WorkloadResult(
            workload_type=WorkloadType.HBM_BANDWIDTH,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

        config.params["actual_array_size_mb"] = array_size_mb
        config.params["cache_level"] = "HBM"

        return result

    def _to_torch_dtype(self, dt: DataType):
        """Convert DataType enum to torch dtype."""
        import torch
        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
        }
        if dt not in mapping:
            raise ValueError(
                f"Unsupported data type {dt.value} for {self.workload_name}. "
                f"Supported: {[d.value for d in mapping.keys()]}"
            )
        return mapping[dt]


class NVIDIAComputeWorkload(ComputeWorkload):
    """Mixbench-style compute workload using PyTorch CUDA.

    Varies arithmetic intensity (FLOPs per byte) to enable separation
    of memory-bound and compute-bound energy consumption. Based on
    the SC'25 paper methodology.

    Uses polynomial evaluation pattern: y = ((x*c+c)*c+c)...
    Each multiply-add is 2 FLOPs per element.

    Paper methodology:
    - Zeros for control energy measurement
    - Random values (1.0-2.0) with const=-2.0 for datapath measurement
      (causes oscillation to keep values bounded)
    """

    workload_name = "CUDA Compute Sweep"

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        """CUDA supports FP64, FP32, FP16, and BF16."""
        import torch
        supported = [DataType.FP64, DataType.FP32, DataType.FP16]
        # BF16 requires Ampere or newer
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            supported.append(DataType.BF16)
        elif torch.cuda.is_available():
            # Fallback check for older PyTorch versions
            try:
                test = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
                supported.append(DataType.BF16)
            except Exception:
                pass
        return supported

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup CUDA by compiling kernels with a few compute iterations."""
        import torch

        array_size_mb = config.params.get("array_size_mb", 100)
        arithmetic_intensity = config.params.get("arithmetic_intensity", 16)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        element_size = self._get_element_size(dtype)
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        arr = torch.zeros(n_elements, dtype=dtype, device=device)
        warmup_result = arr.clone()
        for _ in range(min(arithmetic_intensity, 4)):
            warmup_result.mul_(0.0).add_(0.0)
        torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute compute at varying arithmetic intensity.

        Performs polynomial evaluation: y = ((x*c+c)*c+c)...
        Each multiply-add is 2 FLOPs per element.

        Args:
            config: Must include params["arithmetic_intensity"] and
                    params["array_size_mb"].

        Returns:
            WorkloadResult with compute rate in TFLOP/s.
        """
        import torch

        self.validate_config(config)

        array_size_mb = config.params.get("array_size_mb", 100)
        arithmetic_intensity = config.params.get("arithmetic_intensity", 16)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        element_size = self._get_element_size(dtype)
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        # Initialize tensor
        # Paper methodology: random values 1.0-2.0, const=-2.0 to oscillate between -1 and 2
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=dtype, device=device)
            const = 0.0
        else:
            arr = torch.empty(n_elements, dtype=dtype, device=device).uniform_(1.0, 2.0)
            const = -2.0  # Causes values to oscillate, staying bounded

        total_flops = 0
        iterations = 0
        result = arr.clone()

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            # Copy source data, then compute in-place
            result.copy_(arr)
            for _ in range(arithmetic_intensity):
                # Fused multiply-add: 2 FLOPs per element
                result.mul_(const).add_(const)
            total_flops += n_elements * arithmetic_intensity * 2
            iterations += 1
            if iterations % 50 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = total_flops / elapsed / 1e12

        return WorkloadResult(
            workload_type=WorkloadType.COMPUTE_BOUND,
            config=config,
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _to_torch_dtype(self, dt: DataType):
        """Convert DataType enum to torch dtype."""
        import torch
        mapping = {
            DataType.FP64: torch.float64,
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }
        if dt not in mapping:
            raise ValueError(
                f"Unsupported data type {dt.value} for {self.workload_name}. "
                f"Supported: {[d.value for d in mapping.keys()]}"
            )
        return mapping[dt]

    def _get_element_size(self, dtype) -> int:
        """Get size in bytes for a torch dtype."""
        import torch
        sizes = {
            torch.float64: 8,
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        return sizes.get(dtype, 4)


class NVIDIAIntegerComputeWorkload(ComputeWorkload):
    """INT32 compute workload using PyTorch CUDA.

    Measures integer operation energy (IMAD - integer multiply-add).
    Important because ~10-20% of application power goes to integer ops
    (loop counters, address calculations, array indexing).

    Based on the SC'25 paper methodology which includes INT32 energy
    parameters alongside floating-point measurements.
    """

    workload_name = "CUDA INT32 Compute"

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        """Only supports INT32."""
        return [DataType.INT32]

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup CUDA by running a few integer compute iterations."""
        import torch

        array_size_mb = config.params.get("array_size_mb", 100)
        device = torch.device("cuda")

        # INT32 = 4 bytes per element
        n_elements = (array_size_mb * 1024 * 1024) // 4

        arr = torch.zeros(n_elements, dtype=torch.int32, device=device)
        warmup_result = arr.clone()
        for _ in range(4):
            warmup_result.mul_(1).add_(1)
        torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute integer compute at varying arithmetic intensity.

        Performs integer multiply-add operations: y = ((x*c+c)*c+c)...
        Each multiply-add is 2 integer OPs per element.

        Args:
            config: Must include params["arithmetic_intensity"] and
                    params["array_size_mb"].

        Returns:
            WorkloadResult with compute rate in TOP/s (tera-ops per second).
        """
        import torch

        self.validate_config(config)

        array_size_mb = config.params.get("array_size_mb", 100)
        arithmetic_intensity = config.params.get("arithmetic_intensity", 16)
        device = torch.device("cuda")

        # INT32 = 4 bytes per element
        n_elements = (array_size_mb * 1024 * 1024) // 4

        # Initialize tensor
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=torch.int32, device=device)
            const = 0
        else:
            # Random integers in bounded range to prevent overflow
            arr = torch.randint(1, 100, (n_elements,), dtype=torch.int32, device=device)
            # Use const that exercises the multiplier while keeping values bounded
            # Multiply by 3, add -2 to oscillate: e.g., 50*3-2=148, 148*3-2=442, etc.
            # Use modulo in the computation to prevent overflow
            const_mul = 3
            const_add = -2

        total_ops = 0
        iterations = 0
        result = arr.clone()

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            result.copy_(arr)
            if config.use_zeros:
                for _ in range(arithmetic_intensity):
                    # Integer multiply-add: 2 OPs per element
                    result.mul_(const).add_(const)
            else:
                for _ in range(arithmetic_intensity):
                    # Integer multiply-add: 2 OPs per element
                    # Use modulo to prevent overflow while exercising the multiplier
                    result.mul_(const_mul).add_(const_add)
                    result.remainder_(1000000)  # Keep values bounded
            total_ops += n_elements * arithmetic_intensity * 2
            iterations += 1
            if iterations % 50 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tops = total_ops / elapsed / 1e12

        return WorkloadResult(
            workload_type=WorkloadType.COMPUTE_BOUND,
            config=config,
            throughput=tops,
            throughput_unit="TOP/s",
            flops_executed=total_ops,  # Actually OPs, but reusing field
            duration_seconds=elapsed,
        )


class NVIDIAGEMMWorkload(GEMMWorkload):
    """GEMM workload using PyTorch CUDA backend (cuBLAS).

    Performs matrix multiplication on NVIDIA GPUs, automatically utilizing
    tensor cores when available (Volta and newer with FP16/BF16/TF32).

    cuBLAS performance characteristics:
    - FP64: Standard CUDA cores
    - FP32: TF32 on Ampere+ (automatic with torch.backends.cuda.matmul.allow_tf32)
    - FP16: Tensor cores on Volta+
    - BF16: Tensor cores on Ampere+
    """

    workload_name = "CUDA cuBLAS GEMM"

    def is_available(self) -> bool:
        """Check if PyTorch CUDA backend is available."""
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        """Support FP64, FP32 (with TF32 on Ampere+), FP16, and BF16."""
        import torch
        supported = [DataType.FP64, DataType.FP32, DataType.FP16]
        # BF16 requires Ampere or newer
        if torch.cuda.is_available():
            try:
                test = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
                supported.append(DataType.BF16)
            except Exception:
                pass
        return supported

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute matrix multiplication on CUDA via cuBLAS.

        Args:
            config: Must include params["matrix_size"].

        Returns:
            WorkloadResult with compute rate in TFLOP/s.
        """
        import torch

        self.validate_config(config)

        matrix_size = config.params.get("matrix_size", 4096)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        # Enable TF32 for FP32 on Ampere+ for tensor core utilization
        # This is the default in PyTorch 1.7+, but we make it explicit
        if config.data_type == DataType.FP32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Initialize matrices
        if config.use_zeros:
            A = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
            B = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
        else:
            # Scale to prevent overflow with FP16
            scale = 1.0 / matrix_size if dtype in [torch.float16, torch.bfloat16] else 1.0
            A = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device) * scale
            B = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device) * scale

        # GEMM: C = A @ B
        # FLOPs = 2 * M * N * K (multiply-accumulate for each element)
        flops_per_mm = 2 * matrix_size * matrix_size * matrix_size
        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            _ = torch.mm(A, B)
            total_flops += flops_per_mm
            iterations += 1
            if iterations % 20 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tflops = total_flops / elapsed / 1e12

        return WorkloadResult(
            workload_type=WorkloadType.GEMM,
            config=config,
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup CUDA by running a few GEMM iterations."""
        import torch

        matrix_size = config.params.get("matrix_size", 4096)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        if config.use_zeros:
            A = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
            B = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
        else:
            # Apply same scaling as run() for FP16/BF16
            scale = 1.0 / matrix_size if dtype in [torch.float16, torch.bfloat16] else 1.0
            A = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device) * scale
            B = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device) * scale

        for _ in range(5):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()

    def _to_torch_dtype(self, dt: DataType):
        """Convert DataType enum to torch dtype."""
        import torch

        mapping = {
            DataType.FP64: torch.float64,
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }
        if dt not in mapping:
            raise ValueError(
                f"Unsupported data type {dt.value} for {self.workload_name}. "
                f"Supported: {[d.value for d in mapping.keys()]}"
            )
        return mapping[dt]


class NVIDIAInferenceGEMMWorkload(InferenceGEMMWorkload):
    """Inference-shaped rectangular GEMM workload using PyTorch CUDA.

    Measures energy of non-square matrix multiplications typical in LLM inference:
    - Prefill: tall-skinny GEMM (B*S, d) × (d, d_ff), FLOPs = 2*B*S*d*d_ff
    - Decode: flat GEMM (B, d) × (d, d_ff), FLOPs = 2*B*d*d_ff
    """

    workload_name = "CUDA Inference GEMM"

    def is_available(self) -> bool:
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        import torch
        supported = [DataType.FP32, DataType.FP16]
        if torch.cuda.is_available():
            try:
                torch.zeros(1, dtype=torch.bfloat16, device="cuda")
                supported.append(DataType.BF16)
            except Exception:
                pass
        return supported

    def warmup(self, config: WorkloadConfig) -> None:
        import torch

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        hidden_dim = config.params.get("hidden_dim", 4096)
        ff_dim = config.params.get("ff_dim", 11008)
        mode = config.params.get("mode", "prefill")
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        M = batch_size * seq_len if mode == "prefill" else batch_size
        A = torch.randn(M, hidden_dim, dtype=dtype, device=device)
        B = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)
        for _ in range(5):
            torch.mm(A, B)
        torch.cuda.synchronize()

        if config.use_cuda_graphs:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.mm(A, B)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torch.mm(A, B)
            graph.replay()
            torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch

        self.validate_config(config)

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        hidden_dim = config.params.get("hidden_dim", 4096)
        ff_dim = config.params.get("ff_dim", 11008)
        mode = config.params.get("mode", "prefill")
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        # Prefill: (B*S, d) × (d, d_ff); Decode: (B, d) × (d, d_ff)
        M = batch_size * seq_len if mode == "prefill" else batch_size
        flops_per_mm = 2 * M * hidden_dim * ff_dim

        A = torch.randn(M, hidden_dim, dtype=dtype, device=device)
        W = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)

        if config.use_cuda_graphs:
            return self._run_cuda_graph(config, A, W, flops_per_mm)

        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            torch.mm(A, W)
            total_flops += flops_per_mm
            iterations += 1
            if iterations % 20 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = total_flops / elapsed / 1e12

        return WorkloadResult(
            workload_type=WorkloadType.INFERENCE_GEMM,
            config=config,
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _run_cuda_graph(self, config, A, W, flops_per_mm):
        import torch

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.mm(A, W)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            torch.mm(A, W)

        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            graph.replay()
            total_flops += flops_per_mm
            iterations += 1
            if iterations % 100 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = total_flops / elapsed / 1e12

        return WorkloadResult(
            workload_type=WorkloadType.INFERENCE_GEMM,
            config=config,
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _to_torch_dtype(self, dt: DataType):
        import torch
        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }
        if dt not in mapping:
            raise ValueError(f"Unsupported data type {dt.value} for {self.workload_name}")
        return mapping[dt]


class NVIDIAAttentionWorkload(AttentionWorkload):
    """Scaled dot-product attention workload using PyTorch CUDA.

    Measures attention energy using F.scaled_dot_product_attention,
    which dispatches to FlashAttention when available.
    FLOPs ≈ 4 * B * H * S² * d_h (QK^T + softmax@V)
    """

    workload_name = "CUDA Attention"

    def is_available(self) -> bool:
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        return [DataType.FP32, DataType.FP16]

    def warmup(self, config: WorkloadConfig) -> None:
        import torch

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        num_heads = config.params.get("num_heads", 32)
        head_dim = config.params.get("head_dim", 128)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

        if config.use_cuda_graphs:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            graph.replay()
            torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch
        import torch.nn.functional as F

        self.validate_config(config)

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        num_heads = config.params.get("num_heads", 32)
        head_dim = config.params.get("head_dim", 128)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        # Detect FlashAttention availability
        uses_flash = False
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            uses_flash = torch.backends.cuda.flash_sdp_enabled()

        # FLOPs: QK^T = 2*B*H*S*S*d_h, softmax@V = 2*B*H*S*S*d_h -> 4*B*H*S²*d_h
        flops_per_attn = 4 * batch_size * num_heads * seq_len * seq_len * head_dim

        try:
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
            K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
            V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        except RuntimeError as e:
            # OOM guard
            raise RuntimeError(
                f"OOM allocating attention tensors (B={batch_size}, H={num_heads}, "
                f"S={seq_len}, d_h={head_dim}): {e}"
            ) from e

        # Store FlashAttention info in params for analysis
        config.params["uses_flash_attention"] = uses_flash

        if config.use_cuda_graphs:
            return self._run_cuda_graph(config, Q, K, V, flops_per_attn)

        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            try:
                F.scaled_dot_product_attention(Q, K, V)
            except RuntimeError:
                break  # OOM during computation
            total_flops += flops_per_attn
            iterations += 1
            if iterations % 10 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = total_flops / elapsed / 1e12 if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.ATTENTION,
            config=config,
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _run_cuda_graph(self, config, Q, K, V, flops_per_attn):
        import torch
        import torch.nn.functional as F

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            F.scaled_dot_product_attention(Q, K, V)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            F.scaled_dot_product_attention(Q, K, V)

        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            graph.replay()
            total_flops += flops_per_attn
            iterations += 1
            if iterations % 100 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = total_flops / elapsed / 1e12 if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.ATTENTION,
            config=config,
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _to_torch_dtype(self, dt: DataType):
        import torch
        mapping = {DataType.FP32: torch.float32, DataType.FP16: torch.float16}
        if dt not in mapping:
            raise ValueError(f"Unsupported data type {dt.value} for {self.workload_name}")
        return mapping[dt]


class NVIDIAKVCacheWorkload(KVCacheWorkload):
    """KV cache I/O workload using PyTorch CUDA.

    Measures energy of KV cache access patterns:
    - Write mode: index_copy_ sequential append (prefill pattern)
    - Read mode: index_select with strided indices (decode gather pattern)
    """

    workload_name = "CUDA KV Cache I/O"

    def is_available(self) -> bool:
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        return [DataType.FP32, DataType.FP16]

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch

        self.validate_config(config)

        cache_entries = config.params.get("cache_entries", 2048)
        num_heads = config.params.get("num_heads", 32)
        head_dim = config.params.get("head_dim", 128)
        batch_size = config.params.get("batch_size", 1)
        mode = config.params.get("mode", "read")
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")
        element_size = 4 if dtype == torch.float32 else 2

        # Cache tensors for K and V
        k_cache = torch.randn(cache_entries, num_heads, head_dim, dtype=dtype, device=device)
        v_cache = torch.randn(cache_entries, num_heads, head_dim, dtype=dtype, device=device)

        entry_bytes = num_heads * head_dim * element_size

        if config.use_cuda_graphs:
            return self._run_cuda_graph(
                config, k_cache, v_cache, mode, batch_size,
                cache_entries, entry_bytes, dtype, device,
            )

        bytes_transferred = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        if mode == "write":
            # Prefill append: write sequential entries
            new_k = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            new_v = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            indices = torch.arange(batch_size, device=device) % cache_entries

            while time.perf_counter() < deadline:
                k_cache.index_copy_(0, indices, new_k)
                v_cache.index_copy_(0, indices, new_v)
                # Write bytes: 2 caches × batch_size entries
                bytes_transferred += 2 * batch_size * entry_bytes
                iterations += 1
                if iterations % 100 == 0:
                    torch.cuda.synchronize()
        else:
            # Decode gather: read strided entries
            gather_size = min(batch_size, cache_entries)
            indices = torch.randint(0, cache_entries, (gather_size,), device=device)

            while time.perf_counter() < deadline:
                torch.index_select(k_cache, 0, indices)
                torch.index_select(v_cache, 0, indices)
                # Read bytes: 2 caches × gather_size entries
                bytes_transferred += 2 * gather_size * entry_bytes
                iterations += 1
                if iterations % 100 == 0:
                    torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9 if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.KV_CACHE_IO,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

    def _run_cuda_graph(
        self, config, k_cache, v_cache, mode, batch_size,
        cache_entries, entry_bytes, dtype, device,
    ):
        import torch

        num_heads = config.params.get("num_heads", 32)
        head_dim = config.params.get("head_dim", 128)

        stream = torch.cuda.Stream()

        if mode == "write":
            new_k = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            new_v = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            indices = torch.arange(batch_size, device=device) % cache_entries
            bytes_per_iter = 2 * batch_size * entry_bytes

            with torch.cuda.stream(stream):
                k_cache.index_copy_(0, indices, new_k)
                v_cache.index_copy_(0, indices, new_v)
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                k_cache.index_copy_(0, indices, new_k)
                v_cache.index_copy_(0, indices, new_v)
        else:
            gather_size = min(batch_size, cache_entries)
            indices = torch.randint(0, cache_entries, (gather_size,), device=device)
            bytes_per_iter = 2 * gather_size * entry_bytes

            with torch.cuda.stream(stream):
                torch.index_select(k_cache, 0, indices)
                torch.index_select(v_cache, 0, indices)
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torch.index_select(k_cache, 0, indices)
                torch.index_select(v_cache, 0, indices)

        bytes_transferred = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            graph.replay()
            bytes_transferred += bytes_per_iter
            iterations += 1
            if iterations % 100 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9 if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.KV_CACHE_IO,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

    def _to_torch_dtype(self, dt: DataType):
        import torch
        mapping = {DataType.FP32: torch.float32, DataType.FP16: torch.float16}
        if dt not in mapping:
            raise ValueError(f"Unsupported data type {dt.value} for {self.workload_name}")
        return mapping[dt]


class NVIDIANCCLCollectiveWorkload(NCCLCollectiveWorkload):
    """NCCL collective communication workload (stub for single-GPU).

    On single-GPU systems, this workload reports as unavailable.
    On multi-GPU systems with torch.distributed initialized, it
    performs all-reduce or all-gather operations.
    """

    workload_name = "CUDA NCCL Collective"

    def is_available(self) -> bool:
        """Returns True only if multi-GPU distributed is initialized."""
        try:
            import torch.distributed
            return torch.distributed.is_initialized()
        except Exception:
            return False

    def supported_data_types(self) -> List[DataType]:
        return [DataType.FP32, DataType.FP16]

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch.distributed

        if not torch.distributed.is_initialized():
            raise NotImplementedError(
                "NCCL collective workload requires multi-GPU with "
                "torch.distributed initialized. Single-GPU stub."
            )

        import torch

        self.validate_config(config)

        if config.use_cuda_graphs:
            logger.warning(
                "CUDA graphs not supported for NCCL collectives (peer-to-peer "
                "network traffic is not graphable). Running in standard mode."
            )

        message_size_mb = config.params.get("message_size_mb", 100)
        collective_type = config.params.get("collective_type", "all_reduce")
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        element_size = 4 if dtype == torch.float32 else 2
        n_elements = (message_size_mb * 1024 * 1024) // element_size
        tensor = torch.randn(n_elements, dtype=dtype, device=device)

        bytes_transferred = 0
        iterations = 0
        msg_bytes = n_elements * element_size

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            if collective_type == "all_gather":
                world_size = torch.distributed.get_world_size()
                output = [torch.empty_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(output, tensor)
                bytes_transferred += msg_bytes * world_size
            else:
                torch.distributed.all_reduce(tensor)
                bytes_transferred += msg_bytes * 2  # send + receive
            iterations += 1
            if iterations % 10 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9 if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.NCCL_COLLECTIVE,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

    def _to_torch_dtype(self, dt: DataType):
        import torch
        mapping = {DataType.FP32: torch.float32, DataType.FP16: torch.float16}
        if dt not in mapping:
            raise ValueError(f"Unsupported data type {dt.value} for {self.workload_name}")
        return mapping[dt]


class NVIDIABatchedDecodeWorkload(BatchedDecodeWorkload):
    """Batched decode workload using PyTorch CUDA.

    Simulates batched token generation to measure how decode energy
    scales with batch size. Allocates shared weight matrices and runs
    FFN-like computation (x @ W1, result @ W2) for B sequences.
    """

    workload_name = "CUDA Batched Decode"

    def is_available(self) -> bool:
        return _check_cuda_available()

    def supported_data_types(self) -> List[DataType]:
        return [DataType.FP32, DataType.FP16]

    def warmup(self, config: WorkloadConfig) -> None:
        import torch

        batch_size = config.params.get("batch_size", 1)
        hidden_dim = config.params.get("hidden_dim", 4096)
        ff_dim = config.params.get("ff_dim", 11008)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
        W1 = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)
        W2 = torch.randn(ff_dim, hidden_dim, dtype=dtype, device=device)
        for _ in range(3):
            h = torch.mm(x, W1)
            torch.mm(h, W2)
        torch.cuda.synchronize()

        if config.use_cuda_graphs:
            num_layers = config.params.get("num_layers", 1)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                h = x
                for _ in range(num_layers):
                    h = torch.mm(h, W1)
                    h = torch.mm(h, W2)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                h = x
                for _ in range(num_layers):
                    h = torch.mm(h, W1)
                    h = torch.mm(h, W2)
            graph.replay()
            torch.cuda.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch

        self.validate_config(config)

        batch_size = config.params.get("batch_size", 1)
        hidden_dim = config.params.get("hidden_dim", 4096)
        ff_dim = config.params.get("ff_dim", 11008)
        num_layers = config.params.get("num_layers", 1)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("cuda")

        # Shared weight matrices (simulating model weights)
        W1 = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)
        W2 = torch.randn(ff_dim, hidden_dim, dtype=dtype, device=device)
        x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)

        # FLOPs per iteration: num_layers × (2 × 2*B*d*d_ff)
        # = num_layers × 4*B*d*d_ff
        flops_per_iter = num_layers * 4 * batch_size * hidden_dim * ff_dim

        if config.use_cuda_graphs:
            return self._run_cuda_graph(
                config, x, W1, W2, num_layers, flops_per_iter, batch_size,
            )

        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            h = x
            for _ in range(num_layers):
                h = torch.mm(h, W1)
                h = torch.mm(h, W2)
            total_flops += flops_per_iter
            iterations += 1
            if iterations % 20 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tokens_per_second = batch_size * iterations / elapsed if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.BATCHED_DECODE,
            config=config,
            throughput=tokens_per_second,
            throughput_unit="tokens/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _run_cuda_graph(self, config, x, W1, W2, num_layers, flops_per_iter, batch_size):
        import torch

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            h = x
            for _ in range(num_layers):
                h = torch.mm(h, W1)
                h = torch.mm(h, W2)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            h = x
            for _ in range(num_layers):
                h = torch.mm(h, W1)
                h = torch.mm(h, W2)

        total_flops = 0
        iterations = 0

        torch.cuda.synchronize()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            graph.replay()
            total_flops += flops_per_iter
            iterations += 1
            if iterations % 100 == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tokens_per_second = batch_size * iterations / elapsed if elapsed > 0 else 0.0

        return WorkloadResult(
            workload_type=WorkloadType.BATCHED_DECODE,
            config=config,
            throughput=tokens_per_second,
            throughput_unit="tokens/s",
            flops_executed=total_flops,
            duration_seconds=elapsed,
        )

    def _to_torch_dtype(self, dt: DataType):
        import torch
        mapping = {DataType.FP32: torch.float32, DataType.FP16: torch.float16}
        if dt not in mapping:
            raise ValueError(f"Unsupported data type {dt.value} for {self.workload_name}")
        return mapping[dt]


@BenchmarkRegistry.register("nvidia")
class NVIDIABenchmarkSuite(BenchmarkSuite):
    """Benchmark suite for NVIDIA GPUs.

    Implements the full SC'25 paper methodology (Antepara et al.) with
    separate L1, L2, and HBM memory workloads for accurate energy
    characterization of the memory hierarchy.

    Energy Model:
        P = min(TDP, P_const + epsilon_L1*BW_L1 + epsilon_L2*BW_L2 +
                epsilon_HBM*BW_HBM + epsilon_FPU*PERF)

    By running benchmarks at different operating points and solving the
    resulting linear system, we can extract separate energy parameters
    for each memory tier and compute operations.
    """

    platform = Platform.NVIDIA
    platform_name = "NVIDIA CUDA"

    @classmethod
    def is_available(cls) -> bool:
        """Check if running on a system with NVIDIA GPU and CUDA."""
        return _check_cuda_available()

    @classmethod
    def detect_hardware(cls) -> str:
        """Detect NVIDIA GPU name.

        Uses PyTorch CUDA or pynvml to get the GPU model name.

        Returns:
            GPU name like "NVIDIA A100-SXM4-40GB" or fallback string.
        """
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass

        # Try pynvml as fallback
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            pynvml.nvmlShutdown()
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            return name
        except Exception:
            pass

        # Try nvidia-smi as last resort
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return "NVIDIA GPU (unknown)"

    @classmethod
    def get_gpu_memory_gb(cls) -> float:
        """Get GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024**3)
        except Exception:
            pass
        return 0.0

    @classmethod
    def get_compute_capability(cls) -> tuple:
        """Get CUDA compute capability (major, minor)."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return (props.major, props.minor)
        except Exception:
            pass
        return (0, 0)

    def __init__(self) -> None:
        """Initialize the NVIDIA benchmark suite."""
        self._workloads: Optional[List] = None

    def get_workloads(self) -> List:
        """Return available workloads for NVIDIA GPUs.

        Returns:
            List of CUDA workload instances for energy characterization.
            Includes separate workloads for L1, L2, and HBM memory tiers.
        """
        if self._workloads is not None:
            return self._workloads

        # Full set of workloads implementing paper methodology
        self._workloads = [
            # Memory hierarchy workloads (paper Section 3.1.2)
            NVIDIACacheL1Workload(),
            NVIDIACacheL2Workload(),
            NVIDIAHBMWorkload(),
            # Compute workloads
            NVIDIAComputeWorkload(),
            NVIDIAIntegerComputeWorkload(),
            # GEMM workload (tensor core utilization)
            NVIDIAGEMMWorkload(),
            # Inference-level workloads (energy scaling laws)
            NVIDIAInferenceGEMMWorkload(),
            NVIDIAAttentionWorkload(),
            NVIDIAKVCacheWorkload(),
            NVIDIANCCLCollectiveWorkload(),
            NVIDIABatchedDecodeWorkload(),
        ]

        return self._workloads

    def get_l1_cache_workload(self) -> MemoryWorkload:
        """Return the L1 cache bandwidth workload."""
        return NVIDIACacheL1Workload()

    def get_l2_cache_workload(self) -> MemoryWorkload:
        """Return the L2 cache bandwidth workload."""
        return NVIDIACacheL2Workload()

    def get_hbm_workload(self) -> MemoryWorkload:
        """Return the HBM bandwidth workload."""
        return NVIDIAHBMWorkload()

    def get_integer_compute_workload(self) -> ComputeWorkload:
        """Return the INT32 compute workload for this platform."""
        return NVIDIAIntegerComputeWorkload()


__all__ = [
    "NVIDIACacheL1Workload",
    "NVIDIACacheL2Workload",
    "NVIDIAHBMWorkload",
    "NVIDIAComputeWorkload",
    "NVIDIAIntegerComputeWorkload",
    "NVIDIAGEMMWorkload",
    "NVIDIAInferenceGEMMWorkload",
    "NVIDIAAttentionWorkload",
    "NVIDIAKVCacheWorkload",
    "NVIDIANCCLCollectiveWorkload",
    "NVIDIABatchedDecodeWorkload",
    "NVIDIABenchmarkSuite",
]
