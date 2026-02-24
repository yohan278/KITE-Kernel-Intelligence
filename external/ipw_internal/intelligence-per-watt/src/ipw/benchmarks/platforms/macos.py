"""macOS (Apple Silicon) benchmark suite implementation.

This module provides concrete workload implementations for Apple Silicon Macs
using PyTorch MPS for GPU workloads. All workloads run on the GPU via Metal
Performance Shaders to measure unified memory and GPU compute energy.

Based on the SC'25 paper methodology (Antepara et al.) for energy characterization.
"""

from __future__ import annotations

import platform
import subprocess
import time
import warnings
from typing import List, Optional

import numpy as np

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
    ComputeWorkload,
    GEMMWorkload,
    InferenceGEMMWorkload,
    MemoryWorkload,
)
from ipw.core.registry import BenchmarkRegistry


def _check_mps_available() -> bool:
    """Check if PyTorch MPS is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


class MacOSMemoryWorkload(MemoryWorkload):
    """Memory bandwidth workload using PyTorch MPS.

    Streams through GPU tensors measuring unified memory bandwidth.
    Uses simple add operations to minimize compute and maximize
    memory throughput measurement.
    """

    workload_name = "MPS Memory Streaming"

    def is_available(self) -> bool:
        """Check if MPS is available."""
        return _check_mps_available()

    def supported_data_types(self) -> List[DataType]:
        """MPS supports FP32 and FP16."""
        return [DataType.FP32, DataType.FP16]

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup MPS by running a few memory streaming iterations."""
        import torch

        array_size_mb = config.params.get("array_size_mb", 100)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        element_size = 4 if dtype == torch.float32 else 2
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        arr = torch.zeros(n_elements, dtype=dtype, device=device)
        for _ in range(5):
            arr.add_(0.0)
        torch.mps.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Stream through GPU memory, measuring bandwidth.

        Args:
            config: Must include params["array_size_mb"].

        Returns:
            WorkloadResult with bandwidth in GB/s.
        """
        import torch

        self.validate_config(config)

        array_size_mb = config.params.get("array_size_mb", 100)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        # Calculate array size
        element_size = 4 if dtype == torch.float32 else 2
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        # Initialize tensor - zeros for control energy, random for full energy
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=dtype, device=device)
            const = 0.0
        else:
            # Use bounded random values (paper: 1.0 to 2.0, const = 0.5)
            arr = torch.empty(n_elements, dtype=dtype, device=device).uniform_(1.0, 2.0)
            const = 0.5

        bytes_transferred = 0
        iterations = 0
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            # In-place add: read + write = 2x array size in bytes transferred
            arr.add_(const)
            bytes_transferred += 2 * n_elements * element_size
            iterations += 1
            # Periodic sync to prevent command buffer buildup
            if iterations % 100 == 0:
                torch.mps.synchronize()

        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        bandwidth_gb_s = bytes_transferred / elapsed / 1e9

        return WorkloadResult(
            workload_type=WorkloadType.MEMORY_BANDWIDTH,
            config=config,
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=elapsed,
        )

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


class MacOSComputeWorkload(ComputeWorkload):
    """Mixbench-style compute workload using PyTorch MPS.

    Varies arithmetic intensity (FLOPs per byte) to enable separation
    of memory-bound and compute-bound energy consumption. Based on
    the SC'25 paper methodology.

    At low arithmetic intensity: memory-bound, measures ε_mem
    At high arithmetic intensity: compute-bound, measures ε_compute
    """

    workload_name = "MPS Compute Sweep"

    def is_available(self) -> bool:
        """Check if MPS is available."""
        return _check_mps_available()

    def supported_data_types(self) -> List[DataType]:
        """MPS supports FP32, FP16, and BF16 (M2+)."""
        import torch
        supported = [DataType.FP32, DataType.FP16]
        # BF16 available on M2 and later
        if hasattr(torch, 'bfloat16'):
            try:
                # Test if MPS actually supports BF16
                test = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                supported.append(DataType.BF16)
            except Exception:
                pass
        return supported

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup MPS by compiling shaders with a few compute iterations."""
        import torch

        array_size_mb = config.params.get("array_size_mb", 100)
        arithmetic_intensity = config.params.get("arithmetic_intensity", 16)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        element_size = 4 if dtype == torch.float32 else 2
        n_elements = (array_size_mb * 1024 * 1024) // element_size

        arr = torch.zeros(n_elements, dtype=dtype, device=device)
        warmup_result = arr.clone()
        for _ in range(min(arithmetic_intensity, 4)):
            warmup_result.mul_(0.0).add_(0.0)
        torch.mps.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute compute at varying arithmetic intensity.

        Performs polynomial evaluation: y = ((x*c+c)*c+c)...
        Each multiply-add is 2 FLOPs per element.

        Arithmetic Intensity = (2 * AI_param * n_elements) / (element_size * n_elements)
                            = 2 * AI_param / element_size [FLOP/byte]

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
        device = torch.device("mps")

        # Calculate array size
        element_size = 4 if dtype == torch.float32 else 2
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
        # Pre-allocate result buffer to avoid allocation each iteration
        result = arr.clone()
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
            # Periodic sync to prevent command buffer buildup
            if iterations % 50 == 0:
                torch.mps.synchronize()

        torch.mps.synchronize()
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


class MacOSIntegerComputeWorkload(ComputeWorkload):
    """INT32 compute workload using PyTorch MPS.

    Measures integer operation energy (IMAD - integer multiply-add).
    Important because ~10-20% of application power goes to integer ops
    (loop counters, address calculations, array indexing).

    Based on the SC'25 paper methodology which includes INT32 energy
    parameters alongside floating-point measurements.
    """

    workload_name = "MPS INT32 Compute"

    def is_available(self) -> bool:
        """Check if MPS is available."""
        return _check_mps_available()

    def supported_data_types(self) -> List[DataType]:
        """Only supports INT32."""
        return [DataType.INT32]

    def warmup(self, config: WorkloadConfig) -> None:
        """Warmup MPS by running a few integer compute iterations."""
        import torch

        array_size_mb = config.params.get("array_size_mb", 100)
        device = torch.device("mps")

        # INT32 = 4 bytes per element
        n_elements = (array_size_mb * 1024 * 1024) // 4

        arr = torch.zeros(n_elements, dtype=torch.int32, device=device)
        warmup_result = arr.clone()
        for _ in range(4):
            warmup_result.mul_(1).add_(1)
        torch.mps.synchronize()

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
        device = torch.device("mps")

        # INT32 = 4 bytes per element
        n_elements = (array_size_mb * 1024 * 1024) // 4

        # Initialize tensor
        if config.use_zeros:
            arr = torch.zeros(n_elements, dtype=torch.int32, device=device)
            const = 0
        else:
            # Random integers in bounded range to prevent overflow
            arr = torch.randint(1, 100, (n_elements,), dtype=torch.int32, device=device)
            # Use const that keeps values bounded: multiply by 1, add small value
            # This mimics address calculation patterns (base + offset)
            const = 1

        total_ops = 0
        iterations = 0
        result = arr.clone()
        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            result.copy_(arr)
            for _ in range(arithmetic_intensity):
                # Integer multiply-add: 2 OPs per element
                # Using bitwise operations for more realistic workload
                result.mul_(const).add_(const)
            total_ops += n_elements * arithmetic_intensity * 2
            iterations += 1
            if iterations % 50 == 0:
                torch.mps.synchronize()

        torch.mps.synchronize()
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


class MacOSGEMMWorkload(GEMMWorkload):
    """GEMM workload using PyTorch MPS backend.

    Performs matrix multiplication on Apple's Metal Performance Shaders
    GPU backend, measuring the energy efficiency of matrix operations.
    """

    workload_name = "PyTorch MPS GEMM"

    def is_available(self) -> bool:
        """Check if PyTorch MPS backend is available."""
        return _check_mps_available()

    def supported_data_types(self) -> List[DataType]:
        """MPS supports FP32, FP16, and BF16 (M2+)."""
        import torch
        supported = [DataType.FP32, DataType.FP16]
        # BF16 available on M2 and later
        if hasattr(torch, 'bfloat16'):
            try:
                test = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                supported.append(DataType.BF16)
            except Exception:
                pass
        return supported

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute matrix multiplication on MPS.

        Args:
            config: Must include params["matrix_size"].

        Returns:
            WorkloadResult with compute rate in TFLOP/s.
        """
        import torch

        self.validate_config(config)

        matrix_size = config.params.get("matrix_size", 2048)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        # Initialize matrices
        if config.use_zeros:
            A = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
            B = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
        else:
            A = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device)
            B = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device)

        # GEMM: C = A @ B
        # FLOPs = 2 * M * N * K (multiply-accumulate for each element)
        flops_per_mm = 2 * matrix_size * matrix_size * matrix_size
        total_flops = 0
        iterations = 0

        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            _ = torch.mm(A, B)
            total_flops += flops_per_mm
            iterations += 1
            # Periodic sync to prevent command buffer buildup
            if iterations % 20 == 0:
                torch.mps.synchronize()

        # Ensure all GPU work is complete
        torch.mps.synchronize()
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
        """Warmup MPS by running a few iterations."""
        import torch

        matrix_size = config.params.get("matrix_size", 2048)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        A = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device)
        B = torch.rand(matrix_size, matrix_size, dtype=dtype, device=device)

        for _ in range(5):
            _ = torch.mm(A, B)
        torch.mps.synchronize()

    def _to_torch_dtype(self, dt: DataType):
        """Convert DataType enum to torch dtype."""
        import torch

        mapping = {
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


class MacOSCPUGEMMWorkload(GEMMWorkload):
    """CPU GEMM workload using NumPy (backed by Accelerate/AMX).

    On Apple Silicon, NumPy's BLAS operations use the Accelerate framework
    which leverages the AMX (Apple Matrix Extensions) coprocessor for
    matrix operations.
    """

    workload_name = "NumPy CPU GEMM (Accelerate)"

    def is_available(self) -> bool:
        """NumPy is always available."""
        return True

    def supported_data_types(self) -> List[DataType]:
        """Support FP64 and FP32."""
        return [DataType.FP64, DataType.FP32]

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute matrix multiplication on CPU via Accelerate.

        Args:
            config: Must include params["matrix_size"].

        Returns:
            WorkloadResult with compute rate in TFLOP/s.
        """
        import warnings

        self.validate_config(config)

        matrix_size = config.params.get("matrix_size", 2048)
        dtype = self._to_numpy_dtype(config.data_type)

        # Initialize matrices with small values to prevent overflow
        # For energy benchmarking, numerical accuracy isn't important
        if config.use_zeros:
            A = np.zeros((matrix_size, matrix_size), dtype=dtype)
            B = np.zeros((matrix_size, matrix_size), dtype=dtype)
        else:
            # Scale by 1/sqrt(N) to keep output magnitude bounded
            scale = dtype(1.0 / np.sqrt(matrix_size))
            A = (np.random.rand(matrix_size, matrix_size) * scale).astype(dtype)
            B = (np.random.rand(matrix_size, matrix_size) * scale).astype(dtype)

        # Pre-allocate output to avoid allocation overhead in loop
        C = np.empty((matrix_size, matrix_size), dtype=dtype)

        # Suppress overflow warnings - we care about energy, not numerical accuracy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Warmup
            for _ in range(3):
                np.matmul(A, B, out=C)

            flops_per_mm = 2 * matrix_size * matrix_size * matrix_size
            total_flops = 0
            iterations = 0

            start = time.perf_counter()
            deadline = start + config.duration_seconds

            while time.perf_counter() < deadline:
                np.matmul(A, B, out=C)
                total_flops += flops_per_mm
                iterations += 1

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

    def _to_numpy_dtype(self, dt: DataType) -> type:
        """Convert DataType enum to numpy dtype."""
        mapping = {
            DataType.FP64: np.float64,
            DataType.FP32: np.float32,
        }
        return mapping[dt]


class MacOSInferenceGEMMWorkload(InferenceGEMMWorkload):
    """Inference-shaped rectangular GEMM workload using PyTorch MPS.

    Same logic as NVIDIA version but uses MPS backend.
    """

    workload_name = "MPS Inference GEMM"

    def is_available(self) -> bool:
        return _check_mps_available()

    def supported_data_types(self) -> List[DataType]:
        return [DataType.FP32, DataType.FP16]

    def validate_config(self, config: WorkloadConfig) -> None:
        super().validate_config(config)
        if config.use_cuda_graphs:
            warnings.warn(
                "CUDA graphs not supported on MPS. Running in standard mode.",
                stacklevel=2,
            )

    def warmup(self, config: WorkloadConfig) -> None:
        import torch

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        hidden_dim = config.params.get("hidden_dim", 4096)
        ff_dim = config.params.get("ff_dim", 11008)
        mode = config.params.get("mode", "prefill")
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        M = batch_size * seq_len if mode == "prefill" else batch_size
        A = torch.randn(M, hidden_dim, dtype=dtype, device=device)
        B = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)
        for _ in range(5):
            torch.mm(A, B)
        torch.mps.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch

        self.validate_config(config)

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        hidden_dim = config.params.get("hidden_dim", 4096)
        ff_dim = config.params.get("ff_dim", 11008)
        mode = config.params.get("mode", "prefill")
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        M = batch_size * seq_len if mode == "prefill" else batch_size
        flops_per_mm = 2 * M * hidden_dim * ff_dim

        A = torch.randn(M, hidden_dim, dtype=dtype, device=device)
        W = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)

        total_flops = 0
        iterations = 0

        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            torch.mm(A, W)
            total_flops += flops_per_mm
            iterations += 1
            if iterations % 20 == 0:
                torch.mps.synchronize()

        torch.mps.synchronize()
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
        mapping = {DataType.FP32: torch.float32, DataType.FP16: torch.float16}
        if dt not in mapping:
            raise ValueError(f"Unsupported data type {dt.value} for {self.workload_name}")
        return mapping[dt]


class MacOSAttentionWorkload(AttentionWorkload):
    """Scaled dot-product attention workload using PyTorch MPS.

    Uses F.scaled_dot_product_attention on MPS backend.
    FlashAttention is not available on MPS.
    """

    workload_name = "MPS Attention"

    def is_available(self) -> bool:
        return _check_mps_available()

    def supported_data_types(self) -> List[DataType]:
        return [DataType.FP32, DataType.FP16]

    def validate_config(self, config: WorkloadConfig) -> None:
        super().validate_config(config)
        if config.use_cuda_graphs:
            warnings.warn(
                "CUDA graphs not supported on MPS. Running in standard mode.",
                stacklevel=2,
            )

    def warmup(self, config: WorkloadConfig) -> None:
        import torch

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        num_heads = config.params.get("num_heads", 32)
        head_dim = config.params.get("head_dim", 128)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.mps.synchronize()

    def run(self, config: WorkloadConfig) -> WorkloadResult:
        import torch
        import torch.nn.functional as F

        self.validate_config(config)

        batch_size = config.params.get("batch_size", 1)
        seq_len = config.params.get("seq_len", 512)
        num_heads = config.params.get("num_heads", 32)
        head_dim = config.params.get("head_dim", 128)
        dtype = self._to_torch_dtype(config.data_type)
        device = torch.device("mps")

        flops_per_attn = 4 * batch_size * num_heads * seq_len * seq_len * head_dim

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        config.params["uses_flash_attention"] = False

        total_flops = 0
        iterations = 0

        start = time.perf_counter()
        deadline = start + config.duration_seconds

        while time.perf_counter() < deadline:
            F.scaled_dot_product_attention(Q, K, V)
            total_flops += flops_per_attn
            iterations += 1
            if iterations % 10 == 0:
                torch.mps.synchronize()

        torch.mps.synchronize()
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


@BenchmarkRegistry.register("macos")
class MacOSBenchmarkSuite(BenchmarkSuite):
    """Benchmark suite for Apple Silicon Macs.

    All workloads run on the GPU via PyTorch MPS (Metal Performance Shaders).
    This enables accurate energy characterization of the unified memory
    architecture using powermetrics for CPU+GPU power measurement.

    Implements the SC'25 paper methodology (Antepara et al.) adapted for
    Apple Silicon's unified memory - no L1/L2/HBM separation needed.
    """

    platform = Platform.MACOS
    platform_name = "Apple Silicon"

    @classmethod
    def is_available(cls) -> bool:
        """Check if running on macOS with MPS support."""
        if platform.system() != "Darwin":
            return False
        return _check_mps_available()

    @classmethod
    def detect_hardware(cls) -> str:
        """Detect Apple Silicon chip name.

        Returns:
            Chip name like "Apple M2 Pro" or fallback string.
        """
        try:
            # Try to get chip name from sysctl
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            # Fallback: try system_profiler
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Chip:" in line:
                        return line.split("Chip:")[1].strip()

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return "Apple Silicon (unknown)"

    def __init__(self) -> None:
        """Initialize the macOS benchmark suite."""
        self._workloads: Optional[List] = None

    def get_workloads(self) -> List:
        """Return available workloads for macOS.

        Returns:
            List of MPS workload instances for energy characterization.
            All workloads run on GPU via Metal Performance Shaders.
        """
        if self._workloads is not None:
            return self._workloads

        # MPS workloads for GPU-based energy measurement
        # CPU GEMM available as alternative using Accelerate/AMX
        self._workloads = [
            MacOSMemoryWorkload(),
            MacOSComputeWorkload(),
            MacOSIntegerComputeWorkload(),
            MacOSGEMMWorkload(),
            MacOSCPUGEMMWorkload(),
            # Inference-level workloads (energy scaling laws)
            MacOSInferenceGEMMWorkload(),
            MacOSAttentionWorkload(),
        ]

        return self._workloads

    def get_integer_compute_workload(self):
        """Return the INT32 compute workload for this platform."""
        return MacOSIntegerComputeWorkload()


__all__ = [
    "MacOSMemoryWorkload",
    "MacOSComputeWorkload",
    "MacOSIntegerComputeWorkload",
    "MacOSGEMMWorkload",
    "MacOSCPUGEMMWorkload",
    "MacOSInferenceGEMMWorkload",
    "MacOSAttentionWorkload",
    "MacOSBenchmarkSuite",
]
