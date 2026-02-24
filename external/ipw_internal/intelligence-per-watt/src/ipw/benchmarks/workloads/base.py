"""Abstract base classes for benchmark workloads.

This module defines the interfaces that platform-specific workload
implementations must follow. Each workload type measures a specific
aspect of hardware energy consumption.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ipw.benchmarks.types import DataType, WorkloadConfig, WorkloadResult


class Workload(ABC):
    """Abstract base class for benchmark workloads.

    Each workload measures a specific aspect of the hardware:
    - Memory bandwidth (streaming access patterns)
    - Compute throughput (arithmetic intensity sweep)
    - GEMM performance (matrix multiplication)

    Platform-specific implementations inherit from the specialized
    subclasses (MemoryWorkload, ComputeWorkload, GEMMWorkload) and
    implement the run() method using platform-appropriate APIs.
    """

    workload_name: str  # Human-readable name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this workload can run on the current platform.

        Returns:
            True if the required dependencies and hardware are available.
        """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute the workload and return throughput metrics.

        Args:
            config: Workload configuration including duration, data type,
                    use_zeros flag, and workload-specific parameters.

        Returns:
            WorkloadResult with throughput and operation counts.
        """

    @abstractmethod
    def supported_data_types(self) -> List[DataType]:
        """Return list of data types this workload supports.

        Returns:
            List of DataType enums that can be used with this workload.
        """

    def warmup(self, config: WorkloadConfig) -> None:
        """Optional warmup before measurement.

        Override this method to perform any necessary warmup operations
        (e.g., JIT compilation, memory allocation) before the timed
        measurement begins.

        Args:
            config: Same configuration that will be used for run().
        """
        pass

    def validate_config(self, config: WorkloadConfig) -> None:
        """Validate workload configuration.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If configuration is invalid for this workload.
        """
        if config.data_type not in self.supported_data_types():
            supported = [dt.value for dt in self.supported_data_types()]
            raise ValueError(
                f"Data type {config.data_type.value} not supported by {self.workload_name}. "
                f"Supported types: {supported}"
            )


class MemoryWorkload(Workload):
    """Abstract base class for memory bandwidth workloads.

    Memory workloads measure the energy cost of data movement through
    the memory hierarchy. By varying array sizes, different cache levels
    can be targeted:
    - Small arrays (< L1 size): L1 cache energy
    - Medium arrays (< L2 size): L2 cache energy
    - Large arrays (>> L2 size): Main memory / unified memory energy

    The use_zeros flag enables differentiation of control plane energy
    (instruction fetch, decode, address calculation) from datapath energy
    (actual data movement).

    Expected config.params:
        array_size_mb: Size of array to stream through in megabytes.
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Stream through memory arrays, measuring bandwidth.

        Implementations should:
        1. Allocate an array of size config.params["array_size_mb"]
        2. Initialize with zeros or random data based on config.use_zeros
        3. Repeatedly stream through the array (read + write operations)
        4. Track total bytes transferred and elapsed time

        Args:
            config: Must include params["array_size_mb"].

        Returns:
            WorkloadResult with:
            - throughput: Bandwidth in GB/s
            - throughput_unit: "GB/s"
            - bytes_transferred: Total bytes read + written
        """


class ComputeWorkload(Workload):
    """Abstract base class for compute-bound workloads (Mixbench-style).

    Compute workloads measure the energy cost of arithmetic operations
    at varying arithmetic intensities (FLOPs per byte loaded). This allows
    separation of memory-bound and compute-bound energy consumption.

    At low arithmetic intensity: Memory energy dominates
    At high arithmetic intensity: Compute energy dominates

    By running at both extremes and solving a linear system (as described
    in the SC'25 paper), we can extract separate energy parameters for
    memory and compute.

    Expected config.params:
        arithmetic_intensity: Number of FLOPs per byte loaded (1 to 128+)
        array_size_mb: Working set size in megabytes
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute compute at varying arithmetic intensity.

        Implementations should:
        1. Allocate working array of size config.params["array_size_mb"]
        2. Initialize with zeros or random data based on config.use_zeros
        3. For each array element, perform arithmetic_intensity FLOPs
           (typically via polynomial evaluation: y = ((x*c+c)*c+c)...)
        4. Use bounded constants to prevent overflow/underflow with random data
        5. Track total FLOPs and elapsed time

        Args:
            config: Must include params["arithmetic_intensity"] and
                    params["array_size_mb"].

        Returns:
            WorkloadResult with:
            - throughput: Compute rate in TFLOP/s
            - throughput_unit: "TFLOP/s"
            - flops_executed: Total floating-point operations
        """


class GEMMWorkload(Workload):
    """Abstract base class for matrix multiplication workloads.

    GEMM (General Matrix Multiply) workloads measure the energy efficiency
    of matrix operations, which are fundamental to ML/AI workloads.
    On modern hardware, these operations often use specialized units
    (tensor cores, matrix engines, AMX) that have different energy
    characteristics than general vector FPUs.

    Expected config.params:
        matrix_size: Dimension M=N=K for square matrix multiplication
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute matrix multiplication (C = A @ B).

        Implementations should:
        1. Allocate square matrices A and B of size config.params["matrix_size"]
        2. Initialize with zeros or random data based on config.use_zeros
        3. Perform matrix multiplication using optimized libraries
           (cuBLAS, MPS, Accelerate, hipBLAS, etc.)
        4. Synchronize to ensure computation is complete
        5. Track total FLOPs (2 * M * N * K for standard GEMM)

        Args:
            config: Must include params["matrix_size"].

        Returns:
            WorkloadResult with:
            - throughput: Compute rate in TFLOP/s
            - throughput_unit: "TFLOP/s"
            - flops_executed: Total floating-point operations (2*M*N*K)
        """


class InferenceGEMMWorkload(GEMMWorkload):
    """Abstract base class for inference-shaped rectangular GEMM workloads.

    Measures energy of non-square matrix multiplications typical of LLM inference:
    - Prefill mode: tall-skinny GEMM, shape (B*S, d) × (d, d_ff)
    - Decode mode: flat GEMM, shape (B, d) × (d, d_ff)

    Expected config.params:
        batch_size: Number of sequences in the batch.
        seq_len: Sequence length (tokens).
        hidden_dim: Model hidden dimension d.
        ff_dim: Feed-forward dimension d_ff.
        mode: "prefill" or "decode".
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute inference-shaped GEMM.

        Args:
            config: Must include params with batch_size, seq_len, hidden_dim,
                    ff_dim, and mode ("prefill" or "decode").

        Returns:
            WorkloadResult with compute rate in TFLOP/s.
        """


class AttentionWorkload(Workload):
    """Abstract base class for scaled dot-product attention workloads.

    Measures energy of the attention mechanism:
    QK^T matmul + softmax + AV matmul.
    FLOPs ≈ 4 * B * H * S² * d_h

    Expected config.params:
        batch_size: Number of sequences.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute scaled dot-product attention.

        Args:
            config: Must include params with batch_size, seq_len,
                    num_heads, and head_dim.

        Returns:
            WorkloadResult with compute rate in TFLOP/s.
        """


class KVCacheWorkload(MemoryWorkload):
    """Abstract base class for KV cache I/O workloads.

    Measures energy of KV cache access patterns:
    - Write mode: sequential append (prefill pattern)
    - Read mode: strided gather (decode pattern)

    Expected config.params:
        cache_entries: Number of cached token positions.
        num_heads: Number of KV heads.
        head_dim: Dimension per head.
        batch_size: Batch size for gather indices.
        mode: "read" or "write".
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute KV cache read or write pattern.

        Args:
            config: Must include params with cache_entries, num_heads,
                    head_dim, batch_size, and mode ("read" or "write").

        Returns:
            WorkloadResult with bandwidth in GB/s and bytes_transferred.
        """


class NCCLCollectiveWorkload(Workload):
    """Abstract base class for collective communication workloads.

    Measures energy of multi-GPU communication operations
    (all-reduce, all-gather) used in tensor/pipeline parallelism.

    Expected config.params:
        message_size_mb: Message size in megabytes.
        collective_type: "all_reduce" or "all_gather".
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute collective communication.

        Args:
            config: Must include params with message_size_mb and
                    collective_type.

        Returns:
            WorkloadResult with bandwidth in GB/s and bytes_transferred.
        """


class BatchedDecodeWorkload(Workload):
    """Abstract base class for batched decode workloads.

    Measures how decode energy scales with batch size to extract the
    batch exponent β in E_decode ∝ B^β.

    Expected config.params:
        batch_size: Number of sequences decoded in parallel.
        hidden_dim: Model hidden dimension.
        ff_dim: Feed-forward dimension.
        num_layers: Number of transformer layers.
    """

    @abstractmethod
    def run(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute batched decode iterations.

        Args:
            config: Must include params with batch_size, hidden_dim,
                    ff_dim, and num_layers.

        Returns:
            WorkloadResult with throughput in tokens/s and flops_executed.
        """


__all__ = [
    "Workload",
    "MemoryWorkload",
    "ComputeWorkload",
    "GEMMWorkload",
    "InferenceGEMMWorkload",
    "AttentionWorkload",
    "KVCacheWorkload",
    "NCCLCollectiveWorkload",
    "BatchedDecodeWorkload",
]
