"""Abstract base class for platform-specific benchmark suites.

This module defines the BenchmarkSuite interface that platform-specific
implementations (macOS, NVIDIA, ROCm) must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from ipw.benchmarks.types import Platform

if TYPE_CHECKING:
    from ipw.benchmarks.workloads.base import (
        AttentionWorkload,
        BatchedDecodeWorkload,
        ComputeWorkload,
        GEMMWorkload,
        InferenceGEMMWorkload,
        KVCacheWorkload,
        MemoryWorkload,
        NCCLCollectiveWorkload,
        Workload,
    )


class BenchmarkSuite(ABC):
    """Abstract base class for platform-specific benchmark suites.

    Each platform (macOS, NVIDIA, ROCm) implements this interface to provide:
    - Platform detection and availability checking
    - Hardware identification
    - Platform-specific workload implementations

    Implementations should be registered with BenchmarkRegistry to enable
    automatic platform detection and selection.

    Example:
        @BenchmarkRegistry.register("macos")
        class MacOSBenchmarkSuite(BenchmarkSuite):
            platform = Platform.MACOS
            platform_name = "Apple Silicon"
            ...
    """

    platform: Platform
    platform_name: str  # e.g., "Apple Silicon", "NVIDIA CUDA", "AMD ROCm"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this platform is available on the current system.

        Returns:
            True if the platform's hardware and software dependencies
            are available and functional.
        """

    @classmethod
    @abstractmethod
    def detect_hardware(cls) -> str:
        """Return hardware identifier for this platform.

        Returns:
            Human-readable hardware name, e.g., "Apple M2 Pro",
            "NVIDIA A100", "AMD MI250X".
        """

    @abstractmethod
    def get_workloads(self) -> List[Workload]:
        """Return list of workloads available for this platform.

        Returns:
            List of Workload instances that can be run on this platform.
            Should include at minimum a MemoryWorkload and ComputeWorkload.
        """

    def get_memory_workload(self) -> MemoryWorkload:
        """Get the memory bandwidth workload for this platform.

        Returns:
            MemoryWorkload instance.

        Raises:
            NotImplementedError: If no memory workload is available.
        """
        from ipw.benchmarks.workloads.base import MemoryWorkload

        for w in self.get_workloads():
            if isinstance(w, MemoryWorkload):
                return w
        raise NotImplementedError(f"No memory workload for {self.platform_name}")

    def get_compute_workload(self) -> ComputeWorkload:
        """Get the compute workload for this platform.

        Returns:
            ComputeWorkload instance.

        Raises:
            NotImplementedError: If no compute workload is available.
        """
        from ipw.benchmarks.workloads.base import ComputeWorkload

        for w in self.get_workloads():
            if isinstance(w, ComputeWorkload):
                return w
        raise NotImplementedError(f"No compute workload for {self.platform_name}")

    def get_gemm_workload(self) -> GEMMWorkload:
        """Get the GEMM workload for this platform.

        Returns:
            GEMMWorkload instance.

        Raises:
            NotImplementedError: If no GEMM workload is available.
        """
        from ipw.benchmarks.workloads.base import GEMMWorkload

        for w in self.get_workloads():
            if isinstance(w, GEMMWorkload):
                return w
        raise NotImplementedError(f"No GEMM workload for {self.platform_name}")

    def get_integer_compute_workload(self) -> ComputeWorkload:
        """Get the INT32 compute workload for this platform.

        Returns a compute workload that supports INT32 operations for measuring
        integer ALU energy (important for address calculations, loop counters).

        Returns:
            ComputeWorkload instance that supports DataType.INT32.

        Raises:
            NotImplementedError: If no INT32 compute workload is available.
        """
        from ipw.benchmarks.types import DataType
        from ipw.benchmarks.workloads.base import ComputeWorkload

        for w in self.get_workloads():
            if isinstance(w, ComputeWorkload):
                if DataType.INT32 in w.supported_data_types():
                    return w
        raise NotImplementedError(f"No INT32 compute workload for {self.platform_name}")

    def get_inference_gemm_workload(self) -> InferenceGEMMWorkload:
        """Get the inference-shaped GEMM workload for this platform.

        Returns:
            InferenceGEMMWorkload instance.

        Raises:
            NotImplementedError: If no inference GEMM workload is available.
        """
        from ipw.benchmarks.workloads.base import InferenceGEMMWorkload

        for w in self.get_workloads():
            if isinstance(w, InferenceGEMMWorkload):
                return w
        raise NotImplementedError(
            f"No inference GEMM workload for {self.platform_name}"
        )

    def get_attention_workload(self) -> AttentionWorkload:
        """Get the attention workload for this platform.

        Returns:
            AttentionWorkload instance.

        Raises:
            NotImplementedError: If no attention workload is available.
        """
        from ipw.benchmarks.workloads.base import AttentionWorkload

        for w in self.get_workloads():
            if isinstance(w, AttentionWorkload):
                return w
        raise NotImplementedError(f"No attention workload for {self.platform_name}")

    def get_kv_cache_workload(self) -> KVCacheWorkload:
        """Get the KV cache I/O workload for this platform.

        Returns:
            KVCacheWorkload instance.

        Raises:
            NotImplementedError: If no KV cache workload is available.
        """
        from ipw.benchmarks.workloads.base import KVCacheWorkload

        for w in self.get_workloads():
            if isinstance(w, KVCacheWorkload):
                return w
        raise NotImplementedError(f"No KV cache workload for {self.platform_name}")

    def get_nccl_collective_workload(self) -> NCCLCollectiveWorkload:
        """Get the NCCL collective communication workload for this platform.

        Returns:
            NCCLCollectiveWorkload instance.

        Raises:
            NotImplementedError: If no NCCL collective workload is available.
        """
        from ipw.benchmarks.workloads.base import NCCLCollectiveWorkload

        for w in self.get_workloads():
            if isinstance(w, NCCLCollectiveWorkload):
                return w
        raise NotImplementedError(
            f"No NCCL collective workload for {self.platform_name}"
        )

    def get_batched_decode_workload(self) -> BatchedDecodeWorkload:
        """Get the batched decode workload for this platform.

        Returns:
            BatchedDecodeWorkload instance.

        Raises:
            NotImplementedError: If no batched decode workload is available.
        """
        from ipw.benchmarks.workloads.base import BatchedDecodeWorkload

        for w in self.get_workloads():
            if isinstance(w, BatchedDecodeWorkload):
                return w
        raise NotImplementedError(
            f"No batched decode workload for {self.platform_name}"
        )


__all__ = ["BenchmarkSuite"]
