"""Workload abstract base classes for energy benchmarks.

This module defines the interfaces for different types of benchmark workloads:
- MemoryWorkload: Measures memory bandwidth and data movement energy
- ComputeWorkload: Measures compute throughput at varying arithmetic intensity
- GEMMWorkload: Measures matrix multiplication performance and energy
- InferenceGEMMWorkload: Measures inference-shaped rectangular GEMM energy
- AttentionWorkload: Measures scaled dot-product attention energy
- KVCacheWorkload: Measures KV cache read/write energy
- NCCLCollectiveWorkload: Measures collective communication energy
- BatchedDecodeWorkload: Measures batched decode scaling
"""

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
