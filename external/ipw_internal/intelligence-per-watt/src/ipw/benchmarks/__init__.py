"""Energy characterization benchmarks for intelligence-per-watt.

This module provides a framework for running microbenchmarks to extract
energy parameters (pJ/bit, pJ/FLOP) for different hardware platforms.

The approach is based on:
"Benchmark-driven Models for Energy Analysis and Attribution of
GPU-Accelerated Supercomputing" (SC '25, Antepara et al.)

Key concepts:
- Run workloads with zeros (control energy) vs random data (control + datapath)
- Vary arithmetic intensity to separate memory and compute energy
- Extract energy parameters via linear regression

Example usage:
    from ipw.benchmarks import BenchmarkSuite
    from ipw.core.registry import BenchmarkRegistry

    # Auto-detect platform and get suite
    for name, suite_cls in BenchmarkRegistry.items():
        if suite_cls.is_available():
            suite = suite_cls()
            break

    # Run workloads
    workloads = suite.get_workloads()
    for workload in workloads:
        result = workload.run(config)
"""

from ipw.benchmarks.base import BenchmarkSuite
from ipw.benchmarks.types import (
    BenchmarkResult,
    DataType,
    EnergyMeasurement,
    EnergyParameters,
    Platform,
    WorkloadConfig,
    WorkloadResult,
    WorkloadType,
)
from ipw.benchmarks.runner import (
    BenchmarkRunner,
    CharacterizationConfig,
    InferenceCharacterizationConfig,
)
from ipw.benchmarks.analysis import extract_parameters, predict_power, summarize_results

# Import platform implementations to register them
from ipw.benchmarks.platforms import macos as _macos_impl  # noqa: F401
from ipw.benchmarks.platforms import nvidia as _nvidia_impl  # noqa: F401

__all__ = [
    # Base classes
    "BenchmarkSuite",
    # Runner
    "BenchmarkRunner",
    "CharacterizationConfig",
    "InferenceCharacterizationConfig",
    # Analysis
    "extract_parameters",
    "predict_power",
    "summarize_results",
    # Types
    "Platform",
    "DataType",
    "WorkloadType",
    "WorkloadConfig",
    "WorkloadResult",
    "EnergyMeasurement",
    "BenchmarkResult",
    "EnergyParameters",
]
