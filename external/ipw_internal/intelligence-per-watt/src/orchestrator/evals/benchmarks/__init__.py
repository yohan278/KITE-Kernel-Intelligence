"""Benchmark implementations."""

from .registry import BenchmarkRegistry, register_benchmark

# Import benchmarks to register them
from .hle import HLERunner
from .gaia import GAIARunner
from .simpleqa import SimpleQARunner
from .deepresearch import DeepResearchRunner

__all__ = [
    "BenchmarkRegistry",
    "register_benchmark",
    "HLERunner",
    "GAIARunner",
    "SimpleQARunner",
    "DeepResearchRunner",
]
