# benchmarks/swefficiency/__init__.py
"""
SWEfficiency Benchmark.

Evaluates AI agents on software performance optimization tasks by:
1. Loading tasks from the SWEfficiency dataset
2. Running agents to generate optimization patches
3. Evaluating patches using Docker containers (similar to SWE-bench)

Uses the swefficiency/swefficiency dataset from HuggingFace.

Usage:
    from evals.benchmarks.swefficiency import SWEfficiencyBenchmark, SWEfficiencyConfig

    config = SWEfficiencyConfig(limit=10)
    benchmark = SWEfficiencyBenchmark(config)
    results = benchmark.run_benchmark(orchestrator)

Or via registry:
    from evals.registry import get_benchmark

    benchmark = get_benchmark("swefficiency")(options={"limit": 10})
"""

from evals.benchmarks.swefficiency.main import (
    SWEfficiencyBenchmark,
    SWEfficiencyConfig,
    SWEfficiencyResult,
)
from evals.benchmarks.swefficiency.dataset import (
    SWEfficiencySample,
    load_swefficiency_samples,
    get_swefficiency_repos,
    get_sample_count,
    DEFAULT_INPUT_PROMPT,
)
from evals.benchmarks.swefficiency.env_wrapper import (
    SWEfficiencyEnv,
    BenchmarkResult,
    TestResult,
)

__all__ = [
    # Main benchmark
    "SWEfficiencyBenchmark",
    "SWEfficiencyConfig",
    "SWEfficiencyResult",
    # Dataset
    "SWEfficiencySample",
    "load_swefficiency_samples",
    "get_swefficiency_repos",
    "get_sample_count",
    "DEFAULT_INPUT_PROMPT",
    # Environment
    "SWEfficiencyEnv",
    "BenchmarkResult",
    "TestResult",
]
