"""Grid search evaluation runner for IPW benchmarking.

This module provides a grid search framework for evaluating model/agent
combinations across different benchmarks and hardware configurations.
"""

from grid_eval.config import (
    AgentType,
    BenchmarkType,
    GridConfig,
    HardwareConfig,
    ModelType,
)
from grid_eval.hardware import HardwareManager
from grid_eval.output import JSONLWriter
from grid_eval.runner import GridEvalRunner

__all__ = [
    "AgentType",
    "BenchmarkType",
    "GridConfig",
    "GridEvalRunner",
    "HardwareConfig",
    "HardwareManager",
    "JSONLWriter",
    "ModelType",
    "analyze_grid_results",
]


def __getattr__(name: str):
    """Lazy import for analyze_grid_results to avoid circular deps."""
    if name == "analyze_grid_results":
        from grid_eval.analyze import analyze_grid_results
        return analyze_grid_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
