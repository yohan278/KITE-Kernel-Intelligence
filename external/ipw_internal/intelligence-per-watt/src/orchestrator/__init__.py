"""Orchestrator for energy-aware AI inference and training.

This module provides:

- data: Trajectory generation and dataset loading (shared)
- training: Training pipeline (sft, rl, utils, configs)
- inference: Inference policy and executor
- evals: Evaluation benchmarks
"""

from orchestrator import data
from orchestrator import training
from orchestrator import inference

__all__ = [
    "data",
    "training",
    "inference",
]
