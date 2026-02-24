"""Training pipeline for the orchestrator.

This module provides training-specific infrastructure:

- sft: Supervised fine-tuning trainer
- rl: Reinforcement learning (reward, environment, policy)
- utils: Training utilities (FSDP, tokenizer)
- configs: Training configuration files
"""

from orchestrator.training import sft
from orchestrator.training import rl
from orchestrator.training import utils

__all__ = [
    "sft",
    "rl",
    "utils",
]
