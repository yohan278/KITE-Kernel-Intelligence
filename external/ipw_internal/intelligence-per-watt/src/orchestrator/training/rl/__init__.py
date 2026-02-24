"""Reinforcement learning components for orchestrator training.

Provides:
- MultiObjectiveReward: Energy-aware reward function
- OrchestratorEnvironment: RL environment with telemetry
- PolicyModel: Policy network wrapper
"""

from orchestrator.training.rl.reward import (
    MultiObjectiveReward,
    RewardWeights,
    Normalizers,
    AdaptiveRewardWeights,
)
from orchestrator.training.rl.environment import (
    OrchestratorEnvironment,
    OrchestratorEnvironmentReal,
    EpisodeState,
)
from orchestrator.training.rl.policy import PolicyModel, PolicyOutput

__all__ = [
    # Reward
    "MultiObjectiveReward",
    "RewardWeights",
    "Normalizers",
    "AdaptiveRewardWeights",
    # Environment
    "OrchestratorEnvironment",
    "OrchestratorEnvironmentReal",
    "EpisodeState",
    # Policy
    "PolicyModel",
    "PolicyOutput",
]
