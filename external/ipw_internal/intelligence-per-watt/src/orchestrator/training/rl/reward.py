"""Multi-objective reward function for orchestrator training.

Balances:
- Accuracy (correctness of answer)
- Cost (dollar cost of API calls)
- Energy (joules consumed)
- Latency (seconds elapsed)
- Power (watts used)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from orchestrator.data.episode_builder import Episode


@dataclass
class RewardWeights:
    """Weights for multi-objective reward function.

    Each metric has its own coefficient for fine-grained control:
    - alpha: Accuracy (correctness of answer)
    - beta_cost: API/cloud cost in USD
    - beta_energy: Energy consumption in joules
    - gamma_latency: Response time in seconds
    - gamma_power: Peak power usage in watts
    """

    alpha: float = 0.4
    """Accuracy weight (most important)"""

    beta_cost: float = 0.15
    """Cost weight (API costs)"""

    beta_energy: float = 0.15
    """Energy weight (joules consumed)"""

    gamma_latency: float = 0.15
    """Latency weight (response time)"""

    gamma_power: float = 0.15
    """Max power weight (peak watts)"""

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.alpha + self.beta_cost + self.beta_energy + self.gamma_latency + self.gamma_power
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights should sum to 1.0, got {total}")


@dataclass
class Normalizers:
    """Normalization constants for reward function.

    These are typical values used to scale metrics to similar ranges.
    Should be tuned based on your specific tools and tasks.
    """

    energy_scale: float = 100.0
    """Typical energy consumption in joules (e.g., 100J for local LLM inference)"""

    cost_scale: float = 0.10
    """Typical cost in USD (e.g., $0.10 for GPT-4o query)"""

    latency_scale: float = 30.0
    """Typical latency in seconds (e.g., 30s for complex query)"""

    power_scale: float = 200.0
    """Typical power usage in watts (e.g., 200W GPU)"""


class MultiObjectiveReward:
    """Multi-objective reward function for orchestrator training.

    Combines accuracy, cost, energy, latency, and power into a single scalar reward.

    Example:
        weights = RewardWeights(alpha=0.5, beta=0.3, gamma=0.2)
        normalizers = Normalizers()
        reward_fn = MultiObjectiveReward(weights, normalizers)

        # Compute reward for episode
        reward = reward_fn.compute(episode)
        print(f"Reward: {reward:.4f}")
    """

    def __init__(
        self,
        weights: RewardWeights,
        normalizers: Normalizers,
    ):
        """Initialize reward function.

        Args:
            weights: Weights for each objective
            normalizers: Normalization constants
        """
        self.weights = weights
        self.normalizers = normalizers

    def compute(self, episode: Episode) -> float:
        """Compute multi-objective reward for an episode.

        Formula:
            reward = α * accuracy
                     - β_cost * normalized_cost
                     - β_energy * normalized_energy
                     - γ_latency * normalized_latency
                     - γ_power * normalized_power

        Args:
            episode: Completed episode with metrics

        Returns:
            Scalar reward (higher is better)
        """
        # 1. Accuracy reward (0 or 1)
        accuracy_reward = 1.0 if episode.correct else 0.0

        # 2. Normalized penalties (each metric separately)
        cost_penalty = episode.total_cost_usd / self.normalizers.cost_scale
        energy_penalty = episode.total_energy_joules / self.normalizers.energy_scale
        latency_penalty = episode.total_latency_seconds / self.normalizers.latency_scale
        power_penalty = episode.max_power_watts / self.normalizers.power_scale

        # Combine with separate weights for each metric
        reward = (
            self.weights.alpha * accuracy_reward
            - self.weights.beta_cost * cost_penalty
            - self.weights.beta_energy * energy_penalty
            - self.weights.gamma_latency * latency_penalty
            - self.weights.gamma_power * power_penalty
        )

        return reward

    def compute_with_breakdown(self, episode: Episode) -> Dict[str, float]:
        """Compute reward with detailed breakdown.

        Args:
            episode: Completed episode

        Returns:
            Dictionary with reward components and total
        """
        # Accuracy
        accuracy_reward = 1.0 if episode.correct else 0.0

        # Normalized penalties
        cost_penalty = episode.total_cost_usd / self.normalizers.cost_scale
        energy_penalty = episode.total_energy_joules / self.normalizers.energy_scale
        latency_penalty = episode.total_latency_seconds / self.normalizers.latency_scale
        power_penalty = episode.max_power_watts / self.normalizers.power_scale

        # Weighted components (each metric separate)
        accuracy_component = self.weights.alpha * accuracy_reward
        cost_component = -self.weights.beta_cost * cost_penalty
        energy_component = -self.weights.beta_energy * energy_penalty
        latency_component = -self.weights.gamma_latency * latency_penalty
        power_component = -self.weights.gamma_power * power_penalty

        # Total reward
        total_reward = (
            accuracy_component + cost_component + energy_component +
            latency_component + power_component
        )

        # Compute efficiency metrics
        ipj = episode.compute_ipj() if hasattr(episode, 'compute_ipj') else (
            accuracy_reward / episode.total_energy_joules if episode.total_energy_joules > 0 else 0.0
        )

        return {
            "total_reward": total_reward,
            "accuracy_reward": accuracy_reward,
            "accuracy_component": accuracy_component,
            "cost_penalty": cost_penalty,
            "cost_component": cost_component,
            "energy_penalty": energy_penalty,
            "energy_component": energy_component,
            "latency_penalty": latency_penalty,
            "latency_component": latency_component,
            "power_penalty": power_penalty,
            "power_component": power_component,
            # Efficiency metrics
            "ipj": ipj,
            "total_energy_joules": episode.total_energy_joules,
            "total_cost_usd": episode.total_cost_usd,
            "total_latency_seconds": episode.total_latency_seconds,
            "total_forward_passes": getattr(episode, 'total_forward_passes', 0),
        }

    def compute_batch(self, episodes: list[Episode]) -> list[float]:
        """Compute rewards for a batch of episodes.

        Args:
            episodes: List of episodes

        Returns:
            List of rewards
        """
        return [self.compute(episode) for episode in episodes]


class AdaptiveRewardWeights:
    """Adaptive reward weights that change during training.

    Early training: Focus on accuracy (higher alpha)
    Late training: Optimize cost/energy/power (higher other weights)
    """

    def __init__(
        self,
        initial_alpha: float = 0.6,
        final_alpha: float = 0.3,
        initial_beta_cost: float = 0.1,
        final_beta_cost: float = 0.15,
        initial_beta_energy: float = 0.1,
        final_beta_energy: float = 0.2,
        initial_gamma_latency: float = 0.1,
        final_gamma_latency: float = 0.15,
        initial_gamma_power: float = 0.1,
        final_gamma_power: float = 0.2,
        total_steps: int = 10000,
    ):
        """Initialize adaptive weights.

        Args:
            initial_alpha: Starting accuracy weight
            final_alpha: Final accuracy weight
            initial_beta_cost: Starting cost weight
            final_beta_cost: Final cost weight
            initial_beta_energy: Starting energy weight
            final_beta_energy: Final energy weight
            initial_gamma_latency: Starting latency weight
            final_gamma_latency: Final latency weight
            initial_gamma_power: Starting power weight
            final_gamma_power: Final power weight
            total_steps: Total training steps for schedule
        """
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.initial_beta_cost = initial_beta_cost
        self.final_beta_cost = final_beta_cost
        self.initial_beta_energy = initial_beta_energy
        self.final_beta_energy = final_beta_energy
        self.initial_gamma_latency = initial_gamma_latency
        self.final_gamma_latency = final_gamma_latency
        self.initial_gamma_power = initial_gamma_power
        self.final_gamma_power = final_gamma_power
        self.total_steps = total_steps

    def get_weights(self, current_step: int) -> RewardWeights:
        """Get weights for current training step.

        Uses linear interpolation between initial and final weights.

        Args:
            current_step: Current training step

        Returns:
            RewardWeights for current step
        """
        progress = min(1.0, current_step / self.total_steps)

        alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        beta_cost = self.initial_beta_cost + (self.final_beta_cost - self.initial_beta_cost) * progress
        beta_energy = self.initial_beta_energy + (self.final_beta_energy - self.initial_beta_energy) * progress
        gamma_latency = self.initial_gamma_latency + (self.final_gamma_latency - self.initial_gamma_latency) * progress
        gamma_power = self.initial_gamma_power + (self.final_gamma_power - self.initial_gamma_power) * progress

        # Normalize to sum to 1.0
        total = alpha + beta_cost + beta_energy + gamma_latency + gamma_power
        return RewardWeights(
            alpha=alpha / total,
            beta_cost=beta_cost / total,
            beta_energy=beta_energy / total,
            gamma_latency=gamma_latency / total,
            gamma_power=gamma_power / total,
        )


# Example usage
if __name__ == "__main__":
    from orchestrator.data.episode_builder import Episode
    from orchestrator.data.telemetry_cache import TelemetryProfile

    # Create mock episode
    episode = Episode(
        task_id="test",
        initial_prompt="What is 2+2?",
        ground_truth="4",
        final_answer="4",
        correct=True,
        total_energy_joules=15.0,
        total_cost_usd=0.0,
        total_latency_seconds=0.5,
        max_power_watts=100.0,
    )

    # Create reward function with separate weights
    weights = RewardWeights(
        alpha=0.4,
        beta_cost=0.15,
        beta_energy=0.15,
        gamma_latency=0.15,
        gamma_power=0.15,
    )
    normalizers = Normalizers()
    reward_fn = MultiObjectiveReward(weights, normalizers)

    # Compute reward
    reward = reward_fn.compute(episode)
    print(f"Reward: {reward:.4f}")

    # Get breakdown
    breakdown = reward_fn.compute_with_breakdown(episode)
    print("\nReward breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.4f}")

    # Test adaptive weights
    print("\nAdaptive weights schedule:")
    adaptive = AdaptiveRewardWeights()
    for step in [0, 2500, 5000, 7500, 10000]:
        w = adaptive.get_weights(step)
        print(f"  Step {step:5d}: α={w.alpha:.3f}, β_cost={w.beta_cost:.3f}, "
              f"β_energy={w.beta_energy:.3f}, γ_lat={w.gamma_latency:.3f}, γ_pow={w.gamma_power:.3f}")
