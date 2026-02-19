"""Energy-aware kernel reward."""

from __future__ import annotations

from dataclasses import dataclass, field

from kite.adapters.ipw_adapter import IPWSummary
from kite.rewards.kernel_reward import KernelRewardConfig, compute_kernel_reward
from kite.types import KernelCandidate, RewardBreakdown


@dataclass(slots=True)
class EnergyRewardConfig:
    kernel: KernelRewardConfig = field(default_factory=KernelRewardConfig)
    energy_per_output_token_weight: float = 0.15
    avg_power_weight: float = 0.10
    latency_sla_weight: float = 0.10
    prefill_energy_per_token_weight: float = 0.05
    decode_energy_per_token_weight: float = 0.05
    incorrect_kernel_penalty: float = 2.0


def compute_energy_aware_reward(
    candidate: KernelCandidate,
    summary: IPWSummary,
    p95_latency_s: float,
    sla_latency_s: float,
    timeout_ms: float,
    config: EnergyRewardConfig,
) -> RewardBreakdown:
    base = compute_kernel_reward(candidate, timeout_ms, config.kernel)

    if not candidate.correct:
        base.total -= config.incorrect_kernel_penalty
        return base

    energy_penalty = 0.0
    if summary.energy_per_output_token_j is not None:
        energy_penalty += config.energy_per_output_token_weight * summary.energy_per_output_token_j
    if summary.prefill_energy_per_input_token_j is not None:
        energy_penalty += config.prefill_energy_per_token_weight * summary.prefill_energy_per_input_token_j
    if summary.decode_energy_per_output_token_j is not None:
        energy_penalty += config.decode_energy_per_token_weight * summary.decode_energy_per_output_token_j

    power_penalty = config.avg_power_weight * max(summary.avg_power_w, 0.0)
    latency_penalty = config.latency_sla_weight * max(0.0, p95_latency_s - sla_latency_s)

    total = base.total - energy_penalty - power_penalty - latency_penalty

    return RewardBreakdown(
        correctness=base.correctness,
        performance=base.performance,
        energy=-(energy_penalty + power_penalty),
        latency_sla=-latency_penalty,
        total=total,
    )
