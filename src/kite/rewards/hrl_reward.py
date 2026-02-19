"""Hierarchical RL reward combining kernel and runtime terms."""

from __future__ import annotations

from dataclasses import dataclass

from kite.types import RewardBreakdown


@dataclass(slots=True)
class HRLRewardConfig:
    throughput_weight: float = 0.35
    apj_weight: float = 0.25
    apw_weight: float = 0.20
    latency_violation_weight: float = 0.15
    stability_weight: float = 0.05


def compute_hrl_reward(
    throughput_tps: float,
    apj: float,
    apw: float,
    ttft_p95: float,
    e2e_p95: float,
    ttft_sla: float,
    e2e_sla: float,
    stability_score: float,
    config: HRLRewardConfig,
) -> RewardBreakdown:
    latency_violation = max(0.0, ttft_p95 - ttft_sla) + max(0.0, e2e_p95 - e2e_sla)

    total = (
        config.throughput_weight * max(throughput_tps, 0.0)
        + config.apj_weight * max(apj, 0.0)
        + config.apw_weight * max(apw, 0.0)
        + config.stability_weight * max(stability_score, 0.0)
        - config.latency_violation_weight * latency_violation
    )

    return RewardBreakdown(
        performance=throughput_tps,
        energy=apj + apw,
        latency_sla=-latency_violation,
        stability=stability_score,
        total=total,
    )
