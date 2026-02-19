"""Kernel-only reward."""

from __future__ import annotations

from dataclasses import dataclass

from kite.types import KernelCandidate, RewardBreakdown


@dataclass(slots=True)
class KernelRewardConfig:
    correctness_weight: float = 0.7
    speedup_weight: float = 0.3
    compile_fail_penalty: float = 1.0
    timeout_penalty: float = 0.5


def compute_kernel_reward(
    candidate: KernelCandidate,
    timeout_ms: float,
    config: KernelRewardConfig,
) -> RewardBreakdown:
    correctness = 1.0 if candidate.correct else 0.0
    performance = max(0.0, candidate.speedup or 0.0)

    penalty = 0.0
    if not candidate.compile_ok:
        penalty += config.compile_fail_penalty
    if candidate.runtime_ms is not None and candidate.runtime_ms > timeout_ms:
        penalty += config.timeout_penalty

    total = (
        config.correctness_weight * correctness
        + config.speedup_weight * performance
        - penalty
    )

    return RewardBreakdown(
        correctness=correctness,
        performance=performance,
        total=total,
    )


def staged_kernel_reward(
    candidate: KernelCandidate,
    timeout_ms: float,
    epoch: int,
    correctness_bias_epochs: int = 2,
) -> RewardBreakdown:
    if epoch <= correctness_bias_epochs:
        config = KernelRewardConfig(correctness_weight=0.85, speedup_weight=0.15)
    else:
        config = KernelRewardConfig(correctness_weight=0.55, speedup_weight=0.45)
    return compute_kernel_reward(candidate, timeout_ms, config)
