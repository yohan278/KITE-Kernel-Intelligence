"""Reward used by energy-aware KernelBench environment (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from kite.types import RewardBreakdown


@dataclass(slots=True)
class IPWRewardConfig:
    alpha_speedup: float = 1.0
    beta_joules: float = 0.5
    gamma_latency: float = 0.25
    compile_fail_reward: float = -1.0
    incorrect_reward: float = -0.5


def compute_ipw_reward(
    *,
    compile_ok: bool,
    correct: bool,
    speedup: float | None,
    joules: float | None,
    p95_latency_s: float | None,
    sla_latency_s: float,
    config: IPWRewardConfig,
) -> RewardBreakdown:
    if not compile_ok:
        return RewardBreakdown(
            correctness=0.0,
            performance=0.0,
            energy=0.0,
            latency_sla=0.0,
            total=config.compile_fail_reward,
        )
    if not correct:
        return RewardBreakdown(
            correctness=0.0,
            performance=0.0,
            energy=0.0,
            latency_sla=0.0,
            total=config.incorrect_reward,
        )

    safe_speedup = max(1e-6, float(speedup if speedup is not None else 0.0))
    safe_joules = max(1e-6, float(joules if joules is not None else 0.0))
    latency = float(p95_latency_s if p95_latency_s is not None else 0.0)
    latency_penalty = max(0.0, latency - sla_latency_s)

    perf_term = config.alpha_speedup * math.log(safe_speedup)
    energy_term = -config.beta_joules * math.log(safe_joules)
    latency_term = -config.gamma_latency * latency_penalty
    total = perf_term + energy_term + latency_term

    return RewardBreakdown(
        correctness=1.0,
        performance=perf_term,
        energy=energy_term,
        latency_sla=latency_term,
        total=total,
    )

