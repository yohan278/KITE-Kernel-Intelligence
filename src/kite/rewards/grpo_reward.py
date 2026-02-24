"""Multi-metric reward for kernel GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
import math

from kite.types import RewardBreakdown


@dataclass(slots=True)
class GRPOMultiMetricRewardConfig:
    alpha_speedup: float = 1.0
    beta_joules: float = 0.0
    gamma_latency: float = 0.25
    delta_avg_power: float = 0.01
    eta_runtime: float = 0.10
    correctness_bonus: float = 0.0
    compile_fail_reward: float = -1.0
    incorrect_reward: float = -0.5
    oom_penalty: float = 0.5
    sla_latency_s: float = 1.0


def _is_oom_text(*rows: object) -> bool:
    text = " ".join(str(row or "") for row in rows).lower()
    return "out of memory" in text or "cuda oom" in text or "torch.outofmemoryerror" in text


def compute_grpo_multi_metric_reward(
    *,
    compile_ok: bool,
    correct: bool,
    speedup: float | None,
    runtime_ms: float | None,
    joules: float | None,
    avg_power_w: float | None,
    p95_latency_s: float | None,
    compile_log: str | None,
    correctness_log: str | None,
    config: GRPOMultiMetricRewardConfig,
) -> RewardBreakdown:
    oom = _is_oom_text(compile_log, correctness_log)

    if not compile_ok:
        total = float(config.compile_fail_reward) - (float(config.oom_penalty) if oom else 0.0)
        return RewardBreakdown(
            correctness=0.0,
            performance=0.0,
            energy=0.0,
            latency_sla=0.0,
            total=total,
        )

    if not correct:
        total = float(config.incorrect_reward) - (float(config.oom_penalty) if oom else 0.0)
        return RewardBreakdown(
            correctness=0.0,
            performance=0.0,
            energy=0.0,
            latency_sla=0.0,
            total=total,
        )

    safe_speedup = max(1e-6, float(speedup if speedup is not None else 0.0))
    safe_runtime_ms = max(1e-6, float(runtime_ms if runtime_ms is not None else 0.0))
    safe_joules = max(1e-6, float(joules if joules is not None else 0.0))
    safe_power_w = max(0.0, float(avg_power_w if avg_power_w is not None else 0.0))
    latency_s = float(p95_latency_s if p95_latency_s is not None else 0.0)
    latency_penalty = max(0.0, latency_s - float(config.sla_latency_s))

    speedup_term = float(config.alpha_speedup) * math.log(safe_speedup)
    runtime_term = -float(config.eta_runtime) * math.log(safe_runtime_ms)
    joules_term = -float(config.beta_joules) * math.log(safe_joules)
    power_term = -float(config.delta_avg_power) * safe_power_w
    latency_term = -float(config.gamma_latency) * latency_penalty
    correctness_term = float(config.correctness_bonus)

    total = speedup_term + runtime_term + joules_term + power_term + latency_term + correctness_term

    return RewardBreakdown(
        correctness=1.0 + correctness_term,
        performance=speedup_term + runtime_term,
        energy=joules_term + power_term,
        latency_sla=latency_term,
        total=total,
    )

