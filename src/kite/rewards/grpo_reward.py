"""Multi-metric reward for kernel GRPO training.

Supports kernel-type-conditioned reward weighting: compute-bound kernels
emphasize speedup while memory-bound kernels emphasize energy reduction.
Also supports a memory utilization penalty that discourages kernels that
achieve speed through excessive memory bandwidth usage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from kite.types import (
    KERNEL_TYPE_ACTIVATION,
    KERNEL_TYPE_MATMUL,
    KERNEL_TYPE_NORM,
    KERNEL_TYPE_POOLING,
    KERNEL_TYPE_REDUCTION,
    KERNEL_TYPE_SCAN,
    RewardBreakdown,
)

_COMPUTE_BOUND_TYPES = frozenset({KERNEL_TYPE_MATMUL})
_MEMORY_BOUND_TYPES = frozenset({
    KERNEL_TYPE_ACTIVATION, KERNEL_TYPE_REDUCTION, KERNEL_TYPE_NORM,
    KERNEL_TYPE_POOLING, KERNEL_TYPE_SCAN,
})


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
    zeta_mem_util: float = 0.0
    mem_util_threshold_pct: float = 90.0
    type_weight_boost: float = 0.3


def _is_oom_text(*rows: object) -> bool:
    text = " ".join(str(row or "") for row in rows).lower()
    return "out of memory" in text or "cuda oom" in text or "torch.outofmemoryerror" in text


def _type_adjusted_weights(
    config: GRPOMultiMetricRewardConfig,
    kernel_type: str,
) -> tuple[float, float, float]:
    """Return (alpha, beta, eta) adjusted for kernel type."""
    boost = config.type_weight_boost
    alpha = config.alpha_speedup
    beta = config.beta_joules
    eta = config.eta_runtime

    if kernel_type in _COMPUTE_BOUND_TYPES:
        alpha += boost
    elif kernel_type in _MEMORY_BOUND_TYPES:
        beta += boost
        eta += boost * 0.5

    return alpha, beta, eta


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
    kernel_type: str = "unknown",
    avg_mem_util_pct: float | None = None,
) -> RewardBreakdown:
    oom = _is_oom_text(compile_log, correctness_log)

    if not compile_ok:
        total = float(config.compile_fail_reward) - (float(config.oom_penalty) if oom else 0.0)
        return RewardBreakdown(
            correctness=0.0, performance=0.0, energy=0.0,
            latency_sla=0.0, total=total,
        )

    if not correct:
        total = float(config.incorrect_reward) - (float(config.oom_penalty) if oom else 0.0)
        return RewardBreakdown(
            correctness=0.0, performance=0.0, energy=0.0,
            latency_sla=0.0, total=total,
        )

    alpha, beta, eta = _type_adjusted_weights(config, kernel_type)

    safe_speedup = max(1e-6, float(speedup if speedup is not None else 0.0))
    safe_runtime_ms = max(1e-6, float(runtime_ms if runtime_ms is not None else 0.0))
    safe_joules = max(1e-6, float(joules if joules is not None else 0.0))
    safe_power_w = max(0.0, float(avg_power_w if avg_power_w is not None else 0.0))
    latency_s = float(p95_latency_s if p95_latency_s is not None else 0.0)
    latency_penalty = max(0.0, latency_s - float(config.sla_latency_s))

    speedup_term = alpha * math.log(safe_speedup)
    runtime_term = -eta * math.log(safe_runtime_ms)
    joules_term = -beta * math.log(safe_joules)
    power_term = -float(config.delta_avg_power) * safe_power_w
    latency_term = -float(config.gamma_latency) * latency_penalty
    correctness_term = float(config.correctness_bonus)

    mem_util_term = 0.0
    if config.zeta_mem_util > 0 and avg_mem_util_pct is not None:
        excess = max(0.0, avg_mem_util_pct - config.mem_util_threshold_pct)
        mem_util_term = -config.zeta_mem_util * (excess / 100.0)

    total = (
        speedup_term + runtime_term + joules_term
        + power_term + latency_term + correctness_term
        + mem_util_term
    )

    return RewardBreakdown(
        correctness=1.0 + correctness_term,
        performance=speedup_term + runtime_term,
        energy=joules_term + power_term + mem_util_term,
        latency_sla=latency_term,
        total=total,
    )
