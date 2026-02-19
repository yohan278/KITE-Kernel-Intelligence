"""Runtime control environment."""

from __future__ import annotations

from dataclasses import dataclass

from kite.policies.runtime_actor_critic import RuntimeAction
from kite.types import RuntimeState


@dataclass(slots=True)
class RuntimeStepResult:
    next_state: RuntimeState
    throughput_tps: float
    apj: float
    apw: float
    stability: float


class RuntimeEnv:
    def __init__(self) -> None:
        self.initial_state = RuntimeState(
            queue_depth=16,
            phase_ratio=0.5,
            batch_size=2,
            concurrency=2,
            power_cap=450,
            clocks="balanced",
            ttft_p95=1.0,
            e2e_p95=8.0,
        )

    def reset(self) -> RuntimeState:
        return self.initial_state

    def step(self, state: RuntimeState, action: RuntimeAction) -> RuntimeStepResult:
        power_cap, clocks, microbatch, concurrency = action

        queue_depth = max(1, state.queue_depth + (1 if concurrency < state.concurrency else -1))
        phase_ratio = min(1.0, max(0.0, state.phase_ratio + (0.05 if queue_depth > 20 else -0.02)))

        ttft = max(0.2, state.ttft_p95 * (1.1 if power_cap < 400 else 0.95))
        e2e = max(1.0, state.e2e_p95 * (0.95 if concurrency >= state.concurrency else 1.05))

        throughput = max(1.0, float(30 * concurrency * microbatch) / (1.0 + phase_ratio))
        apj = throughput / max(1.0, power_cap * 0.01)
        apw = throughput / max(1.0, power_cap)
        stability = max(0.0, 1.0 - abs(ttft - state.ttft_p95) * 0.1)

        next_state = RuntimeState(
            queue_depth=queue_depth,
            phase_ratio=phase_ratio,
            batch_size=microbatch,
            concurrency=concurrency,
            power_cap=power_cap,
            clocks=clocks,
            ttft_p95=ttft,
            e2e_p95=e2e,
        )

        return RuntimeStepResult(
            next_state=next_state,
            throughput_tps=throughput,
            apj=apj,
            apw=apw,
            stability=stability,
        )
