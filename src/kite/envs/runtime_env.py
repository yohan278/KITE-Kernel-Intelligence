"""Runtime control environment with simulation and optional vLLM backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kite.policies.runtime_actor_critic import RuntimeAction
from kite.telemetry.energy_capture import EnergyCapture
from kite.types import RuntimeState
from kite.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RuntimeStepResult:
    next_state: RuntimeState
    throughput_tps: float
    apj: float
    apw: float
    stability: float
    energy_j: float = 0.0
    latency_s: float = 0.0


class RuntimeEnv:
    """Runtime environment that maps (state, action) -> next_state + metrics.

    In simulation mode (default), uses an analytical model of GPU behavior.
    In live mode (``use_live_telemetry=True``), drives real GPU power settings
    and reads back telemetry via ``EnergyCapture``.
    """

    def __init__(
        self,
        use_live_telemetry: bool = False,
        energy_capture: Optional[EnergyCapture] = None,
    ) -> None:
        self.use_live_telemetry = use_live_telemetry
        self.energy_capture = energy_capture or EnergyCapture()
        self.initial_state = RuntimeState(
            queue_depth=16,
            phase_ratio=0.5,
            batch_size=2,
            concurrency=2,
            power_cap=450,
            clocks="balanced",
            ttft_p95=1.0,
            e2e_p95=8.0,
            throughput_tps=0.0,
            avg_power_w=0.0,
            phase_id="mixed",
        )
        self._step_count = 0

    def reset(self) -> RuntimeState:
        self._step_count = 0
        return self.initial_state

    def step(self, state: RuntimeState, action: RuntimeAction) -> RuntimeStepResult:
        self._step_count += 1
        if self.use_live_telemetry:
            return self._step_live(state, action)
        return self._step_simulated(state, action)

    def _step_simulated(self, state: RuntimeState, action: RuntimeAction) -> RuntimeStepResult:
        """Analytical simulation of GPU runtime dynamics."""
        power_cap, clocks, microbatch, concurrency = action

        queue_depth = max(1, state.queue_depth + (1 if concurrency < state.concurrency else -1))
        phase_ratio = min(1.0, max(0.0, state.phase_ratio + (0.05 if queue_depth > 20 else -0.02)))

        clock_factor = {"efficiency": 1.15, "balanced": 1.0, "performance": 0.90}.get(clocks, 1.0)
        power_factor = min(1.0, power_cap / 450.0)

        ttft = max(0.2, state.ttft_p95 * clock_factor * (1.0 / power_factor))
        e2e = max(1.0, state.e2e_p95 * (0.95 if concurrency >= state.concurrency else 1.05))

        throughput = max(1.0, float(30 * concurrency * microbatch) / (1.0 + phase_ratio))
        throughput *= power_factor

        energy_per_step = power_cap * (e2e / 1000.0)
        apj = throughput / max(1.0, energy_per_step)
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
            throughput_tps=throughput,
            avg_power_w=float(power_cap),
            phase_id="decode" if phase_ratio >= 0.5 else "prefill",
        )

        return RuntimeStepResult(
            next_state=next_state,
            throughput_tps=throughput,
            apj=apj,
            apw=apw,
            stability=stability,
            energy_j=energy_per_step,
            latency_s=e2e,
        )

    def _step_live(self, state: RuntimeState, action: RuntimeAction) -> RuntimeStepResult:
        """Live step: apply GPU settings and measure real telemetry."""
        power_cap, clocks, microbatch, concurrency = action

        self._apply_gpu_settings(power_cap, clocks)

        try:
            import torch  # type: ignore

            def _dummy_workload(inputs):
                x = torch.randn(microbatch, 4096, 4096, device="cuda")
                return torch.mm(x, x.T)

            trace = self.energy_capture.capture_kernel_trace(
                kernel_fn=_dummy_workload,
                inputs=None,
                warmup_iters=2,
                measure_iters=5,
            )
            total_energy = trace.energy_j[-1] if trace.energy_j else 0.0
            avg_power = sum(trace.power_w) / len(trace.power_w) if trace.power_w else float(power_cap)
            duration = trace.timestamps[-1] - trace.timestamps[0] if len(trace.timestamps) >= 2 else 1.0
        except Exception as exc:
            logger.warning("Live telemetry failed: %s; falling back to simulation", exc)
            return self._step_simulated(state, action)

        throughput = max(1.0, float(30 * concurrency * microbatch) / max(0.1, duration))
        apj = throughput / max(1.0, total_energy)
        apw = throughput / max(1.0, avg_power)

        ttft = max(0.2, duration * 0.3)
        e2e = max(1.0, duration)
        stability = max(0.0, 1.0 - abs(ttft - state.ttft_p95) * 0.1)

        next_state = RuntimeState(
            queue_depth=max(1, state.queue_depth),
            phase_ratio=min(1.0, max(0.0, state.phase_ratio)),
            batch_size=microbatch,
            concurrency=concurrency,
            power_cap=power_cap,
            clocks=clocks,
            ttft_p95=ttft,
            e2e_p95=e2e,
            throughput_tps=throughput,
            avg_power_w=avg_power,
            phase_id="decode" if state.phase_ratio >= 0.5 else "prefill",
        )

        return RuntimeStepResult(
            next_state=next_state,
            throughput_tps=throughput,
            apj=apj,
            apw=apw,
            stability=stability,
            energy_j=total_energy,
            latency_s=duration,
        )

    @staticmethod
    def _apply_gpu_settings(power_cap: int, clocks: str) -> None:
        """Attempt to set GPU power limit via nvidia-smi."""
        try:
            import subprocess

            subprocess.run(
                ["nvidia-smi", "-pl", str(power_cap)],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
