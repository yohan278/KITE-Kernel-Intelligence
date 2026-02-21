"""KernelBench energy-aware evaluation environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.measurement.protocol import MeasurementConfig, MeasurementProtocol
from kite.rewards.ipw_reward import IPWRewardConfig, compute_ipw_reward
from kite.types import KernelTask, RewardBreakdown


@dataclass(slots=True)
class KernelBenchEnergyStep:
    task_id: str
    compile_ok: bool
    correct: bool
    runtime_ms: float
    speedup: float
    avg_power_w: float
    joules: float
    reward: RewardBreakdown
    logs: dict[str, object]


def _default_workload() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            x = torch.randn(256, 256, device="cuda")
            _ = torch.mm(x, x)
            return
    except Exception:
        pass

    # CPU fallback
    import math

    _ = sum(math.sqrt(i) for i in range(1000))


class KernelBenchEnergyEnv:
    def __init__(
        self,
        adapter: KernelBenchAdapter,
        measurement_config: MeasurementConfig | None = None,
        reward_config: IPWRewardConfig | None = None,
        sla_latency_s: float = 1.0,
    ) -> None:
        self.adapter = adapter
        self.measurement = MeasurementProtocol(measurement_config or MeasurementConfig())
        self.reward_config = reward_config or IPWRewardConfig()
        self.sla_latency_s = sla_latency_s

    def evaluate(
        self,
        task: KernelTask,
        code: str,
        workload: Optional[Callable[[], object]] = None,
        baseline_runtime_ms: float = 100.0,
    ) -> KernelBenchEnergyStep:
        candidate = self.adapter.evaluate_candidate(
            task=task,
            candidate_code=code,
            baseline_runtime_ms=baseline_runtime_ms,
        )

        # Always run timing protocol so Phase 0/1 output schema is consistent.
        fn = workload or _default_workload
        meas = self.measurement.measure(fn)
        runtime_ms = meas.runtime_ms_mean
        speedup = baseline_runtime_ms / runtime_ms if runtime_ms > 0 else 0.0
        avg_power = meas.avg_power_w_mean
        joules = meas.energy_j_mean

        reward = compute_ipw_reward(
            compile_ok=candidate.compile_ok,
            correct=candidate.correct,
            speedup=speedup,
            joules=joules,
            p95_latency_s=runtime_ms / 1000.0,
            sla_latency_s=self.sla_latency_s,
            config=self.reward_config,
        )

        logs = {
            "candidate_logs": candidate.logs,
            "task": asdict(task),
            "measurement_repeats": meas.repeats,
            "runtime_ms_std": meas.runtime_ms_std,
            "energy_j_std": meas.energy_j_std,
            "power_trace": [
                {"timestamp_s": s.timestamp_s, "power_w": s.power_w}
                for s in (meas.runs[0].samples if meas.runs else [])
            ],
        }

        return KernelBenchEnergyStep(
            task_id=task.task_id,
            compile_ok=candidate.compile_ok,
            correct=candidate.correct,
            runtime_ms=runtime_ms,
            speedup=speedup,
            avg_power_w=avg_power,
            joules=joules,
            reward=reward,
            logs=logs,
        )
