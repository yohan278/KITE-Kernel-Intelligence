"""Kernel optimization environment with optional energy capture."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from kite.adapters.ipw_adapter import IPWAdapter
from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.rewards.energy_reward import EnergyRewardConfig, compute_energy_aware_reward
from kite.rewards.kernel_reward import KernelRewardConfig, compute_kernel_reward
from kite.telemetry.energy_capture import EnergyCapture
from kite.telemetry.phase_attribution import attribute_prefill_decode
from kite.types import EnergyTrace, EpisodeRecord, KernelTask, RewardBreakdown


class KernelEnv:
    def __init__(
        self,
        adapter: KernelBenchAdapter,
        reward_config: KernelRewardConfig | None = None,
        energy_aware: bool = False,
        energy_config: EnergyRewardConfig | None = None,
        sla_latency_s: float = 1.0,
        timeout_ms: float = 500.0,
    ) -> None:
        self.adapter = adapter
        self.reward_config = reward_config or KernelRewardConfig()
        self.energy_aware = energy_aware
        self.energy_config = energy_config or EnergyRewardConfig()
        self.sla_latency_s = sla_latency_s
        self.timeout_ms = timeout_ms
        self.energy_capture = EnergyCapture()
        self.ipw_adapter = IPWAdapter()

    def evaluate(
        self,
        task: KernelTask,
        candidate_code: str,
        episode_id: str,
        trace: Optional[EnergyTrace] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> EpisodeRecord:
        candidate = self.adapter.evaluate_candidate(task, candidate_code)

        if self.energy_aware:
            if trace is None:
                trace = self.energy_capture.synthetic_trace(steps=120)
            if not trace.phase_segments:
                trace = attribute_prefill_decode(trace, ttft_s=0.4)
            summary = self.ipw_adapter.summarize(trace, input_tokens=input_tokens, output_tokens=output_tokens)
            reward = compute_energy_aware_reward(
                candidate=candidate,
                summary=summary,
                p95_latency_s=(candidate.runtime_ms or 0.0) / 1000.0,
                sla_latency_s=self.sla_latency_s,
                timeout_ms=self.timeout_ms,
                config=self.energy_config,
            )
            energy_metric = summary.total_energy_j
        else:
            reward = compute_kernel_reward(candidate, timeout_ms=self.timeout_ms, config=self.reward_config)
            energy_metric = 0.0

        return EpisodeRecord(
            episode_id=episode_id,
            task_id=task.task_id,
            kernel_candidate=candidate,
            energy_trace=trace if self.energy_aware else None,
            reward=reward,
            metrics={
                "compile_ok": float(candidate.compile_ok),
                "correct": float(candidate.correct),
                "runtime_ms": candidate.runtime_ms or 0.0,
                "speedup": candidate.speedup or 0.0,
                "energy_j": energy_metric,
            },
            metadata={"task": asdict(task)},
        )
