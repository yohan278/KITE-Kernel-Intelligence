"""GRPO-style kernel trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from kite.adapters.ipw_adapter import IPWAdapter
from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.adapters.kevin_style_rollouts import RolloutConfig, filter_trajectories, grouped_rollouts
from kite.policies.qwen_policy import QwenPolicy
from kite.rewards.energy_reward import EnergyRewardConfig, compute_energy_aware_reward
from kite.rewards.kernel_reward import staged_kernel_reward
from kite.telemetry.energy_capture import EnergyCapture
from kite.telemetry.phase_attribution import attribute_prefill_decode
from kite.utils.serialization import save_json, save_jsonl


@dataclass(slots=True)
class GRPOKernelConfig:
    output_dir: Path = Path("checkpoints/kernel_grpo")
    epochs: int = 3
    group_size: int = 8
    keep_top_k: int = 4
    energy_aware: bool = False


class GRPOKernelTrainer:
    def __init__(
        self,
        adapter: KernelBenchAdapter,
        policy: QwenPolicy,
        config: GRPOKernelConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.config = config or GRPOKernelConfig()
        self.energy_capture = EnergyCapture()
        self.ipw_adapter = IPWAdapter()

    def run(self) -> dict[str, object]:
        tasks = self.adapter.discover_tasks()
        rollout_cfg = RolloutConfig(group_size=self.config.group_size)

        history: list[dict[str, object]] = []

        for epoch in range(1, self.config.epochs + 1):
            epoch_rewards: list[float] = []
            for task in tasks:
                candidates = grouped_rollouts(self.policy, task, rollout_cfg)
                shortlisted = filter_trajectories(candidates, keep_top_k=self.config.keep_top_k)

                for cand in shortlisted:
                    if self.config.energy_aware:
                        trace = self.energy_capture.synthetic_trace(steps=120)
                        trace = attribute_prefill_decode(trace, ttft_s=0.4)
                        summary = self.ipw_adapter.summarize(trace, input_tokens=512, output_tokens=128)
                        reward = compute_energy_aware_reward(
                            candidate=cand,
                            summary=summary,
                            p95_latency_s=(cand.runtime_ms or 0.0) / 1000.0,
                            sla_latency_s=1.0,
                            timeout_ms=500.0,
                            config=EnergyRewardConfig(),
                        )
                    else:
                        reward = staged_kernel_reward(cand, timeout_ms=500.0, epoch=epoch)
                    epoch_rewards.append(reward.total)

                    history.append(
                        {
                            "epoch": epoch,
                            "task_id": task.task_id,
                            "compile_ok": cand.compile_ok,
                            "correct": cand.correct,
                            "runtime_ms": cand.runtime_ms,
                            "speedup": cand.speedup,
                            "reward": reward.total,
                        }
                    )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.config.output_dir / "training_history.jsonl", history)

        avg_reward = sum(item["reward"] for item in history) / len(history) if history else 0.0
        checkpoint = {
            "stage": "energy_grpo" if self.config.energy_aware else "kernel_grpo",
            "epochs": self.config.epochs,
            "num_records": len(history),
            "avg_reward": avg_reward,
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint
