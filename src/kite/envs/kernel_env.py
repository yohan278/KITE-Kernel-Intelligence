"""Kernel optimization environment."""

from __future__ import annotations

from dataclasses import asdict

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.rewards.kernel_reward import KernelRewardConfig, compute_kernel_reward
from kite.types import EpisodeRecord, KernelTask


class KernelEnv:
    def __init__(self, adapter: KernelBenchAdapter, reward_config: KernelRewardConfig | None = None) -> None:
        self.adapter = adapter
        self.reward_config = reward_config or KernelRewardConfig()

    def evaluate(self, task: KernelTask, candidate_code: str, episode_id: str) -> EpisodeRecord:
        candidate = self.adapter.evaluate_candidate(task, candidate_code)
        reward = compute_kernel_reward(candidate, timeout_ms=500.0, config=self.reward_config)

        return EpisodeRecord(
            episode_id=episode_id,
            task_id=task.task_id,
            kernel_candidate=candidate,
            reward=reward,
            metrics={
                "compile_ok": float(candidate.compile_ok),
                "correct": float(candidate.correct),
                "runtime_ms": candidate.runtime_ms or 0.0,
                "speedup": candidate.speedup or 0.0,
            },
            metadata={"task": asdict(task)},
        )
