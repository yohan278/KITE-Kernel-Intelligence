"""Runtime PPO-style trainer (table-based actor-critic backend)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kite.envs.runtime_env import RuntimeEnv
from kite.policies.runtime_actor_critic import RuntimeActorCritic
from kite.rewards.hrl_reward import HRLRewardConfig, compute_hrl_reward
from kite.utils.serialization import save_json, save_jsonl


@dataclass(slots=True)
class PPORuntimeConfig:
    output_dir: Path = Path("checkpoints/runtime_ppo")
    episodes: int = 20
    horizon: int = 10
    ttft_sla: float = 2.0
    e2e_sla: float = 30.0


class PPORuntimeTrainer:
    def __init__(
        self,
        env: RuntimeEnv | None = None,
        actor: RuntimeActorCritic | None = None,
        config: PPORuntimeConfig | None = None,
    ) -> None:
        self.env = env or RuntimeEnv()
        self.actor = actor or RuntimeActorCritic()
        self.config = config or PPORuntimeConfig()

    def run(self) -> dict[str, object]:
        logs: list[dict[str, float]] = []

        for episode in range(self.config.episodes):
            state = self.env.reset()
            for step in range(self.config.horizon):
                action = self.actor.select_action(state)
                result = self.env.step(state, action)
                reward = compute_hrl_reward(
                    throughput_tps=result.throughput_tps,
                    apj=result.apj,
                    apw=result.apw,
                    ttft_p95=result.next_state.ttft_p95,
                    e2e_p95=result.next_state.e2e_p95,
                    ttft_sla=self.config.ttft_sla,
                    e2e_sla=self.config.e2e_sla,
                    stability_score=result.stability,
                    config=HRLRewardConfig(),
                )

                self.actor.update_value(state, reward.total)
                state = result.next_state

                logs.append(
                    {
                        "episode": float(episode),
                        "step": float(step),
                        "throughput_tps": result.throughput_tps,
                        "apj": result.apj,
                        "apw": result.apw,
                        "reward": reward.total,
                    }
                )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.config.output_dir / "training_history.jsonl", logs)

        avg_reward = sum(l["reward"] for l in logs) / len(logs) if logs else 0.0
        checkpoint = {
            "stage": "runtime_ppo",
            "episodes": self.config.episodes,
            "horizon": self.config.horizon,
            "num_records": len(logs),
            "avg_reward": avg_reward,
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint
