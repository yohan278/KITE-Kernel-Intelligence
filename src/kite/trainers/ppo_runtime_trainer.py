"""Runtime PPO trainer with real PPO update loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kite.envs.runtime_env import RuntimeEnv
from kite.policies.runtime_actor_critic import RuntimeActorCritic, RuntimeActorCriticConfig
from kite.rewards.hrl_reward import HRLRewardConfig, compute_hrl_reward
from kite.utils.logging import get_logger
from kite.utils.serialization import save_json, save_jsonl

logger = get_logger(__name__)


@dataclass(slots=True)
class PPORuntimeConfig:
    output_dir: Path = Path("checkpoints/runtime_ppo")
    episodes: int = 20
    horizon: int = 10
    ttft_sla: float = 2.0
    e2e_sla: float = 30.0
    ppo_update_every: int = 5
    use_live_telemetry: bool = False


class PPORuntimeTrainer:
    def __init__(
        self,
        env: RuntimeEnv | None = None,
        actor: RuntimeActorCritic | None = None,
        config: PPORuntimeConfig | None = None,
    ) -> None:
        self.config = config or PPORuntimeConfig()
        self.env = env or RuntimeEnv(use_live_telemetry=self.config.use_live_telemetry)
        self.actor = actor or RuntimeActorCritic(RuntimeActorCriticConfig())

    def run(self) -> dict[str, object]:
        logs: list[dict[str, float]] = []
        ppo_stats: list[dict[str, float]] = []

        for episode in range(self.config.episodes):
            state = self.env.reset()
            episode_reward = 0.0

            for step in range(self.config.horizon):
                action = self.actor.select_action(state, explore=True)
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

                done = step == self.config.horizon - 1
                self.actor.store_reward(reward.total, done=done)
                self.actor.update_value(state, reward.total)
                state = result.next_state
                episode_reward += reward.total

                logs.append(
                    {
                        "episode": float(episode),
                        "step": float(step),
                        "throughput_tps": result.throughput_tps,
                        "apj": result.apj,
                        "apw": result.apw,
                        "energy_j": result.energy_j,
                        "ttft_p95": result.next_state.ttft_p95,
                        "e2e_p95": result.next_state.e2e_p95,
                        "reward": reward.total,
                    }
                )

            if (episode + 1) % self.config.ppo_update_every == 0:
                stats = self.actor.ppo_update()
                ppo_stats.append({
                    "episode": float(episode),
                    **stats,
                })
                logger.info(
                    "PPO update at episode %d: policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                    episode, stats["policy_loss"], stats["value_loss"], stats["entropy"],
                )

            if (episode + 1) % max(1, self.config.episodes // 5) == 0:
                avg_recent = sum(l["reward"] for l in logs[-self.config.horizon:]) / self.config.horizon
                logger.info("Episode %d/%d  avg_reward=%.3f", episode + 1, self.config.episodes, avg_recent)

        # Final PPO update on remaining trajectory
        final_stats = self.actor.ppo_update()
        if any(v != 0.0 for v in final_stats.values()):
            ppo_stats.append({"episode": float(self.config.episodes), **final_stats})

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.config.output_dir / "training_history.jsonl", logs)
        if ppo_stats:
            save_jsonl(self.config.output_dir / "ppo_stats.jsonl", ppo_stats)

        model_path = self.config.output_dir / "actor_critic.pt"
        self.actor.save(model_path)

        avg_reward = sum(l["reward"] for l in logs) / len(logs) if logs else 0.0
        checkpoint = {
            "stage": "runtime_ppo",
            "episodes": self.config.episodes,
            "horizon": self.config.horizon,
            "num_records": len(logs),
            "avg_reward": avg_reward,
            "ppo_updates": len(ppo_stats),
            "has_torch_model": self.actor.has_torch,
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint
