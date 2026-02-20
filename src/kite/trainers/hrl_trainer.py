"""Alternating hierarchical trainer: kernel GRPO + runtime PPO + hierarchy selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.envs.runtime_env import RuntimeEnv
from kite.policies.hierarchy_controller import HierarchyController, HierarchyControllerConfig
from kite.policies.qwen_policy import QwenPolicy
from kite.policies.runtime_actor_critic import RuntimeActorCritic, RuntimeActorCriticConfig
from kite.rewards.hrl_reward import HRLRewardConfig, compute_hrl_reward
from kite.trainers.grpo_kernel_trainer import GRPOKernelConfig, GRPOKernelTrainer
from kite.trainers.ppo_runtime_trainer import PPORuntimeConfig, PPORuntimeTrainer
from kite.utils.logging import get_logger
from kite.utils.serialization import save_json, save_jsonl

logger = get_logger(__name__)


@dataclass(slots=True)
class HRLTrainerConfig:
    output_dir: Path = Path("checkpoints/hrl")
    alternating_rounds: int = 2
    kernel_epochs_per_round: int = 1
    runtime_episodes_per_round: int = 10
    runtime_horizon: int = 10
    joint_finetune_episodes: int = 5
    use_live_telemetry: bool = False


class HRLTrainer:
    def __init__(
        self,
        kernelbench_root: Path,
        config: HRLTrainerConfig | None = None,
    ) -> None:
        self.config = config or HRLTrainerConfig()
        self.adapter = KernelBenchAdapter(kernelbench_root)
        self.policy = QwenPolicy()
        self.hierarchy = HierarchyController(HierarchyControllerConfig())
        self.runtime_actor = RuntimeActorCritic(RuntimeActorCriticConfig())
        self.runtime_env = RuntimeEnv(use_live_telemetry=self.config.use_live_telemetry)

    def run(self) -> dict[str, object]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        all_logs: list[dict] = []

        for round_idx in range(1, self.config.alternating_rounds + 1):
            round_dir = self.config.output_dir / f"round_{round_idx}"
            logger.info("=== HRL Round %d/%d ===", round_idx, self.config.alternating_rounds)

            # Step 1: Kernel policy update (GRPO)
            kernel_summary = self._kernel_step(round_dir, round_idx)

            # Step 2: Runtime policy update (PPO)
            runtime_summary = self._runtime_step(round_dir, round_idx)

            # Step 3: Joint fine-tune with hierarchy selection
            joint_logs = self._joint_finetune(round_dir, round_idx)
            all_logs.extend(joint_logs)

            round_summary = {
                "round": round_idx,
                "kernel": kernel_summary,
                "runtime": runtime_summary,
                "joint_steps": len(joint_logs),
                "joint_avg_reward": (
                    sum(l.get("reward", 0.0) for l in joint_logs) / len(joint_logs)
                    if joint_logs else 0.0
                ),
            }
            summaries.append(round_summary)
            logger.info("Round %d complete: joint_avg_reward=%.3f",
                        round_idx, round_summary["joint_avg_reward"])

        # Save hierarchy and runtime models
        self.hierarchy.save(self.config.output_dir / "hierarchy_controller.pt")
        self.runtime_actor.save(self.config.output_dir / "runtime_actor_critic.pt")

        if all_logs:
            save_jsonl(self.config.output_dir / "joint_training_history.jsonl", all_logs)

        final_summary = {
            "stage": "hrl",
            "rounds": self.config.alternating_rounds,
            "summaries": summaries,
            "has_torch_hierarchy": self.hierarchy.has_torch,
            "has_torch_runtime": self.runtime_actor.has_torch,
        }
        save_json(self.config.output_dir / "checkpoint.json", final_summary)
        return final_summary

    def _kernel_step(self, round_dir: Path, round_idx: int) -> dict[str, object]:
        logger.info("  Kernel GRPO step (round %d)", round_idx)
        kernel_trainer = GRPOKernelTrainer(
            adapter=self.adapter,
            policy=self.policy,
            config=GRPOKernelConfig(
                output_dir=round_dir / "kernel",
                epochs=self.config.kernel_epochs_per_round,
                energy_aware=True,
            ),
        )
        return kernel_trainer.run()

    def _runtime_step(self, round_dir: Path, round_idx: int) -> dict[str, object]:
        logger.info("  Runtime PPO step (round %d)", round_idx)
        runtime_trainer = PPORuntimeTrainer(
            env=self.runtime_env,
            actor=self.runtime_actor,
            config=PPORuntimeConfig(
                output_dir=round_dir / "runtime",
                episodes=self.config.runtime_episodes_per_round,
                horizon=self.config.runtime_horizon,
            ),
        )
        return runtime_trainer.run()

    def _joint_finetune(self, round_dir: Path, round_idx: int) -> list[dict]:
        """Joint fine-tuning: hierarchy selects kernel family, runtime acts on it."""
        logger.info("  Joint fine-tune step (round %d)", round_idx)
        logs: list[dict] = []
        hrl_config = HRLRewardConfig()

        for ep in range(self.config.joint_finetune_episodes):
            state = self.runtime_env.reset()
            episode_reward = 0.0

            for step in range(self.config.runtime_horizon):
                # High-level: select kernel family
                decision = self.hierarchy.choose_kernel_family(state, explore=True)

                # Low-level: select runtime action conditioned on state
                action = self.runtime_actor.select_action(state, explore=True)
                result = self.runtime_env.step(state, action)

                reward = compute_hrl_reward(
                    throughput_tps=result.throughput_tps,
                    apj=result.apj,
                    apw=result.apw,
                    ttft_p95=result.next_state.ttft_p95,
                    e2e_p95=result.next_state.e2e_p95,
                    ttft_sla=2.0,
                    e2e_sla=30.0,
                    stability_score=result.stability,
                    config=hrl_config,
                )

                done = step == self.config.runtime_horizon - 1
                self.runtime_actor.store_reward(reward.total, done=done)

                # Update hierarchy controller from joint reward
                h_stats = self.hierarchy.update_from_reward(reward.total)

                state = result.next_state
                episode_reward += reward.total

                logs.append({
                    "round": round_idx,
                    "episode": ep,
                    "step": step,
                    "kernel_family": decision.kernel_family,
                    "family_confidence": decision.confidence,
                    "throughput_tps": result.throughput_tps,
                    "apj": result.apj,
                    "apw": result.apw,
                    "energy_j": result.energy_j,
                    "reward": reward.total,
                    "hierarchy_loss": h_stats.get("loss", 0.0),
                })

            # PPO update for runtime actor at end of each joint episode
            ppo_stats = self.runtime_actor.ppo_update()

        return logs
