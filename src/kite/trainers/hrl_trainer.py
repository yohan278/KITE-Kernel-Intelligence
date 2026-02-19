"""Alternating hierarchical trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.policies.qwen_policy import QwenPolicy
from kite.trainers.grpo_kernel_trainer import GRPOKernelConfig, GRPOKernelTrainer
from kite.trainers.ppo_runtime_trainer import PPORuntimeConfig, PPORuntimeTrainer
from kite.utils.serialization import save_json


@dataclass(slots=True)
class HRLTrainerConfig:
    output_dir: Path = Path("checkpoints/hrl")
    alternating_rounds: int = 2


class HRLTrainer:
    def __init__(
        self,
        kernelbench_root: Path,
        config: HRLTrainerConfig | None = None,
    ) -> None:
        self.config = config or HRLTrainerConfig()
        self.adapter = KernelBenchAdapter(kernelbench_root)
        self.policy = QwenPolicy()

    def run(self) -> dict[str, object]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for round_idx in range(1, self.config.alternating_rounds + 1):
            kernel_trainer = GRPOKernelTrainer(
                adapter=self.adapter,
                policy=self.policy,
                config=GRPOKernelConfig(
                    output_dir=self.config.output_dir / f"round_{round_idx}" / "kernel",
                    epochs=1,
                    energy_aware=True,
                ),
            )
            runtime_trainer = PPORuntimeTrainer(
                config=PPORuntimeConfig(
                    output_dir=self.config.output_dir / f"round_{round_idx}" / "runtime",
                    episodes=5,
                    horizon=5,
                )
            )

            kernel_summary = kernel_trainer.run()
            runtime_summary = runtime_trainer.run()

            summaries.append(
                {
                    "round": round_idx,
                    "kernel": kernel_summary,
                    "runtime": runtime_summary,
                }
            )

        final_summary = {
            "stage": "hrl",
            "rounds": self.config.alternating_rounds,
            "summaries": summaries,
        }
        save_json(self.config.output_dir / "checkpoint.json", final_summary)
        return final_summary
