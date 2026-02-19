"""SFT trainer (lightweight scaffold)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.policies.qwen_policy import QwenPolicy, QwenPolicyConfig
from kite.utils.serialization import save_json, save_jsonl


@dataclass(slots=True)
class SFTConfig:
    output_dir: Path = Path("checkpoints/sft")
    max_examples: int = 256


class SFTTrainer:
    def __init__(self, adapter: KernelBenchAdapter, policy: QwenPolicy, config: SFTConfig | None = None) -> None:
        self.adapter = adapter
        self.policy = policy
        self.config = config or SFTConfig()

    def run(self) -> dict[str, object]:
        tasks = self.adapter.discover_tasks()[: self.config.max_examples]

        samples = []
        for task in tasks:
            candidate = self.policy.generate_candidate(task, attempt=1)
            if candidate.compile_ok:
                samples.append(
                    {
                        "prompt": task.prompt,
                        "target": candidate.code,
                        "task_id": task.task_id,
                    }
                )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.config.output_dir / "sft_dataset.jsonl"
        save_jsonl(dataset_path, samples)

        checkpoint = {
            "model": asdict(self.policy.config),
            "num_tasks": len(tasks),
            "num_samples": len(samples),
            "stage": "sft",
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint


def build_default_sft_trainer(kernelbench_root: Path) -> SFTTrainer:
    adapter = KernelBenchAdapter(kernelbench_root)
    policy = QwenPolicy(QwenPolicyConfig())
    return SFTTrainer(adapter=adapter, policy=policy)
