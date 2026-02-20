"""SFT trainer with real LoRA fine-tuning via trl + peft."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.policies.qwen_policy import QwenPolicy, QwenPolicyConfig
from kite.utils.logging import get_logger
from kite.utils.serialization import load_yaml, save_json, save_jsonl

logger = get_logger(__name__)


@dataclass(slots=True)
class SFTConfig:
    output_dir: Path = Path("checkpoints/sft")
    max_examples: int = 256
    model_config_path: Optional[Path] = None
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    learning_rate: float = 1e-5
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    compile_gate: float = 0.70
    correctness_gate: float = 0.50


def _load_sft_config_from_yaml(path: Path, base: SFTConfig) -> SFTConfig:
    """Override SFTConfig fields from model YAML if present."""
    try:
        cfg = load_yaml(path)
    except Exception:
        return base
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("rank"):
        base.lora_rank = int(lora_cfg["rank"])
    if lora_cfg.get("alpha"):
        base.lora_alpha = int(lora_cfg["alpha"])
    if lora_cfg.get("dropout") is not None:
        base.lora_dropout = float(lora_cfg["dropout"])
    if train_cfg.get("batch_size"):
        base.batch_size = int(train_cfg["batch_size"])
    if train_cfg.get("grad_accum_steps"):
        base.grad_accum_steps = int(train_cfg["grad_accum_steps"])
    if train_cfg.get("learning_rate"):
        base.learning_rate = float(train_cfg["learning_rate"])
    if train_cfg.get("epochs"):
        base.epochs = int(train_cfg["epochs"])
    gen_cfg = model_cfg.get("generation", {})
    if gen_cfg.get("max_new_tokens"):
        base.max_seq_length = int(gen_cfg["max_new_tokens"]) * 2
    return base


class SFTTrainer:
    def __init__(
        self,
        adapter: KernelBenchAdapter,
        policy: QwenPolicy,
        config: SFTConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.config = config or SFTConfig()
        if self.config.model_config_path:
            self.config = _load_sft_config_from_yaml(
                self.config.model_config_path, self.config
            )

    def run(self) -> dict[str, object]:
        tasks = self.adapter.discover_tasks()[: self.config.max_examples]

        samples = self._build_dataset(tasks)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.config.output_dir / "sft_dataset.jsonl"
        save_jsonl(dataset_path, samples)
        logger.info("SFT dataset: %d samples written to %s", len(samples), dataset_path)

        if not self._can_train():
            logger.info("Training deps unavailable; saving dataset only (stub mode)")
            checkpoint = {
                "model": asdict(self.policy.config),
                "num_tasks": len(tasks),
                "num_samples": len(samples),
                "stage": "sft",
                "mode": "dataset_only",
            }
            save_json(self.config.output_dir / "checkpoint.json", checkpoint)
            return checkpoint

        return self._train_real(samples, tasks)

    def _build_dataset(self, tasks) -> list[dict]:
        samples = []
        for task in tasks:
            ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
            if not ref_src or not ref_src.strip():
                continue

            prompt = (
                "You are an expert GPU kernel engineer. "
                "Given the following PyTorch reference implementation, write an optimized "
                "CUDA/Triton kernel replacement that produces identical outputs.\n\n"
                f"Reference implementation:\n```python\n{ref_src}\n```\n\n"
                "Write the optimized kernel implementation:"
            )

            candidate = self.policy.generate_candidate(task, attempt=1)
            if candidate.compile_ok:
                completion = candidate.code
            else:
                completion = ref_src

            samples.append({
                "prompt": prompt,
                "completion": completion,
                "task_id": task.task_id,
                "level": task.level,
            })
        return samples

    @staticmethod
    def _can_train() -> bool:
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401
            import peft  # type: ignore  # noqa: F401
            import trl  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _train_real(self, samples: list[dict], tasks) -> dict[str, object]:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from trl import SFTTrainer as HFSFTTrainer, SFTConfig as HFSFTConfig  # type: ignore

        model_name = self.policy.config.model_name
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.policy.config.dtype, torch.bfloat16)

        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model: %s (dtype=%s)", model_name, dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied: %d trainable / %d total params (%.2f%%)",
            trainable, total, 100.0 * trainable / total,
        )

        def format_sample(example):
            return {"text": f"{example['prompt']}\n\n{example['completion']}"}

        ds = Dataset.from_list(samples).map(format_sample)

        lora_out = self.config.output_dir / "lora_weights"
        training_args = HFSFTConfig(
            output_dir=str(self.config.output_dir / "runs"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_accum_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            max_seq_length=self.config.max_seq_length,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            report_to="none",
            seed=42,
        )

        trainer = HFSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            processing_class=tokenizer,
        )

        logger.info("Starting SFT training (%d epochs, %d samples)", self.config.epochs, len(samples))
        train_result = trainer.train()
        logger.info("SFT training complete: %s", train_result.metrics)

        lora_out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(lora_out))
        tokenizer.save_pretrained(str(lora_out))
        logger.info("LoRA weights saved to %s", lora_out)

        val_metrics = self._validate_gate(tokenizer, model, tasks)

        checkpoint = {
            "model": asdict(self.policy.config),
            "num_tasks": len(tasks),
            "num_samples": len(samples),
            "stage": "sft",
            "mode": "trained",
            "lora_weights_path": str(lora_out),
            "train_loss": train_result.metrics.get("train_loss"),
            "validation": val_metrics,
            "gate_passed": (
                val_metrics["compile_rate"] >= self.config.compile_gate
                and val_metrics["correctness_rate"] >= self.config.correctness_gate
            ),
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint

    def _validate_gate(self, tokenizer, model, tasks) -> dict[str, float]:
        """Quick validation: generate on a subset and check compile/correctness."""
        val_tasks = tasks[: min(20, len(tasks))]
        compile_ok_count = 0
        correct_count = 0

        for task in val_tasks:
            candidate = self.adapter.evaluate_candidate(
                task, self.policy.generate_candidate(task, attempt=0).code
            )
            if candidate.compile_ok:
                compile_ok_count += 1
            if candidate.correct:
                correct_count += 1

        n = max(1, len(val_tasks))
        return {
            "num_val_tasks": n,
            "compile_rate": compile_ok_count / n,
            "correctness_rate": correct_count / n,
        }


def build_default_sft_trainer(kernelbench_root: Path) -> SFTTrainer:
    adapter = KernelBenchAdapter(kernelbench_root)
    policy = QwenPolicy(QwenPolicyConfig())
    return SFTTrainer(adapter=adapter, policy=policy)
