#!/usr/bin/env python3
"""Minimal GRPO training smoke test for 4x NVIDIA L40S GPUs.

Usage (inside an srun GPU allocation):
    module load cuda/12.9.0 python/3.13.5
    source .venv/bin/activate
    python scripts/smoke_train_l40s.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Run this inside an srun GPU allocation:")
        print("  srun --partition=gpu --gres=gpu:4 --pty bash")
        return 1

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s):")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from kite.adapters.kernelbench_adapter import KernelBenchAdapter
    from kite.rewards.kernel_reward import staged_kernel_reward

    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    output_dir = ROOT / "checkpoints" / "smoke_l40s"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Loading model: {model_name} ===")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded in {time.time() - t0:.1f}s")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    kb_root = ROOT / "external" / "KernelBench"
    adapter = KernelBenchAdapter(kb_root, enable_kernelbench_eval=False)
    tasks = adapter.discover_tasks()[:3]
    print(f"Using {len(tasks)} tasks for smoke test")

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for i, code in enumerate(completions):
            if isinstance(code, list):
                code = code[0].get("content", "") if code else ""
            task = tasks[i % len(tasks)]
            candidate = adapter.evaluate_candidate(task, code)
            r = staged_kernel_reward(candidate, timeout_ms=500.0, epoch=1, correctness_bias_epochs=2)
            rewards.append(r.total)
        return rewards

    prompts = []
    for task in tasks:
        ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
        prompt = (
            "You are an expert GPU kernel engineer. "
            "Optimize this PyTorch model with a custom GPU kernel.\n\n"
            f"```python\n{ref_src}\n```\n\n"
            "Write the optimized kernel:"
        )
        prompts.append([{"role": "user", "content": prompt}])

    train_dataset = Dataset.from_dict({"prompt": prompts})

    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "runs"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=128,
        beta=0.04,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
        seed=42,
        gradient_checkpointing=True,
        max_steps=4,
    )

    print("\n=== Starting GRPO Training ===")
    print(f"  epochs: {grpo_config.num_train_epochs}")
    print(f"  max_steps: {grpo_config.max_steps}")
    print(f"  batch_size: {grpo_config.per_device_train_batch_size}")
    print(f"  num_generations: {grpo_config.num_generations}")
    print(f"  max_completion_length: {grpo_config.max_completion_length}")
    print(f"  learning_rate: {grpo_config.learning_rate}")
    print(f"  output_dir: {output_dir}")
    print()

    t0 = time.time()
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"\n=== Training complete in {elapsed:.1f}s ===")
    print(f"Metrics: {train_result.metrics}")

    lora_out = output_dir / "lora_weights"
    lora_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(lora_out))
    tokenizer.save_pretrained(str(lora_out))
    print(f"LoRA weights saved to {lora_out}")

    print("\n=== GPU memory after training ===")
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

    print("\nSmoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
