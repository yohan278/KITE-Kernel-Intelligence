#!/usr/bin/env python3
"""
5-step online RL smoke test with real KernelBench evaluation.

GPU layout:
  - GPU 0: Qwen-0.5B model (generation + gradient updates)
  - GPU 1: KernelBench kernel evaluation (compile, correctness, timing)

Each step:
  1. Sample a Level-1 KernelBench task
  2. Generate GROUP_SIZE candidate kernels from the policy
  3. Evaluate each candidate on GPU 1 via kernelbench.eval
  4. Compute GRPO-style rewards (group-relative advantages)
  5. Policy gradient update on the LoRA parameters
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
KB_SRC = ROOT / "external" / "KernelBench" / "src"
for p in (str(SRC), str(KB_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class SmokeConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    num_steps: int = 5
    group_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.8
    learning_rate: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 16
    model_device: str = "cuda:0"
    eval_device: str = "cuda:1"
    output_dir: Path = ROOT / "outputs" / "smoke_rl_online"


def build_prompt(task_code: str) -> str:
    return (
        "You are an expert GPU kernel engineer. Given the following PyTorch model, "
        "write an optimized CUDA kernel replacement.\n\n"
        "IMPORTANT: Your code MUST define a class called `ModelNew` (not `Model`) that "
        "has the same interface as the original `Model` class. `ModelNew` must produce "
        "identical outputs to the original.\n\n"
        f"```python\n{task_code}\n```\n\n"
        "Write a complete Python module with `ModelNew`, `get_inputs`, and `get_init_inputs`:"
    )


def evaluate_candidate_online(
    ref_arch_src: str,
    candidate_code: str,
    eval_device: str,
) -> dict:
    """Run KernelBench eval on the eval GPU. Returns a result dict."""
    import torch
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

    device = torch.device(eval_device)
    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=candidate_code,
            num_correct_trials=1,
            num_perf_trials=5,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=False,
            device=device,
            backend="cuda",
            precision=torch.float32,
        )
    except Exception as e:
        return {
            "compiled": False, "correct": False,
            "runtime_us": -1, "error": str(e), "eval_mode": "online",
        }

    if result is None:
        return {
            "compiled": False, "correct": False,
            "runtime_us": -1, "error": "eval returned None", "eval_mode": "online",
        }

    compiled = bool(getattr(result, "compiled", False))
    correct = bool(getattr(result, "correctness", False)) if compiled else False
    runtime_us = float(getattr(result, "runtime", -1.0) or -1.0)

    return {
        "compiled": compiled,
        "correct": correct,
        "runtime_us": runtime_us,
        "error": None,
        "eval_mode": "online",
        "metadata": {
            k: str(v) for k, v in (getattr(result, "metadata", {}) or {}).items()
        },
    }


def compute_reward(eval_result: dict, baseline_runtime_us: float = 100_000.0) -> float:
    if not eval_result["compiled"]:
        return -1.0
    if not eval_result["correct"]:
        return -0.5
    runtime_us = eval_result["runtime_us"]
    if runtime_us <= 0:
        return 0.0
    speedup = baseline_runtime_us / runtime_us
    return min(5.0, 0.5 + 0.5 * speedup)


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantage: subtract group mean, divide by std."""
    import torch
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std()
    if std < 1e-8:
        return [0.0] * len(rewards)
    return ((r - mean) / std).tolist()


def main() -> int:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kite.adapters.kernelbench_adapter import KernelBenchAdapter

    cfg = SmokeConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run inside srun --gres=gpu:2")
        return 1

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if n_gpus < 2:
        print("WARNING: Only 1 GPU found. Model and eval will share GPU 0.")
        cfg.eval_device = "cuda:0"

    # --- Discover Level-1 tasks ---
    adapter = KernelBenchAdapter(ROOT / "external" / "KernelBench", enable_kernelbench_eval=False)
    all_tasks = adapter.discover_tasks()
    level1_tasks = [t for t in all_tasks if t.level == 1]
    print(f"Discovered {len(all_tasks)} tasks total, {len(level1_tasks)} Level-1 (easy)")
    if not level1_tasks:
        print("ERROR: No Level-1 tasks found")
        return 1

    # --- Load model on GPU 0 ---
    print(f"\nLoading model: {cfg.model_name} on {cfg.model_device}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(cfg.model_device)
    # Ensure valid pad/eos token IDs to avoid CUDA device-side assert during generation
    vocab_size = getattr(model.config, "vocab_size", None) or len(tokenizer)
    pad_id = tokenizer.pad_token_id or getattr(model.config, "eos_token_id", 0)
    if vocab_size is not None and (pad_id < 0 or pad_id >= vocab_size):
        pad_id = 0
    eos_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", pad_id)
    if vocab_size is not None and (eos_id < 0 or eos_id >= vocab_size):
        eos_id = pad_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = eos_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model loaded in {time.time()-t0:.1f}s | {trainable:,} trainable / {total:,} total params")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
    )

    # --- Training loop ---
    history = []
    print(f"\n{'='*60}")
    print(f"Starting {cfg.num_steps}-step online RL training")
    print(f"  group_size={cfg.group_size}, max_new_tokens={cfg.max_new_tokens}")
    print(f"  model on {cfg.model_device}, eval on {cfg.eval_device}")
    print(f"{'='*60}\n")

    for step in range(1, cfg.num_steps + 1):
        step_t0 = time.time()
        task = level1_tasks[(step - 1) % len(level1_tasks)]
        ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
        prompt_text = build_prompt(ref_src)

        print(f"--- Step {step}/{cfg.num_steps} | Task: {task.task_id} ---")

        # 1) Generate candidates
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=2048).to(cfg.model_device)

        all_log_probs = []
        all_codes = []

        model.eval()
        for g in range(cfg.group_size):
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                )
            new_ids = gen_out[0][inputs["input_ids"].shape[1]:]
            code = tokenizer.decode(new_ids, skip_special_tokens=True)

            # Extract code from markdown blocks if present
            if "```python" in code:
                parts = code.split("```python")
                if len(parts) > 1:
                    code = parts[1].split("```")[0].strip()
            elif "```" in code:
                parts = code.split("```")
                if len(parts) > 1:
                    block = parts[1]
                    if block.startswith(("python\n", "py\n")):
                        block = "\n".join(block.split("\n")[1:])
                    code = block.strip()

            all_codes.append(code)

            # Compute log-prob of the generated sequence under current policy
            model.train()
            full_ids = gen_out[0].unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(full_ids).logits
            prompt_len = inputs["input_ids"].shape[1]
            shift_logits = logits[0, prompt_len - 1:-1, :]
            shift_labels = full_ids[0, prompt_len:]
            log_probs_per_token = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            selected_log_probs = log_probs_per_token.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            all_log_probs.append(selected_log_probs)
            model.eval()

        gen_time = time.time() - step_t0

        # 2) Evaluate candidates online on GPU 1
        eval_t0 = time.time()
        eval_results = []
        for g, code in enumerate(all_codes):
            print(f"  Evaluating candidate {g+1}/{cfg.group_size}...", end=" ", flush=True)
            r = evaluate_candidate_online(ref_src, code, cfg.eval_device)
            eval_results.append(r)
            status = "COMPILE_FAIL"
            if r["compiled"] and not r["correct"]:
                status = "WRONG"
            elif r["correct"]:
                status = f"CORRECT (runtime={r['runtime_us']:.0f} us)"
            print(status)
        eval_time = time.time() - eval_t0

        # 3) Compute rewards and GRPO advantages
        rewards = [compute_reward(r) for r in eval_results]
        advantages = grpo_advantages(rewards)

        # 4) Policy gradient update
        model.train()
        optimizer.zero_grad()
        policy_loss = torch.tensor(0.0, device=cfg.model_device)
        for log_probs, adv in zip(all_log_probs, advantages):
            policy_loss += -(adv * log_probs.sum())
        policy_loss = policy_loss / cfg.group_size
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        update_time = time.time() - step_t0 - gen_time - eval_time
        total_step_time = time.time() - step_t0

        # 5) Log
        compile_rate = sum(1 for r in eval_results if r["compiled"]) / cfg.group_size
        correct_rate = sum(1 for r in eval_results if r["correct"]) / cfg.group_size
        avg_reward = sum(rewards) / len(rewards)

        step_record = {
            "step": step,
            "task_id": task.task_id,
            "compile_rate": compile_rate,
            "correct_rate": correct_rate,
            "avg_reward": avg_reward,
            "rewards": rewards,
            "advantages": advantages,
            "policy_loss": policy_loss.item(),
            "gen_time_s": gen_time,
            "eval_time_s": eval_time,
            "update_time_s": update_time,
            "total_time_s": total_step_time,
            "eval_results": eval_results,
        }
        history.append(step_record)

        print(f"  compile={compile_rate:.0%} | correct={correct_rate:.0%} | "
              f"avg_reward={avg_reward:.3f} | loss={policy_loss.item():.4f}")
        print(f"  time: gen={gen_time:.1f}s eval={eval_time:.1f}s update={update_time:.1f}s total={total_step_time:.1f}s\n")

    # --- Save results ---
    out_path = cfg.output_dir / "training_history.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nTraining history saved to {out_path}")

    lora_path = cfg.output_dir / "lora_weights"
    lora_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"LoRA weights saved to {lora_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for rec in history:
        print(f"  Step {rec['step']:2d} | {rec['task_id']:8s} | "
              f"compile={rec['compile_rate']:.0%} correct={rec['correct_rate']:.0%} "
              f"reward={rec['avg_reward']:+.3f} loss={rec['policy_loss']:.4f}")

    total_compile = sum(r["compile_rate"] for r in history) / len(history)
    total_correct = sum(r["correct_rate"] for r in history) / len(history)
    total_reward = sum(r["avg_reward"] for r in history) / len(history)
    print(f"\n  Overall: compile={total_compile:.0%} correct={total_correct:.0%} reward={total_reward:+.3f}")
    print(f"  Total wall time: {sum(r['total_time_s'] for r in history):.1f}s")
    print(f"\nSmoke RL online test COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
