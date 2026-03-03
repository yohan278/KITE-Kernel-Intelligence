"""Post-training evaluation: generate optimized kernels and compare to baseline.

Loads a trained LoRA checkpoint, generates kernels for each task, evaluates
them with KernelBench + NVML energy sampling, and compares against the
baseline energy profile.

    python -m kite.eval.post_training_eval \
        --checkpoint-dir checkpoints/grpo_12h_l1l2_energy_YYYYMMDD_HHMM \
        --baseline data/baseline_energy_profile.json \
        --kernelbench-root external/KernelBench \
        --output data/post_training_results.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _generate_kernel(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single kernel completion."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def run_post_training_eval(
    checkpoint_dir: Path,
    baseline_path: Path | None,
    kernelbench_root: Path,
    output_path: Path,
    levels: list[int] | None = None,
    num_generations: int = 4,
    max_new_tokens: int = 512,
    eval_timeout_s: float = 120.0,
) -> dict:
    """Run full post-training evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from kite.adapters.kernelbench_adapter import KernelBenchAdapter
    from kite.classification.inference_profile import (
        energy_weight_for_type,
        inference_relevance_weight,
        is_inference_critical,
    )
    from kite.classification.prompt_hints import build_energy_aware_prompt
    from kite.policies.qwen_policy import QwenPolicy, QwenPolicyConfig
    from kite.utils.serialization import load_json, save_json

    lora_dir = checkpoint_dir / "lora_weights"
    checkpoint_json = checkpoint_dir / "checkpoint.json"

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA weights not found at {lora_dir}")

    baseline: dict[str, dict] = {}
    if baseline_path and baseline_path.exists():
        baseline = load_json(baseline_path)
        print(f"Loaded baseline profile: {len(baseline)} tasks")

    print("Loading model + LoRA weights ...")
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.merge_and_unload()
    model = model.to("cuda:0")
    model.eval()
    print("Model loaded.")

    policy = QwenPolicy(QwenPolicyConfig())
    adapter = KernelBenchAdapter(
        kernelbench_root=kernelbench_root,
        enable_kernelbench_eval=True,
        num_correct_trials=1,
        num_perf_trials=3,
        levels=levels,
    )
    tasks = adapter.discover_tasks()
    print(f"Evaluating {len(tasks)} tasks with {num_generations} generations each")

    results: list[dict[str, Any]] = []

    for i, task in enumerate(tasks):
        ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
        prompt = build_energy_aware_prompt(ref_src, kernel_type=task.kernel_type)
        bl = baseline.get(task.task_id, {})

        print(
            f"  [{i+1}/{len(tasks)}] {task.task_id} ({task.kernel_type}) ...",
            end=" ", flush=True,
        )

        best_candidate = None
        best_reward = float("-inf")

        for g in range(num_generations):
            raw = _generate_kernel(model, tokenizer, prompt, max_new_tokens)
            code = policy.extract_code(raw).strip()
            if not code:
                continue

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(adapter.evaluate_candidate, task, code)
                try:
                    cand = fut.result(timeout=eval_timeout_s)
                except concurrent.futures.TimeoutError:
                    continue

            if cand.correct and cand.speedup is not None:
                score = cand.speedup
                if score > best_reward:
                    best_reward = score
                    best_candidate = cand

        if best_candidate is None:
            print("no valid kernel")
            results.append({
                "task_id": task.task_id,
                "kernel_type": task.kernel_type,
                "level": task.level,
                "inference_critical": is_inference_critical(task),
                "success": False,
            })
            continue

        c = best_candidate
        entry: dict[str, Any] = {
            "task_id": task.task_id,
            "kernel_type": task.kernel_type,
            "level": task.level,
            "inference_critical": is_inference_critical(task),
            "energy_weight": energy_weight_for_type(task.kernel_type),
            "success": True,
            "runtime_ms": c.runtime_ms,
            "speedup": c.speedup,
            "energy_j": c.logs.get("energy_j"),
            "avg_power_w": c.logs.get("avg_power_w"),
            "avg_gpu_util_pct": c.logs.get("avg_gpu_util_pct"),
            "avg_mem_util_pct": c.logs.get("avg_mem_util_pct"),
        }

        bl_energy = bl.get("energy_j")
        bl_runtime = bl.get("runtime_ms")
        cand_energy = c.logs.get("energy_j")

        if bl_energy and cand_energy and bl_energy > 0:
            entry["energy_savings_pct"] = (bl_energy - cand_energy) / bl_energy * 100
        if bl_runtime and c.runtime_ms and bl_runtime > 0:
            entry["runtime_delta_pct"] = (c.runtime_ms - bl_runtime) / bl_runtime * 100

        results.append(entry)

        es = entry.get("energy_savings_pct")
        rd = entry.get("runtime_delta_pct")
        parts = [f"speedup={c.speedup:.2f}x"]
        if es is not None:
            parts.append(f"energy_save={es:+.1f}%")
        if rd is not None:
            parts.append(f"runtime_delta={rd:+.1f}%")
        print("  ".join(parts))

    save_json(output_path, {"results": results})
    print(f"\nResults saved to {output_path}")

    _print_summary(results)
    return {"results": results}


def _print_summary(results: list[dict]) -> None:
    """Print per-kernel-type summary table."""
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["kernel_type"]].append(r)

    print("\n" + "=" * 80)
    print("POST-TRAINING EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Kernel Type':<16} {'Count':>5} {'Success':>7} "
        f"{'Avg Speedup':>11} {'Avg Energy Save':>15} {'Avg RT Delta':>12}"
    )
    print("-" * 80)

    for kt in sorted(by_type.keys()):
        rows = by_type[kt]
        n = len(rows)
        ok = sum(1 for r in rows if r.get("success"))
        speedups = [r["speedup"] for r in rows if r.get("speedup")]
        e_saves = [r["energy_savings_pct"] for r in rows if r.get("energy_savings_pct") is not None]
        rt_deltas = [r["runtime_delta_pct"] for r in rows if r.get("runtime_delta_pct") is not None]

        avg_sp = f"{sum(speedups)/len(speedups):.2f}x" if speedups else "n/a"
        avg_es = f"{sum(e_saves)/len(e_saves):+.1f}%" if e_saves else "n/a"
        avg_rd = f"{sum(rt_deltas)/len(rt_deltas):+.1f}%" if rt_deltas else "n/a"

        print(f"{kt:<16} {n:>5} {ok:>7} {avg_sp:>11} {avg_es:>15} {avg_rd:>12}")

    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-training kernel evaluation")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--kernelbench-root", type=Path, default=Path("external/KernelBench"))
    parser.add_argument("--output", type=Path, default=Path("data/post_training_results.json"))
    parser.add_argument("--levels", type=str, default="1,2")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    run_post_training_eval(
        checkpoint_dir=args.checkpoint_dir,
        baseline_path=args.baseline,
        kernelbench_root=args.kernelbench_root,
        output_path=args.output,
        levels=levels,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        eval_timeout_s=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
