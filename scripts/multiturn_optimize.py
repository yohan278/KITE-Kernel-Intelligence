#!/usr/bin/env python3
"""Phase 3 Kevin-style multi-turn optimization (no training)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.agents.llm_agent import LLMKernelAgent
from kite.envs.kernelbench_env_energy import KernelBenchEnergyEnv
from kite.measurement.protocol import MeasurementConfig
from kite.policies.qwen_policy import QwenPolicy, QwenPolicyConfig
from kite.rewards.ipw_reward import IPWRewardConfig
from kite.utils.serialization import save_json

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/multiturn"))
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--num-correct-trials", type=int, default=3)
    parser.add_argument("--num-perf-trials", type=int, default=25)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=8)
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--generation-mode", default="stub", choices=["stub", "local", "kernelbench_server"])
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--allow-triton", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")
    parser.add_argument("--verbose-turns", action="store_true", help="Print per-turn metrics for each task")
    args = parser.parse_args()

    adapter = KernelBenchAdapter(
        args.kernelbench_root,
        num_correct_trials=max(1, args.num_correct_trials),
        num_perf_trials=max(1, args.num_perf_trials),
    )
    tasks = adapter.discover_tasks()[: args.num_tasks]
    print(f"[multiturn] discovered {len(tasks)} tasks", flush=True)

    print(f"[multiturn] initializing policy (mode={args.generation_mode})", flush=True)
    policy = QwenPolicy(
        QwenPolicyConfig(
            generation_mode=args.generation_mode,
            model_name=args.model_name,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            allow_triton=bool(args.allow_triton),
            kernelbench_root=args.kernelbench_root,
            hf_cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
            local_files_only=bool(args.local_files_only),
        )
    )
    print("[multiturn] policy initialized", flush=True)
    agent = LLMKernelAgent(policy)
    env = KernelBenchEnergyEnv(
        adapter=adapter,
        measurement_config=MeasurementConfig(
            warmup_iters=max(0, args.warmup_iters),
            measure_iters=max(1, args.measure_iters),
            repeats=max(1, args.measurement_repeats),
        ),
        reward_config=IPWRewardConfig(),
        sla_latency_s=1.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_tasks": len(tasks),
        "pass_at_k_count": 0,
        "avg_turns_to_success": 0.0,
        "task_results": [],
    }
    turns = []

    task_iter = tasks
    progress = None
    if tqdm is not None and not args.no_progress:
        progress = tqdm(tasks, total=len(tasks), desc="multiturn tasks", dynamic_ncols=True)
        task_iter = progress

    for idx, task in enumerate(task_iter, start=1):
        print(f"[multiturn] [{idx}/{len(tasks)}] start task={task.task_id}", flush=True)

        def _eval(code: str) -> dict:
            step = env.evaluate(task=task, code=code)
            return {
                "compile_ok": step.compile_ok,
                "correct": step.correct,
                "reward": step.reward.total,
                "runtime_ms": step.runtime_ms,
                "joules": step.joules,
            }

        def _on_step(step) -> None:
            if progress is not None:
                progress.set_postfix(
                    task=task.task_id,
                    turn=step.turn,
                    correct=int(step.correct),
                    reward=f"{step.reward:.3f}",
                )
            if args.verbose_turns:
                print(
                    f"[multiturn] task={task.task_id} turn={step.turn} "
                    f"compile_ok={step.compile_ok} correct={step.correct} "
                    f"reward={step.reward:.4f} runtime_ms={step.runtime_ms:.4f} joules={step.joules:.6f}",
                    flush=True,
                )

        result = agent.optimize_task(task=task, evaluate_fn=_eval, max_turns=args.max_turns, on_step=_on_step)
        summary["pass_at_k_count"] += int(result.pass_at_k)
        if result.turns_to_success > 0:
            turns.append(result.turns_to_success)

        task_payload = {
            "task_id": result.task_id,
            "pass_at_k": result.pass_at_k,
            "turns_to_success": result.turns_to_success,
            "steps": [
                {
                    "turn": s.turn,
                    "compile_ok": s.compile_ok,
                    "correct": s.correct,
                    "reward": s.reward,
                    "runtime_ms": s.runtime_ms,
                    "joules": s.joules,
                    "code": s.code,
                }
                for s in result.steps
            ],
        }
        save_json(args.output_dir / f"{task.task_id}.json", task_payload)
        summary["task_results"].append(
            {
                "task_id": result.task_id,
                "pass_at_k": result.pass_at_k,
                "turns_to_success": result.turns_to_success,
            }
        )
        print(
            f"[multiturn] [{idx}/{len(tasks)}] done task={task.task_id} "
            f"pass_at_k={result.pass_at_k} turns_to_success={result.turns_to_success}",
            flush=True,
        )

    summary["avg_turns_to_success"] = (sum(turns) / len(turns)) if turns else 0.0
    summary["pass_at_k"] = summary["pass_at_k_count"] / max(1, len(tasks))
    save_json(args.output_dir / "summary.json", summary)
    print(f"Saved trajectories to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
