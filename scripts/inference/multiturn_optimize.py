#!/usr/bin/env python3
"""Phase 3 Kevin-style multi-turn optimization (no training)."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
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


def _forward_arity_from_source(source: str, class_name: str) -> int | None:
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    positional = [arg.arg for arg in item.args.args if arg.arg != "self"]
                    return len(positional)
    return None


def _precheck_candidate_code(
    task,
    code: str,
    allow_triton: bool,
    block_custom_autograd: bool = False,
) -> str | None:
    if not code or not code.strip():
        return "precheck: empty completion"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"precheck: syntax_error: {exc}"

    if "class ModelNew" not in code:
        return "precheck: missing ModelNew class"
    cand_arity = _forward_arity_from_source(code, "ModelNew")
    if cand_arity is None:
        return "precheck: missing ModelNew.forward method"

    ref_src = str(task.metadata.get("ref_arch_src", "") or "")
    ref_arity = _forward_arity_from_source(ref_src, "Model")
    if ref_arity is not None and cand_arity != ref_arity:
        return f"precheck: forward_arity_mismatch ref={ref_arity} candidate={cand_arity}"

    # Guard rails against slow/hanging compile paths during multiturn evaluation.
    blocked_patterns = [
        ("torch.utils.cpp_extension", "precheck: disallowed cpp_extension usage"),
        ("load_inline(", "precheck: disallowed inline extension compilation"),
    ]
    if block_custom_autograd:
        blocked_patterns.extend(
            [
                ("from torch.autograd import Function", "precheck: disallowed custom autograd Function"),
                ("@custom_fwd", "precheck: disallowed custom_fwd decorator"),
                ("@custom_bwd", "precheck: disallowed custom_bwd decorator"),
            ]
        )
    if not allow_triton:
        blocked_patterns.extend(
            [
                ("import triton", "precheck: triton_not_allowed"),
                ("@triton", "precheck: triton_not_allowed"),
            ]
        )
    for needle, reason in blocked_patterns:
        if needle in code:
            return reason
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/multiturn_l40"))
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument(
        "--task-start-index",
        type=int,
        default=0,
        help="Start index (inclusive) within the first --num-tasks discovered tasks.",
    )
    parser.add_argument(
        "--task-end-index",
        type=int,
        default=None,
        help="End index (exclusive) within the first --num-tasks discovered tasks.",
    )
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--generation-mode", default="local", choices=["stub", "local", "kernelbench_server"])
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--allow-triton", action="store_true")
    parser.add_argument(
        "--block-custom-autograd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally block custom autograd decorators/classes during precheck.",
    )
    default_hf_cache = os.environ.get("KITE_HF_CACHE")
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path(default_hf_cache) if default_hf_cache else None,
    )
    default_local_only_env = os.environ.get("KITE_HF_LOCAL_FILES_ONLY", "").strip().lower()
    default_local_only = (
        default_local_only_env in {"1", "true", "yes", "on"} if default_local_only_env else True
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=default_local_only,
        help="Use only local HF cache files. Disable via --no-local-files-only.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")
    parser.add_argument(
        "--verbose-turns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-turn metrics for each task. Disable via --no-verbose-turns.",
    )
    parser.add_argument(
        "--turn-timeout-seconds",
        type=float,
        default=60.0,
        help="Max seconds for each evaluate/generate action within a turn; 0 disables timeout.",
    )
    parser.add_argument(
        "--debug-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print prompt/output snippets for each multiturn repair step. Disable via --no-debug-generation.",
    )
    parser.add_argument(
        "--debug-max-chars",
        type=int,
        default=600,
        help="Max chars to print/store per debug text field.",
    )
    args = parser.parse_args()

    adapter = KernelBenchAdapter(
        args.kernelbench_root,
        num_correct_trials=max(1, args.num_correct_trials),
        num_perf_trials=max(1, args.num_perf_trials),
    )
    discovered = adapter.discover_tasks()[: max(0, args.num_tasks)]
    start_idx = max(0, int(args.task_start_index))
    raw_end_idx = len(discovered) if args.task_end_index is None else int(args.task_end_index)
    end_idx = min(max(start_idx, raw_end_idx), len(discovered))
    tasks = discovered[start_idx:end_idx]
    print(
        f"[multiturn] discovered={len(discovered)} selected={len(tasks)} "
        f"range=[{start_idx}:{end_idx})",
        flush=True,
    )

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
        "task_start_index": start_idx,
        "task_end_index": end_idx,
        "discovered_tasks": len(discovered),
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
        debug_events: list[dict[str, Any]] = []

        def _eval(code: str) -> dict:
            precheck_error = _precheck_candidate_code(
                task=task,
                code=code,
                allow_triton=bool(args.allow_triton),
                block_custom_autograd=bool(args.block_custom_autograd),
            )
            if precheck_error:
                return {
                    "compile_ok": False,
                    "correct": False,
                    "reward": -1.0,
                    "runtime_ms": 0.0,
                    "speedup": 0.0,
                    "joules": 0.0,
                    "compile_log": precheck_error,
                    "correctness_log": None,
                }

            step = env.evaluate(task=task, code=code)
            candidate_logs = step.logs.get("candidate_logs", {})
            metadata = candidate_logs.get("metadata", {}) if isinstance(candidate_logs, dict) else {}
            compile_log = step.logs.get("compile_log")
            correctness_log = step.logs.get("correctness_log")

            if not compile_log and isinstance(metadata, dict):
                compile_log = (
                    metadata.get("compilation_error")
                    or metadata.get("compile_error")
                    or metadata.get("error")
                )
            if not correctness_log and isinstance(metadata, dict):
                correctness_log = metadata.get("correctness_error")

            return {
                "compile_ok": step.compile_ok,
                "correct": step.correct,
                "reward": step.reward.total,
                "runtime_ms": step.runtime_ms,
                "speedup": step.speedup,
                "joules": step.joules,
                "compile_log": compile_log,
                "correctness_log": correctness_log,
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
                    f"reward={step.reward:.4f} runtime_ms={step.runtime_ms:.4f} joules={step.joules:.6f}"
                    f" timeout={step.timeout}",
                    flush=True,
                )
                if step.error:
                    print(f"[multiturn] task={task.task_id} turn={step.turn} error={step.error}", flush=True)

        def _trim(value: object) -> str:
            text = str(value)
            if args.debug_max_chars <= 0:
                return text
            if len(text) <= args.debug_max_chars:
                return text
            return text[: args.debug_max_chars] + "\n...<truncated>"

        def _on_event(event: dict[str, Any]) -> None:
            if not args.debug_generation:
                return
            copied: dict[str, Any] = dict(event)
            for key in ("prompt", "raw_output", "extracted_code", "code", "feedback", "error"):
                if key in copied and copied[key] is not None:
                    copied[key] = _trim(copied[key])
            debug_events.append(copied)
            name = str(copied.get("event", "event"))
            turn = copied.get("turn", "?")
            print(f"[multiturn-debug] task={task.task_id} turn={turn} event={name}", flush=True)
            if "feedback" in copied:
                print(f"[multiturn-debug] feedback:\n{copied['feedback']}", flush=True)
            if "prompt" in copied:
                print(f"[multiturn-debug] prompt:\n{copied['prompt']}", flush=True)
            if "raw_output" in copied:
                print(f"[multiturn-debug] raw_output:\n{copied['raw_output']}", flush=True)
            if "extracted_code" in copied:
                print(f"[multiturn-debug] extracted_code:\n{copied['extracted_code']}", flush=True)
            if "error" in copied:
                print(f"[multiturn-debug] error: {copied['error']}", flush=True)

        result = agent.optimize_task(
            task=task,
            evaluate_fn=_eval,
            max_turns=args.max_turns,
            turn_timeout_seconds=max(0.0, args.turn_timeout_seconds),
            on_step=_on_step,
            on_event=_on_event if args.debug_generation else None,
        )
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
                    "timeout": s.timeout,
                    "error": s.error,
                    "code": s.code,
                }
                for s in result.steps
            ],
        }
        if args.debug_generation:
            task_payload["debug_events"] = debug_events
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
