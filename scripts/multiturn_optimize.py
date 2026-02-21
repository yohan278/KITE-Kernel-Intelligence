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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/multiturn"))
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--generation-mode", default="stub", choices=["stub", "local", "kernelbench_server"])
    args = parser.parse_args()

    adapter = KernelBenchAdapter(args.kernelbench_root)
    tasks = adapter.discover_tasks()[: args.num_tasks]
    policy = QwenPolicy(QwenPolicyConfig(generation_mode=args.generation_mode))
    agent = LLMKernelAgent(policy)
    env = KernelBenchEnergyEnv(
        adapter=adapter,
        measurement_config=MeasurementConfig(warmup_iters=2, measure_iters=8, repeats=1),
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

    for task in tasks:
        def _eval(code: str) -> dict:
            step = env.evaluate(task=task, code=code)
            return {
                "compile_ok": step.compile_ok,
                "correct": step.correct,
                "reward": step.reward.total,
                "runtime_ms": step.runtime_ms,
                "joules": step.joules,
            }

        result = agent.optimize_task(task=task, evaluate_fn=_eval, max_turns=args.max_turns)
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

    summary["avg_turns_to_success"] = (sum(turns) / len(turns)) if turns else 0.0
    summary["pass_at_k"] = summary["pass_at_k_count"] / max(1, len(tasks))
    save_json(args.output_dir / "summary.json", summary)
    print(f"Saved trajectories to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
