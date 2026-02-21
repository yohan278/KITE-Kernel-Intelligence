#!/usr/bin/env python3
"""Evaluate one candidate kernel on one task and return metrics JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.envs.kernelbench_env_energy import KernelBenchEnergyEnv
from kite.measurement.protocol import MeasurementConfig
from kite.rewards.ipw_reward import IPWRewardConfig
from kite.utils.serialization import save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Kernel task id, e.g. L1_1")
    parser.add_argument("--code", type=Path, required=True, help="Path to candidate kernel file")
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output", type=Path, default=Path("./results/measurement/eval_candidate.json"))
    parser.add_argument("--sla-latency-s", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.25)
    args = parser.parse_args()

    adapter = KernelBenchAdapter(args.kernelbench_root)
    tasks = adapter.discover_tasks()
    task = next((t for t in tasks if t.task_id == args.task), None)
    if task is None:
        # Fallback for environments where KernelBench dataset API is unavailable.
        # Supports commands like `--task L1_1` by selecting the first level-matched task.
        if "_" in args.task and args.task.startswith("L"):
            level_str, _, ordinal_str = args.task.partition("_")
            try:
                level = int(level_str[1:])
                ordinal = max(1, int(ordinal_str))
                level_tasks = [t for t in tasks if t.level == level]
                if level_tasks:
                    task = level_tasks[min(len(level_tasks), ordinal) - 1]
            except ValueError:
                task = None
    if task is None:
        if not tasks:
            raise ValueError("No tasks available to evaluate")
        task = tasks[0]

    code = args.code.read_text()

    env = KernelBenchEnergyEnv(
        adapter=adapter,
        measurement_config=MeasurementConfig(warmup_iters=3, measure_iters=10, repeats=1),
        reward_config=IPWRewardConfig(
            alpha_speedup=args.alpha,
            beta_joules=args.beta,
            gamma_latency=args.gamma,
        ),
        sla_latency_s=args.sla_latency_s,
    )
    out = env.evaluate(task=task, code=code)

    payload = {
        "task_id": out.task_id,
        "compile_ok": out.compile_ok,
        "correct": out.correct,
        "runtime_ms": out.runtime_ms,
        "speedup": out.speedup,
        "avg_power_w": out.avg_power_w,
        "joules": out.joules,
        "reward": out.reward.total,
        "reward_breakdown": {
            "correctness": out.reward.correctness,
            "performance": out.reward.performance,
            "energy": out.reward.energy,
            "latency_sla": out.reward.latency_sla,
            "stability": out.reward.stability,
            "total": out.reward.total,
        },
        "logs": out.logs,
    }
    save_json(args.output, payload)
    print(args.output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
