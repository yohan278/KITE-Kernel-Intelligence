#!/usr/bin/env python3
"""Phase 0 smoke test: one task end-to-end with runtime + power + energy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from kite.utils.serialization import load_yaml, save_json, save_jsonl


def _choose_task(adapter: KernelBenchAdapter, task_id: str | None, task_index: int):
    tasks = adapter.discover_tasks()
    if not tasks:
        raise RuntimeError("No KernelBench tasks discovered")
    if task_id:
        for task in tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Task not found: {task_id}")
    idx = max(0, min(task_index, len(tasks) - 1))
    return tasks[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    kernelbench_root = Path(cfg.get("kernelbench_root", "./external/KernelBench"))
    output_jsonl = Path(cfg.get("output_jsonl", "./results/measurement/smoke_one_task.jsonl"))
    power_trace_dir = Path(cfg.get("power_trace_dir", "./results/measurement/power_traces"))
    task_id = cfg.get("task_id")
    task_index = int(cfg.get("task_index", 0))
    baseline_runtime_ms = float(cfg.get("baseline_runtime_ms", 100.0))

    measurement_cfg = MeasurementConfig(
        warmup_iters=int(cfg.get("warmup_iters", 3)),
        measure_iters=int(cfg.get("measure_iters", 10)),
        repeats=int(cfg.get("repeats", 1)),
        sampling_interval_ms=float(cfg.get("sampling_interval_ms", 50.0)),
    )
    reward_cfg = IPWRewardConfig()
    sla_latency_s = float(cfg.get("sla_latency_s", 1.0))

    adapter = KernelBenchAdapter(kernelbench_root)
    env = KernelBenchEnergyEnv(
        adapter=adapter,
        measurement_config=measurement_cfg,
        reward_config=reward_cfg,
        sla_latency_s=sla_latency_s,
    )

    task = _choose_task(adapter, task_id=task_id, task_index=task_index)
    code = task.reference_kernel or "def kernel(*args, **kwargs):\n    return args[0] if args else None\n"
    result = env.evaluate(task=task, code=code, baseline_runtime_ms=baseline_runtime_ms)

    power_trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = power_trace_dir / f"{task.task_id}_trace.json"
    trace_payload = {
        "task_id": task.task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": result.logs.get("power_trace", []),
    }
    save_json(trace_path, trace_payload)

    row = {
        "task_id": task.task_id,
        "compile_ok": result.compile_ok,
        "correct": result.correct,
        "runtime_ms": result.runtime_ms,
        "avg_watts": result.avg_power_w,
        "joules": result.joules,
        "speedup_vs_baseline": result.speedup,
        "logs_path": str(trace_path),
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_jsonl, [row])
    print(f"Wrote smoke result to {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
