#!/usr/bin/env python3
"""Launch Phase 3 multiturn optimization across multiple GPUs by sharding tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def _parse_gpus(raw: str) -> list[str]:
    gpus = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpus:
        raise ValueError("No GPU ids provided. Use --gpus '0,1,2,3'.")
    return gpus


def _slice_bounds(num_tasks: int, num_shards: int, shard_idx: int) -> tuple[int, int]:
    start = shard_idx * num_tasks // num_shards
    end = (shard_idx + 1) * num_tasks // num_shards
    return start, end


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/multiturn_l40_multi"))
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker shards to run. Defaults to len(--gpus).",
    )
    parser.add_argument(
        "--multiturn-script",
        type=Path,
        default=ROOT / "scripts" / "multiturn_optimize.py",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Status poll interval while child shards are running.",
    )
    args, passthrough = parser.parse_known_args()

    if not args.multiturn_script.exists():
        raise FileNotFoundError(f"multiturn script not found: {args.multiturn_script}")
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")

    gpu_ids = _parse_gpus(args.gpus)
    max_workers = len(gpu_ids)
    workers = max_workers if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be > 0")
    if workers > max_workers:
        raise ValueError(f"--workers={workers} exceeds available gpu ids ({max_workers})")
    if workers > args.num_tasks:
        workers = args.num_tasks

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    shard_meta: list[dict] = []
    procs: list[tuple[subprocess.Popen, Path, object]] = []

    print(f"[multi-gpu] launching {workers} shard(s) over {args.num_tasks} tasks", flush=True)
    for shard_idx in range(workers):
        start_idx, end_idx = _slice_bounds(args.num_tasks, workers, shard_idx)
        gpu = gpu_ids[shard_idx]
        shard_out = output_dir / f"shard_{shard_idx:02d}_gpu{gpu}"
        shard_out.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"shard_{shard_idx:02d}_gpu{gpu}.log"
        log_fh = open(log_path, "w", encoding="utf-8")

        cmd = [
            sys.executable,
            str(args.multiturn_script),
            "--kernelbench-root",
            str(args.kernelbench_root),
            "--output-dir",
            str(shard_out),
            "--num-tasks",
            str(args.num_tasks),
            "--task-start-index",
            str(start_idx),
            "--task-end-index",
            str(end_idx),
        ] + passthrough

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
        )
        shard_meta.append(
            {
                "shard": shard_idx,
                "gpu": gpu,
                "start_index": start_idx,
                "end_index": end_idx,
                "output_dir": str(shard_out),
                "log_path": str(log_path),
                "pid": proc.pid,
            }
        )
        procs.append((proc, log_path, log_fh))
        print(
            f"[multi-gpu] shard={shard_idx} gpu={gpu} tasks=[{start_idx}:{end_idx}) pid={proc.pid}",
            flush=True,
        )

    while True:
        still_running = 0
        for proc, _, _ in procs:
            if proc.poll() is None:
                still_running += 1
        if still_running == 0:
            break
        print(f"[multi-gpu] running shards: {still_running}/{len(procs)}", flush=True)
        time.sleep(max(0.2, args.poll_seconds))

    for _, _, log_fh in procs:
        try:
            log_fh.close()
        except Exception:
            pass

    merged_task_results: list[dict] = []
    pass_count = 0
    total_tasks = 0
    total_turns_to_success = 0.0
    successful_task_count = 0
    failed_shards = 0

    for meta, (proc, _, _) in zip(shard_meta, procs):
        rc = int(proc.returncode if proc.returncode is not None else -1)
        meta["return_code"] = rc
        shard_summary_path = Path(meta["output_dir"]) / "summary.json"
        meta["summary_path"] = str(shard_summary_path)
        summary = _load_json(shard_summary_path) if rc == 0 else {}
        meta["summary"] = summary
        if rc != 0:
            failed_shards += 1
            continue

        task_results = summary.get("task_results", [])
        if isinstance(task_results, list):
            merged_task_results.extend(task_results)
        shard_num_tasks = int(summary.get("num_tasks", 0))
        shard_pass = int(summary.get("pass_at_k_count", 0))
        shard_avg_turns = float(summary.get("avg_turns_to_success", 0.0))

        total_tasks += shard_num_tasks
        pass_count += shard_pass
        successful_task_count += shard_pass
        total_turns_to_success += shard_avg_turns * shard_pass

    merged = {
        "num_tasks": total_tasks,
        "pass_at_k_count": pass_count,
        "pass_at_k": (pass_count / total_tasks) if total_tasks > 0 else 0.0,
        "avg_turns_to_success": (
            total_turns_to_success / successful_task_count if successful_task_count > 0 else 0.0
        ),
        "failed_shards": failed_shards,
        "workers": workers,
        "gpus": gpu_ids[:workers],
        "task_results": merged_task_results,
        "shards": shard_meta,
    }
    merged_path = output_dir / "summary_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    print(f"[multi-gpu] merged summary: {merged_path}", flush=True)
    if failed_shards > 0:
        print(f"[multi-gpu] warning: {failed_shards} shard(s) failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
