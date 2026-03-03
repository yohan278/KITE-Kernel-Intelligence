#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _fail(msg: str) -> int:
    print(f"[pl0] error: {msg}", file=sys.stderr)
    return 1


def _stage_sort_key(stage: str) -> tuple[int, int, str]:
    match = re.search(r"(\d+)(?!.*\d)", stage)
    if match:
        return (0, int(match.group(1)), stage)
    return (1, 0, stage)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    out_dir = args.root / "outputs" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try using existing project plot script first.
    script = args.root / "scripts" / "09_plot_pareto.py"
    if script.exists():
        result = subprocess.run(["python", str(script)], cwd=str(args.root), check=False)
        if result.returncode != 0:
            print(
                f"[pl0] warning: optional {script} exited with code {result.returncode}",
                file=sys.stderr,
            )

    stats_path = args.root / "outputs" / "agent_queue" / "stats_summary.json"
    if not stats_path.exists():
        return _fail(f"required stats not found: {stats_path}")

    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _fail(f"invalid JSON in {stats_path}: {exc}")

    if not isinstance(data, dict) or not data:
        return _fail(f"{stats_path} is empty or not keyed by stage")

    stages = sorted((str(stage) for stage in data.keys()), key=_stage_sort_key)
    means: list[float] = []
    for stage in stages:
        metric_block = data.get(stage, {})
        avg_reward = metric_block.get("avg_reward", {}) if isinstance(metric_block, dict) else {}
        mean_val = avg_reward.get("mean", 0.0) if isinstance(avg_reward, dict) else 0.0
        try:
            means.append(float(mean_val))
        except (TypeError, ValueError):
            means.append(0.0)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return _fail(f"matplotlib not available: {exc}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stages, means)
    ax.set_title("Avg Reward by Stage")
    ax.set_ylabel("avg_reward")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    png = out_dir / "avg_reward_by_stage.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)

    if not png.exists() or png.stat().st_size == 0:
        return _fail(f"expected figure missing or empty: {png}")

    print(f"[pl0] wrote {png}")

    state = args.root / "outputs" / "agent_queue" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "pl0.done").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
