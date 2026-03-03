#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    out_dir = args.root / "outputs" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try using existing project plot script first.
    script = args.root / "scripts" / "09_plot_pareto.py"
    if script.exists():
        import subprocess

        subprocess.run(["python", str(script)], cwd=str(args.root), check=False)

    stats_path = args.root / "outputs" / "agent_queue" / "stats_summary.json"
    if not stats_path.exists():
        print(f"[pl0] stats not found: {stats_path}")
        return 0

    data = json.loads(stats_path.read_text(encoding="utf-8"))
    stages = sorted(data.keys())
    means = [float(data[s].get("avg_reward", {}).get("mean", 0.0)) for s in stages]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[pl0] matplotlib not available; skipping plot generation")
        return 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stages, means)
    ax.set_title("Avg Reward by Stage")
    ax.set_ylabel("avg_reward")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    png = out_dir / "avg_reward_by_stage.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[pl0] wrote {png}")

    state = args.root / "outputs" / "agent_queue" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "pl0.done").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
