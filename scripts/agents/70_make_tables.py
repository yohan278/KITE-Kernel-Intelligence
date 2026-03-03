#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    stats_path = args.root / "outputs" / "agent_queue" / "stats_summary.json"
    out_dir = args.root / "outputs" / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "stage_metrics.md"

    if not stats_path.exists():
        out_md.write_text("No stats available yet.\n", encoding="utf-8")
        print(f"[tb0] wrote placeholder {out_md}")
        return 0

    data = json.loads(stats_path.read_text(encoding="utf-8"))
    lines = []
    lines.append("| stage | metric | n | mean | std | ci95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for stage in sorted(data.keys()):
        for metric, stats in sorted(data[stage].items()):
            lines.append(
                "| {stage} | {metric} | {n} | {mean:.6f} | {std:.6f} | {ci95:.6f} |".format(
                    stage=stage,
                    metric=metric,
                    n=int(stats.get("n", 0)),
                    mean=float(stats.get("mean", 0.0)),
                    std=float(stats.get("std", 0.0)),
                    ci95=float(stats.get("ci95", 0.0)),
                )
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[tb0] wrote {out_md}")

    state = args.root / "outputs" / "agent_queue" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "tb0.done").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
