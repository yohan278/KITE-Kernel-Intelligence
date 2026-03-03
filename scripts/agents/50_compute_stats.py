#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    in_path = args.root / "outputs" / "agent_queue" / "parsed_metrics.jsonl"
    out_dir = args.root / "outputs" / "agent_queue"
    out_json = out_dir / "stats_summary.json"

    if not in_path.exists():
        raise SystemExit(f"missing parsed metrics: {in_path}")

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = str(row.get("stage") or "unknown")
            for k in ("avg_reward", "train_loss"):
                v = row.get(k)
                if isinstance(v, (int, float)):
                    grouped[key][k].append(float(v))

    summary: dict[str, Any] = {}
    for stage, metrics in grouped.items():
        summary[stage] = {}
        for metric_name, values in metrics.items():
            m = _mean(values)
            s = _std(values)
            n = len(values)
            ci95 = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
            summary[stage][metric_name] = {
                "n": n,
                "mean": m,
                "std": s,
                "ci95": ci95,
            }

    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[st0] wrote stats to {out_json}")

    state = out_dir / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "st0.done").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
