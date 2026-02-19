#!/usr/bin/env python3
from pathlib import Path

from kite.eval.reports import pareto_frontier
from kite.utils.serialization import load_json


def main() -> int:
    path = Path("outputs/eval/suite_results.json")
    if not path.exists():
        print("Missing outputs/eval/suite_results.json. Run scripts/08_eval_all.py first.")
        return 1

    suite = load_json(path)
    points = []
    for row in suite.get("results", []):
        metrics = row.get("metrics", {})
        points.append({
            "id": row.get("id", "unknown"),
            "apj": float(metrics.get("apj", 0.0)),
            "apw": float(metrics.get("apw", 0.0)),
        })

    frontier = pareto_frontier(points, "apj", "apw")
    print("Pareto frontier (apj, apw):")
    for p in frontier:
        print(f"- {p['id']}: apj={p['apj']:.4f}, apw={p['apw']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
