#!/usr/bin/env python3
"""Generate Pareto frontier plots from suite results."""

from pathlib import Path

from kite.eval.reports import pareto_frontier, plot_pareto_frontiers
from kite.utils.serialization import load_json


def main() -> int:
    path = Path("outputs/eval/suite_results.json")
    if not path.exists():
        print("Missing outputs/eval/suite_results.json. Run scripts/08_eval_all.py first.")
        return 1

    suite = load_json(path)
    results = suite.get("results", [])

    points = []
    for row in results:
        metrics = row.get("metrics", {})
        points.append({
            "id": row.get("id", "unknown"),
            "apj": float(metrics.get("apj", 0.0)),
            "apw": float(metrics.get("apw", 0.0)),
            "throughput_tps": float(metrics.get("throughput_tps", metrics.get("speedup", 1.0) * 100)),
            "energy_per_token_j": float(metrics.get("energy_per_token_j", 0.25)),
            "ttft_p95_s": float(metrics.get("ttft_p95_s", 2.0)),
        })

    frontier = pareto_frontier(points, "apj", "apw")
    print("Pareto frontier (apj, apw):")
    for p in frontier:
        print(f"  {p['id']}: apj={p['apj']:.4f}, apw={p['apw']:.4f}")

    output_dir = Path("outputs/eval")
    plots = plot_pareto_frontiers(results, output_dir)
    if plots:
        print(f"\nPlots saved:")
        for name, fpath in plots.items():
            print(f"  {name}: {fpath}")
    else:
        print("\nNo plots generated (matplotlib may not be installed).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
