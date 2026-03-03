"""Project per-kernel energy savings to end-to-end transformer inference.

Takes baseline_energy_profile.json and post_training_results.json (or
energy_by_kernel_type.json from training) and produces a projection:

    python -m kite.eval.inference_projection \
        --results data/post_training_results.json \
        --output data/inference_projection.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.classification.inference_profile import TRANSFORMER_ENERGY_WEIGHTS


def project_inference_savings(
    results: list[dict],
    energy_weights: dict[str, float] | None = None,
) -> dict:
    """Compute end-to-end transformer inference energy savings projection.

    Args:
        results: Per-task evaluation results with energy_savings_pct and
                 runtime_delta_pct fields.
        energy_weights: Kernel type to energy fraction mapping.
                        Defaults to TRANSFORMER_ENERGY_WEIGHTS.

    Returns:
        Projection dict with per-type and total savings.
    """
    weights = energy_weights or TRANSFORMER_ENERGY_WEIGHTS

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("success"):
            by_type[r["kernel_type"]].append(r)

    projection: dict[str, dict] = {}
    total_weighted_energy_savings = 0.0
    total_weighted_runtime_delta = 0.0
    total_weight_covered = 0.0

    for kt in sorted(weights.keys()):
        w = weights[kt]
        rows = by_type.get(kt, [])

        e_saves = [r["energy_savings_pct"] for r in rows if r.get("energy_savings_pct") is not None]
        rt_deltas = [r["runtime_delta_pct"] for r in rows if r.get("runtime_delta_pct") is not None]
        speedups = [r["speedup"] for r in rows if r.get("speedup") is not None]

        avg_energy_save = sum(e_saves) / len(e_saves) if e_saves else 0.0
        avg_rt_delta = sum(rt_deltas) / len(rt_deltas) if rt_deltas else 0.0
        avg_speedup = sum(speedups) / len(speedups) if speedups else None

        weighted_savings = w * avg_energy_save
        weighted_rt_delta = w * avg_rt_delta

        total_weighted_energy_savings += weighted_savings
        total_weighted_runtime_delta += weighted_rt_delta
        if rows:
            total_weight_covered += w

        projection[kt] = {
            "energy_share": w,
            "num_tasks_evaluated": len(rows),
            "num_with_energy_data": len(e_saves),
            "avg_energy_savings_pct": avg_energy_save,
            "avg_runtime_delta_pct": avg_rt_delta,
            "avg_speedup": avg_speedup,
            "weighted_energy_savings": weighted_savings,
            "weighted_runtime_delta": weighted_rt_delta,
        }

    summary = {
        "total_projected_energy_savings_pct": total_weighted_energy_savings,
        "total_projected_runtime_delta_pct": total_weighted_runtime_delta,
        "energy_budget_coverage": total_weight_covered,
        "per_kernel_type": projection,
    }
    return summary


def print_projection_report(projection: dict) -> None:
    """Print a formatted projection report."""
    per_type = projection["per_kernel_type"]

    print()
    print("=" * 90)
    print("TRANSFORMER INFERENCE ENERGY PROJECTION")
    print("=" * 90)
    print(
        f"{'Kernel Type':<16} {'Energy Share':>12} {'Tasks':>5} "
        f"{'Avg Savings':>11} {'Avg RT Delta':>12} {'Weighted Save':>13}"
    )
    print("-" * 90)

    for kt in sorted(per_type.keys(), key=lambda k: per_type[k]["energy_share"], reverse=True):
        d = per_type[kt]
        share = f"{d['energy_share']*100:.0f}%"
        n = d["num_tasks_evaluated"]
        avg_s = f"{d['avg_energy_savings_pct']:+.1f}%" if d["num_with_energy_data"] else "n/a"
        avg_r = f"{d['avg_runtime_delta_pct']:+.1f}%" if d["num_with_energy_data"] else "n/a"
        wt_s = f"{d['weighted_energy_savings']:+.2f}%"
        print(f"{kt:<16} {share:>12} {n:>5} {avg_s:>11} {avg_r:>12} {wt_s:>13}")

    print("-" * 90)
    total_e = projection["total_projected_energy_savings_pct"]
    total_r = projection["total_projected_runtime_delta_pct"]
    coverage = projection["energy_budget_coverage"]
    print(
        f"{'TOTAL PROJECTED':<16} {coverage*100:.0f}% covered"
        f"{'':>5} {'':>11} {total_r:>+11.1f}% {total_e:>+12.2f}%"
    )
    print("=" * 90)

    print(f"\nProjected end-to-end inference energy savings: {total_e:+.1f}%")
    print(f"Projected end-to-end runtime change: {total_r:+.1f}%")
    if abs(total_r) < 3.0 and total_e > 0:
        print(
            f"\nClaim: RL-optimized kernels achieve ~{total_e:.0f}% energy savings "
            f"for transformer inference at <{max(abs(total_r), 1.0):.0f}% runtime change."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Project inference energy savings")
    parser.add_argument("--results", type=Path, required=True,
                        help="post_training_results.json from post-training eval")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from kite.utils.serialization import load_json, save_json

    data = load_json(args.results)
    results = data if isinstance(data, list) else data.get("results", [])

    projection = project_inference_savings(results)
    print_projection_report(projection)

    if args.output:
        save_json(args.output, projection)
        print(f"\nProjection saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
