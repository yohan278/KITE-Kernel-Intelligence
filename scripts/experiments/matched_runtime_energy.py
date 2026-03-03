#!/usr/bin/env python3
"""Matched-runtime energy experiment: same kernel runtime, different energy.

Builds candidate pairs from M1/M2/M3 on the same KernelBench tasks,
filters to pairs with |delta_runtime| <= 3%, and analyzes energy gaps.
This is the "energy-aware RL matters" claim's main supporting evidence.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def parse_args():
    p = argparse.ArgumentParser(description="Matched-runtime energy analysis")
    p.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "h100" / "2026-03")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--runtime-tolerance", type=float, default=0.03, help="Max relative runtime gap (default 3%%)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_per_task_metrics(results_root: Path, exp_name: str) -> list[dict]:
    """Load per-task metrics CSV for an experiment."""
    path = results_root / exp_name / f"{exp_name}_metrics.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def build_task_index(rows: list[dict]) -> dict[str, list[dict]]:
    """Index rows by task_id."""
    idx = {}
    for r in rows:
        tid = r.get("task_id", "")
        idx.setdefault(tid, []).append(r)
    return idx


def find_matched_pairs(
    tasks_a: dict[str, list[dict]],
    tasks_b: dict[str, list[dict]],
    model_a: str,
    model_b: str,
    tolerance: float,
) -> list[dict]:
    """Find pairs on the same task where runtime gap <= tolerance."""
    pairs = []
    common_tasks = set(tasks_a.keys()) & set(tasks_b.keys())

    for tid in sorted(common_tasks):
        for ra in tasks_a[tid]:
            rt_a = ra.get("runtime_ms")
            j_a = ra.get("joules")
            if rt_a is None or j_a is None:
                continue
            rt_a = float(rt_a)
            j_a = float(j_a)
            if rt_a <= 0:
                continue

            for rb in tasks_b[tid]:
                rt_b = rb.get("runtime_ms")
                j_b = rb.get("joules")
                if rt_b is None or j_b is None:
                    continue
                rt_b = float(rt_b)
                j_b = float(j_b)
                if rt_b <= 0:
                    continue

                gap = abs(rt_a - rt_b) / rt_a
                if gap <= tolerance:
                    pairs.append({
                        "task_id": tid,
                        "model_a": model_a,
                        "model_b": model_b,
                        "seed_a": ra.get("seed", ""),
                        "seed_b": rb.get("seed", ""),
                        "runtime_a_ms": round(rt_a, 6),
                        "runtime_b_ms": round(rt_b, 6),
                        "delta_runtime_pct": round(gap * 100, 4),
                        "joules_a": round(j_a, 6),
                        "joules_b": round(j_b, 6),
                        "delta_joules_j": round(j_a - j_b, 6),
                        "delta_joules_pct": round((j_a - j_b) / j_a * 100, 4) if j_a > 0 else 0.0,
                        "power_a_w": float(ra.get("power_w", 0)),
                        "power_b_w": float(rb.get("power_w", 0)),
                    })

    return pairs


def compute_statistics(pairs: list[dict]) -> dict:
    """Compute summary statistics and significance tests on pairs."""
    if not pairs:
        return {"n_pairs": 0}

    deltas_j = [p["delta_joules_pct"] for p in pairs]
    deltas_j.sort()
    n = len(deltas_j)
    mean_delta = sum(deltas_j) / n
    median_delta = deltas_j[n // 2]
    std_delta = (sum((d - mean_delta) ** 2 for d in deltas_j) / max(1, n - 1)) ** 0.5

    q1 = deltas_j[n // 4]
    q3 = deltas_j[3 * n // 4]

    result = {
        "n_pairs": n,
        "mean_delta_joules_pct": round(mean_delta, 4),
        "median_delta_joules_pct": round(median_delta, 4),
        "std_delta_joules_pct": round(std_delta, 4),
        "iqr": [round(q1, 4), round(q3, 4)],
        "min_delta": round(deltas_j[0], 4),
        "max_delta": round(deltas_j[-1], 4),
    }

    # Effect size (Cohen's d)
    if std_delta > 0:
        cohens_d = mean_delta / std_delta
        result["cohens_d"] = round(cohens_d, 4)
    else:
        result["cohens_d"] = 0.0

    # Wilcoxon signed-rank test
    if HAS_SCIPY and n >= 5:
        joules_a = [p["joules_a"] for p in pairs]
        joules_b = [p["joules_b"] for p in pairs]
        try:
            stat, p_val = scipy_stats.wilcoxon(joules_a, joules_b, alternative="greater")
            result["wilcoxon_statistic"] = round(float(stat), 4)
            result["wilcoxon_p_value"] = round(float(p_val), 6)
            result["significant_at_005"] = p_val < 0.05
            result["significant_at_001"] = p_val < 0.01
        except Exception:
            result["wilcoxon_p_value"] = None
    else:
        result["wilcoxon_p_value"] = None

    return result


def plot_paired_analysis(pairs: list[dict], stats: dict, output_dir: Path, dpi: int):
    """Generate paired-line and distribution plots."""
    if not HAS_MPL or not pairs:
        return

    deltas = [p["delta_joules_pct"] for p in pairs]

    # Violin + paired points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ax1.violinplot([deltas], positions=[0], showmeans=True, showmedians=True)
    ax1.scatter([0] * len(deltas), deltas, alpha=0.3, s=10, color="#238b45", zorder=5)
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Delta Joules (%)", fontsize=12)
    ax1.set_title("Energy Advantage Distribution", fontsize=14)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Energy-aware vs Throughput"])
    ax1.grid(True, alpha=0.3, axis="y")

    n_show = min(40, len(pairs))
    for i, p in enumerate(pairs[:n_show]):
        ax2.plot([0, 1], [p["joules_a"], p["joules_b"]], "o-",
                 color="#238b45" if p["delta_joules_j"] > 0 else "#cb181d",
                 alpha=0.4, markersize=4)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Throughput (M1)", "Energy-aware (M2/M3)"])
    ax2.set_ylabel("Joules", fontsize=12)
    ax2.set_title("Paired Energy Comparison", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Matched-Runtime Energy Analysis (n={stats['n_pairs']}, "
        f"median={stats.get('median_delta_joules_pct', 0):.1f}%)",
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / "matched_runtime_energy_analysis.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_root = args.results_root
    output_dir = args.output or (results_root / "2026-03_M1_M2_M3__matched_runtime_different_energy")

    print(f"Matched-runtime energy experiment")
    print(f"  Results root: {results_root}")
    print(f"  Runtime tolerance: {args.runtime_tolerance * 100:.1f}%")

    # Load per-task metrics for M1, M2, M3
    model_exps = {
        "M1": "2026-03_M1_GRPO_THROUGHPUT__throughput_rl",
        "M2": "2026-03_M2_GRPO_ENERGY__energy_aware_rl",
        "M3": "2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep",
    }

    model_tasks = {}
    for mk, exp in model_exps.items():
        rows = load_per_task_metrics(results_root, exp)
        model_tasks[mk] = build_task_index(rows)
        print(f"  {mk}: {len(rows)} rows, {len(model_tasks[mk])} unique tasks")

    # Build pairs
    all_pairs = []
    for m_a, m_b in [("M1", "M2"), ("M1", "M3"), ("M2", "M3")]:
        pairs = find_matched_pairs(
            model_tasks[m_a], model_tasks[m_b], m_a, m_b, args.runtime_tolerance,
        )
        all_pairs.extend(pairs)
        print(f"  {m_a} vs {m_b}: {len(pairs)} matched pairs")

    print(f"  Total pairs: {len(all_pairs)}")

    # Compute statistics
    stats = compute_statistics(all_pairs)
    print(f"  Median delta joules: {stats.get('median_delta_joules_pct', 0):.2f}%")
    print(f"  Cohen's d: {stats.get('cohens_d', 0):.3f}")
    if stats.get("wilcoxon_p_value") is not None:
        print(f"  Wilcoxon p-value: {stats['wilcoxon_p_value']:.6f}")

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = output_dir.name

    if all_pairs:
        pairs_path = output_dir / f"{exp_name}_pairs.csv"
        with open(pairs_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_pairs[0].keys()))
            w.writeheader()
            w.writerows(all_pairs)

    stats_path = output_dir / f"{exp_name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Plot
    plot_paired_analysis(all_pairs, stats, output_dir, args.dpi)

    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
