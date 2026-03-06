"""
Per-task delta forest plot: (M2 - M1) and (M3 - M1) energy delta per task.
Regenerates appendix figure 16 with readable y-axis labels.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "h100" / "2026-03"

MODELS = {
    "M1": ("M1_GRPO_THROUGHPUT", "throughput_rl"),
    "M2": ("M2_GRPO_ENERGY",     "energy_aware_rl"),
    "M3": ("M3_GRPO_IPW_BLEND",  "ipw_blend_sweep"),
}

LEVEL_COLORS = {
    "L1": "#1565C0",
    "L2": "#2E7D32",
    "L3": "#E65100",
    "L4": "#AD1457",
}


def load_per_task(model_tag: str, experiment: str) -> list[dict]:
    folder = RESULTS_DIR / f"2026-03_{model_tag}__{experiment}"
    jsonl = folder / f"2026-03_{model_tag}__{experiment}_per_task.jsonl"
    rows = []
    with open(jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean_joules_by_task(rows: list[dict]) -> dict[str, float]:
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        acc[r["task_id"]].append(r["joules"])
    return {tid: np.mean(vals) for tid, vals in acc.items()}


def main():
    m1_j = mean_joules_by_task(load_per_task(*MODELS["M1"]))
    m2_j = mean_joules_by_task(load_per_task(*MODELS["M2"]))
    m3_j = mean_joules_by_task(load_per_task(*MODELS["M3"]))

    common = sorted(set(m1_j) & set(m2_j) & set(m3_j))
    delta_m2 = {t: m2_j[t] - m1_j[t] for t in common}
    delta_m3 = {t: m3_j[t] - m1_j[t] for t in common}

    tasks_sorted = sorted(common, key=lambda t: delta_m3[t])

    n = len(tasks_sorted)
    col_width = 0.22
    fig_w = max(16, n * col_width + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    x = np.arange(n)
    d2_vals = [delta_m2[t] for t in tasks_sorted]
    d3_vals = [delta_m3[t] for t in tasks_sorted]

    # Alternating column shading
    for i in range(n):
        if i % 2 == 0:
            ax.axvspan(i - 0.4, i + 0.4, color="#f5f5f5", zorder=0)

    ax.vlines(x, 0, d2_vals, color="#2ca02c", alpha=0.4, linewidth=1.2, zorder=1)
    ax.plot(x, d2_vals, "o", color="#2ca02c", markersize=5, label="M2 Energy \u2212 M1", zorder=2)

    ax.vlines(x, 0, d3_vals, color="#d62728", alpha=0.4, linewidth=1.2, zorder=1)
    ax.plot(x, d3_vals, "s", color="#d62728", markersize=4, label="M3 IPW Blend \u2212 M1", zorder=2)

    ax.axhline(0, color="black", linewidth=1.0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks_sorted, fontsize=7, fontfamily="monospace", rotation=90, ha="right")

    for i, t in enumerate(tasks_sorted):
        level = t.split("_")[0]
        color = LEVEL_COLORS.get(level, "#333333")
        ax.get_xticklabels()[i].set_color(color)

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylabel("Delta Joules  (negative = less energy than M1)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Task Energy Delta Forest Plot  (vs M1 Throughput baseline)",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(
        loc="upper left", fontsize=10, framealpha=0.95, edgecolor="#cccccc",
        bbox_to_anchor=(0, 1.02), ncol=2,
    )
    legend.get_frame().set_linewidth(0.8)

    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_dir = RESULTS_DIR / "paper_outputs" / "appendix_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "16_per_task_delta_forest_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
