"""
Per-task scatter: Runtime (ms) vs Energy (Joules) for each KITE model stage.
Shows that energy-aware models (M2, M3) achieve similar runtimes but consume
significantly less energy per kernel invocation.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "h100" / "2026-03"

MODEL_EXPERIMENTS = {
    "M0 SFT":        ("M0_SFT",            "kernel_generation_baseline"),
    "M1 Throughput":  ("M1_GRPO_THROUGHPUT", "throughput_rl"),
    "M2 Energy":      ("M2_GRPO_ENERGY",     "energy_aware_rl"),
    "M3 IPW Blend":   ("M3_GRPO_IPW_BLEND",  "ipw_blend_sweep"),
    "M4 Runtime-PPO": ("M4_RUNTIME_PPO",     "runtime_control"),
    "M5 HRL":         ("M5_HRL",             "hierarchical_control"),
}

COLORS = {
    "M0 SFT":        "#9E9E9E",
    "M1 Throughput":  "#4A9EDA",
    "M2 Energy":      "#2BC490",
    "M3 IPW Blend":   "#F5A623",
    "M4 Runtime-PPO": "#EF5350",
    "M5 HRL":         "#7C4DFF",
}


def load_per_task(model_tag: str, experiment: str) -> list[dict]:
    folder = RESULTS_DIR / f"2026-03_{model_tag}__{experiment}"
    jsonl = folder / f"2026-03_{model_tag}__{experiment}_per_task.jsonl"
    rows = []
    with open(jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    fig, ax = plt.subplots(figsize=(11, 7))

    means = {}
    for label, (model_tag, experiment) in MODEL_EXPERIMENTS.items():
        rows = load_per_task(model_tag, experiment)
        joules = np.array([r["joules"] for r in rows])
        runtime = np.array([r["runtime_ms"] for r in rows])
        color = COLORS[label]

        ax.scatter(
            joules, runtime,
            c=color, alpha=0.30, s=18, edgecolors="none", zorder=2,
            label=label,
        )

        coeffs = np.polyfit(joules, runtime, 1)
        x_fit = np.linspace(joules.min(), joules.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color=color, linewidth=2.0, alpha=0.85, zorder=3)

        mean_j = joules.mean()
        mean_rt = runtime.mean()
        means[label] = (mean_j, mean_rt)
        ax.scatter(
            [mean_j], [mean_rt],
            c=color, s=160, edgecolors="black", linewidths=1.4,
            zorder=5, marker="o",
        )

    # Place mean-marker labels with manual offsets to avoid overlap
    offsets = {
        "M0 SFT":        (10, 12),
        "M1 Throughput":  (10, -14),
        "M2 Energy":      (-80, 14),
        "M3 IPW Blend":   (10, -16),
        "M4 Runtime-PPO": (-110, -10),
        "M5 HRL":         (-50, -18),
    }
    for label, (mj, mrt) in means.items():
        ox, oy = offsets.get(label, (8, 6))
        ax.annotate(
            label,
            (mj, mrt),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=9, fontweight="bold", color=COLORS[label],
            zorder=6,
        )

    # Draw a horizontal dashed line between M0 and M2 means to highlight energy saving
    m0_j, m0_rt = means["M0 SFT"]
    m2_j, m2_rt = means["M2 Energy"]
    mid_rt = (m0_rt + m2_rt) / 2
    pct = (m0_j - m2_j) / m0_j * 100
    ax.annotate(
        "",
        xy=(m2_j, mid_rt), xytext=(m0_j, mid_rt),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.5, ls="--"),
        zorder=4,
    )
    ax.text(
        (m0_j + m2_j) / 2, mid_rt + 1.5,
        f"{pct:.0f}% less energy\nat matched runtime",
        ha="center", va="bottom", fontsize=9,
        fontstyle="italic", color="#444444", zorder=7,
    )

    ax.set_xlabel("Energy (Joules)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Runtime (ms)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Runtime vs Energy per Kernel Task",
        fontsize=15, fontweight="bold", pad=12,
    )

    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper left", fontsize=9, framealpha=0.9,
        edgecolor="#cccccc", title="Model", title_fontsize=10,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    out_dir = RESULTS_DIR / "paper_outputs" / "custom_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runtime_vs_energy_scatter.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
