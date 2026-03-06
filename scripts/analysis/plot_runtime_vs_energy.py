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
from matplotlib.lines import Line2D
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
    "M1 Throughput":  "#2196F3",
    "M2 Energy":      "#00C853",
    "M3 IPW Blend":   "#FF9800",
    "M4 Runtime-PPO": "#EF5350",
    "M5 HRL":         "#7C4DFF",
}

MARKERS = {
    "M0 SFT":        "o",
    "M1 Throughput":  "s",
    "M2 Energy":      "D",
    "M3 IPW Blend":   "^",
    "M4 Runtime-PPO": "v",
    "M5 HRL":         "P",
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
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(13, 8))

    all_data = {}
    means = {}
    for label, (model_tag, experiment) in MODEL_EXPERIMENTS.items():
        rows = load_per_task(model_tag, experiment)
        joules = np.array([r["joules"] for r in rows])
        runtime = np.array([r["runtime_ms"] for r in rows])
        all_data[label] = (joules, runtime)

        mean_j = joules.mean()
        mean_rt = runtime.mean()
        means[label] = (mean_j, mean_rt)

    # --- Main axes (linear) ---
    for label in MODEL_EXPERIMENTS:
        joules, runtime = all_data[label]
        color = COLORS[label]
        marker = MARKERS[label]

        ax.scatter(
            joules, runtime,
            c=color, alpha=0.35, s=24, edgecolors="none", zorder=2,
            marker=marker,
        )

        # Linear regression through the origin: runtime = slope * joules
        coeffs = np.polyfit(joules, runtime, 1)
        x_max = joules.max()
        x_fit = np.linspace(0, x_max * 1.05, 150)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color=color, linewidth=2.4, alpha=0.85, zorder=3,
                label=f"{label}  (slope={coeffs[0]:.1f} ms/J)")

        mj, mrt = means[label]
        ax.scatter(
            [mj], [mrt],
            c=color, s=200, edgecolors="black", linewidths=1.6,
            zorder=5, marker=marker,
        )

    # --- Inset zoom on the mean-marker cluster ---
    all_mj = [v[0] for v in means.values()]
    all_mrt = [v[1] for v in means.values()]
    pad_j = 2.0
    pad_rt = 4.0
    inset_x0 = min(all_mj) - pad_j
    inset_x1 = max(all_mj) + pad_j
    inset_y0 = min(all_mrt) - pad_rt
    inset_y1 = max(all_mrt) + pad_rt

    ax_inset = inset_axes(ax, width="38%", height="42%", loc="upper left",
                          borderpad=3.5)
    for label in MODEL_EXPERIMENTS:
        joules, runtime = all_data[label]
        color = COLORS[label]
        marker = MARKERS[label]
        mj, mrt = means[label]

        mask = (
            (joules >= inset_x0) & (joules <= inset_x1)
            & (runtime >= inset_y0) & (runtime <= inset_y1)
        )
        ax_inset.scatter(
            joules[mask], runtime[mask],
            c=color, alpha=0.45, s=20, edgecolors="none", marker=marker, zorder=2,
        )
        ax_inset.scatter(
            [mj], [mrt],
            c=color, s=140, edgecolors="black", linewidths=1.4,
            zorder=5, marker=marker,
        )
        ax_inset.annotate(
            label.split()[0],  # short name: M0, M1, ...
            (mj, mrt),
            textcoords="offset points", xytext=(7, 5),
            fontsize=8, fontweight="bold", color=color, zorder=6,
        )

    ax_inset.set_xlim(inset_x0, inset_x1)
    ax_inset.set_ylim(inset_y0, inset_y1)
    ax_inset.set_title("Mean cluster (zoom)", fontsize=9, fontweight="bold", pad=4)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(alpha=0.2, linewidth=0.4)
    ax_inset.set_axisbelow(True)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.8)
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="#888888",
               lw=0.8, ls="--")

    # --- Energy-saving annotation on main axes ---
    m0_j, m0_rt = means["M0 SFT"]
    m2_j, m2_rt = means["M2 Energy"]
    pct = (m0_j - m2_j) / m0_j * 100
    mid_rt = (m0_rt + m2_rt) / 2
    ax.annotate(
        "",
        xy=(m2_j, mid_rt), xytext=(m0_j, mid_rt),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6, ls="--"),
        zorder=4,
    )
    ax.text(
        (m0_j + m2_j) / 2, mid_rt + 2.5,
        f"{pct:.0f}% less energy\nat matched runtime",
        ha="center", va="bottom", fontsize=10,
        fontstyle="italic", color="#333333", fontweight="semibold", zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa",
                  alpha=0.92, linewidth=0.6),
    )

    ax.set_xlabel("Energy (Joules)  \u2190 lower is better", fontsize=13, fontweight="bold")
    ax.set_ylabel("Runtime (ms)  \u2190 lower is better", fontsize=13, fontweight="bold")
    ax.set_title(
        "Runtime vs Energy per Kernel Task",
        fontsize=16, fontweight="bold", pad=14,
    )

    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="lower right", fontsize=9.5, framealpha=0.92,
        edgecolor="#cccccc", title="Model  (linear slope = ms per Joule)",
        title_fontsize=10, handletextpad=0.6,
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
