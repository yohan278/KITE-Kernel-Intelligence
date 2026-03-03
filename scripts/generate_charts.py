#!/usr/bin/env python3
"""Generate charts from energy profiling and scaling experiment results.

    python scripts/generate_charts.py \
        --profile-dir results/energy_profiling \
        --scaling-dir results/input_scaling \
        --output-dir results/charts
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {
    "matmul": "#e41a1c", "attention": "#377eb8", "norm": "#4daf4a",
    "activation": "#984ea3", "reduction": "#ff7f00", "conv": "#a65628",
    "conv_transpose": "#f781bf", "composite": "#999999", "pooling": "#66c2a5",
    "loss": "#fc8d62", "scan": "#8da0cb", "unknown": "#cccccc",
}


def _color(kt):
    return COLORS.get(kt, "#888888")


def load_profiles(profile_dir):
    p = Path(profile_dir) / "kernel_profiles.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_scaling(scaling_dir):
    p = Path(scaling_dir) / "scaling_results.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def chart_energy_by_type(profiles, output_dir):
    """Box plot of energy per ms by kernel type."""
    by_type = defaultdict(list)
    for p in profiles:
        if p.get("energy_per_ms"):
            by_type[p["kernel_type"]].append(p["energy_per_ms"])

    types = sorted(by_type.keys(), key=lambda k: np.median(by_type[k]))
    data = [by_type[t] for t in types]
    colors = [_color(t) for t in types]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, patch_artist=True, labels=types)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Energy per ms (J/ms)")
    ax.set_title("Energy Efficiency by Kernel Type")
    ax.set_xticklabels(types, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "1_energy_per_ms_by_type.png", dpi=150)
    plt.close(fig)
    print("  1_energy_per_ms_by_type.png")


def chart_energy_vs_runtime(profiles, output_dir):
    """Scatter: energy vs runtime colored by type."""
    fig, ax = plt.subplots(figsize=(12, 8))
    for p in profiles:
        if p.get("energy_j") and p.get("runtime_ms"):
            ax.scatter(p["runtime_ms"], p["energy_j"],
                       c=_color(p["kernel_type"]), alpha=0.6, s=40,
                       edgecolors="black", linewidths=0.3)

    patches = [mpatches.Patch(color=_color(t), label=t)
               for t in sorted(set(p["kernel_type"] for p in profiles if p.get("energy_j")))]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy vs Runtime by Kernel Type")
    plt.tight_layout()
    fig.savefig(output_dir / "2_energy_vs_runtime.png", dpi=150)
    plt.close(fig)
    print("  2_energy_vs_runtime.png")


def chart_power_by_type(profiles, output_dir):
    """Bar chart of average power draw by kernel type."""
    by_type = defaultdict(list)
    for p in profiles:
        if p.get("avg_power_w"):
            by_type[p["kernel_type"]].append(p["avg_power_w"])

    types = sorted(by_type.keys())
    means = [np.mean(by_type[t]) for t in types]
    stds = [np.std(by_type[t]) for t in types]
    colors = [_color(t) for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(types, means, yerr=stds, color=colors, alpha=0.7,
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_ylabel("Average Power (W)")
    ax.set_title("Power Draw by Kernel Type")
    ax.set_xticklabels(types, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "3_power_by_type.png", dpi=150)
    plt.close(fig)
    print("  3_power_by_type.png")


def chart_gpu_util_vs_energy(profiles, output_dir):
    """Scatter: GPU utilization vs energy efficiency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for p in profiles:
        if p.get("energy_per_ms") and p.get("avg_gpu_util_pct") is not None:
            c = _color(p["kernel_type"])
            ax1.scatter(p["avg_gpu_util_pct"], p["energy_per_ms"],
                        c=c, alpha=0.6, s=30)
        if p.get("energy_per_ms") and p.get("avg_mem_util_pct") is not None:
            c = _color(p["kernel_type"])
            ax2.scatter(p["avg_mem_util_pct"], p["energy_per_ms"],
                        c=c, alpha=0.6, s=30)

    ax1.set_xlabel("GPU Utilization (%)")
    ax1.set_ylabel("Energy per ms (J/ms)")
    ax1.set_title("GPU Util vs Energy Efficiency")
    ax2.set_xlabel("Memory Utilization (%)")
    ax2.set_ylabel("Energy per ms (J/ms)")
    ax2.set_title("Mem Util vs Energy Efficiency")
    plt.tight_layout()
    fig.savefig(output_dir / "4_utilization_vs_energy.png", dpi=150)
    plt.close(fig)
    print("  4_utilization_vs_energy.png")


def chart_variance_within_type(profiles, output_dir):
    """Bar chart showing coefficient of variation in energy within each type."""
    by_type = defaultdict(list)
    for p in profiles:
        if p.get("energy_per_ms"):
            by_type[p["kernel_type"]].append(p["energy_per_ms"])

    types = []
    cvs = []
    for kt in sorted(by_type.keys()):
        vals = by_type[kt]
        if len(vals) < 3:
            continue
        m = np.mean(vals)
        if m > 0:
            types.append(kt)
            cvs.append(np.std(vals) / m * 100)

    order = sorted(range(len(cvs)), key=lambda i: cvs[i], reverse=True)
    types = [types[i] for i in order]
    cvs = [cvs[i] for i in order]
    colors = [_color(t) for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(types, cvs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Coefficient of Variation (%)")
    ax.set_title("Within-Type Energy Variance\n(Higher = more room for optimization)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / "5_energy_variance_within_type.png", dpi=150)
    plt.close(fig)
    print("  5_energy_variance_within_type.png")


def chart_compute_vs_memory(profiles, output_dir):
    """Scatter: GPU util vs mem util showing compute/memory bound split."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for p in profiles:
        gu = p.get("avg_gpu_util_pct")
        mu = p.get("avg_mem_util_pct")
        if gu is not None and mu is not None:
            ax.scatter(mu, gu, c=_color(p["kernel_type"]), alpha=0.6, s=40,
                       edgecolors="black", linewidths=0.3)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="balanced line")
    ax.set_xlabel("Memory Utilization (%)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("Compute vs Memory Bound Classification")
    patches = [mpatches.Patch(color=_color(t), label=t)
               for t in sorted(set(p["kernel_type"] for p in profiles
                                    if p.get("avg_gpu_util_pct") is not None))]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "6_compute_vs_memory_bound.png", dpi=150)
    plt.close(fig)
    print("  6_compute_vs_memory_bound.png")


def chart_scaling_energy(scaling_data, output_dir):
    """Line chart: energy vs input scale by kernel type."""
    if not scaling_data:
        return
    by_type = defaultdict(lambda: defaultdict(list))
    for r in scaling_data:
        if r.get("error") or not r.get("energy_j"):
            continue
        by_type[r["kernel_type"]][r["scale"]].append(r["energy_j"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    scales = sorted(set(r["scale"] for r in scaling_data if not r.get("error")))

    for kt in sorted(by_type.keys()):
        means = []
        valid_scales = []
        for s in scales:
            vals = by_type[kt][s]
            if vals:
                means.append(np.mean(vals))
                valid_scales.append(s)
        if len(valid_scales) >= 2:
            ax1.plot(valid_scales, means, "o-", color=_color(kt), label=kt,
                     markersize=6, linewidth=2)

    ax1.set_xlabel("Input Scale Factor")
    ax1.set_ylabel("Energy (J)")
    ax1.set_title("Energy vs Input Size Scale")
    ax1.legend(fontsize=8)
    ax1.set_xticks(scales)

    # Normalized: energy relative to 1.0x scale
    for kt in sorted(by_type.keys()):
        base_vals = by_type[kt].get(1.0, [])
        if not base_vals:
            continue
        base = np.mean(base_vals)
        if base <= 0:
            continue
        norms = []
        valid_scales = []
        for s in scales:
            vals = by_type[kt][s]
            if vals:
                norms.append(np.mean(vals) / base)
                valid_scales.append(s)
        if len(valid_scales) >= 2:
            ax2.plot(valid_scales, norms, "o-", color=_color(kt), label=kt,
                     markersize=6, linewidth=2)

    ax2.set_xlabel("Input Scale Factor")
    ax2.set_ylabel("Normalized Energy (1.0 = baseline)")
    ax2.set_title("Energy Scaling (Normalized to 1x)")
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_xticks(scales)
    plt.tight_layout()
    fig.savefig(output_dir / "7_energy_scaling_by_input_size.png", dpi=150)
    plt.close(fig)
    print("  7_energy_scaling_by_input_size.png")


def chart_scaling_runtime(scaling_data, output_dir):
    """Line chart: runtime vs input scale by kernel type."""
    if not scaling_data:
        return
    by_type = defaultdict(lambda: defaultdict(list))
    for r in scaling_data:
        if r.get("error") or not r.get("runtime_ms"):
            continue
        by_type[r["kernel_type"]][r["scale"]].append(r["runtime_ms"])

    fig, ax = plt.subplots(figsize=(10, 6))
    scales = sorted(set(r["scale"] for r in scaling_data if not r.get("error")))

    for kt in sorted(by_type.keys()):
        means = []
        valid_scales = []
        for s in scales:
            vals = by_type[kt][s]
            if vals:
                means.append(np.mean(vals))
                valid_scales.append(s)
        if len(valid_scales) >= 2:
            ax.plot(valid_scales, means, "o-", color=_color(kt), label=kt,
                    markersize=6, linewidth=2)

    ax.set_xlabel("Input Scale Factor")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime vs Input Size Scale")
    ax.legend(fontsize=8)
    ax.set_xticks(scales)
    plt.tight_layout()
    fig.savefig(output_dir / "8_runtime_scaling_by_input_size.png", dpi=150)
    plt.close(fig)
    print("  8_runtime_scaling_by_input_size.png")


def chart_energy_efficiency_scaling(scaling_data, output_dir):
    """Does energy efficiency (J/ms) change with scale?"""
    if not scaling_data:
        return
    by_type = defaultdict(lambda: defaultdict(list))
    for r in scaling_data:
        if r.get("error") or not r.get("energy_j") or not r.get("runtime_ms"):
            continue
        epm = r["energy_j"] / r["runtime_ms"] if r["runtime_ms"] > 0 else None
        if epm:
            by_type[r["kernel_type"]][r["scale"]].append(epm)

    fig, ax = plt.subplots(figsize=(10, 6))
    scales = sorted(set(r["scale"] for r in scaling_data if not r.get("error")))

    for kt in sorted(by_type.keys()):
        means = []
        valid_scales = []
        for s in scales:
            vals = by_type[kt][s]
            if vals:
                means.append(np.mean(vals))
                valid_scales.append(s)
        if len(valid_scales) >= 2:
            ax.plot(valid_scales, means, "o-", color=_color(kt), label=kt,
                    markersize=6, linewidth=2)

    ax.set_xlabel("Input Scale Factor")
    ax.set_ylabel("Energy per ms (J/ms)")
    ax.set_title("Energy Efficiency vs Input Size\n(Does efficiency change with scale?)")
    ax.legend(fontsize=8)
    ax.set_xticks(scales)
    plt.tight_layout()
    fig.savefig(output_dir / "9_efficiency_vs_input_size.png", dpi=150)
    plt.close(fig)
    print("  9_efficiency_vs_input_size.png")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment charts")
    parser.add_argument("--profile-dir", type=Path,
                        default=Path("results/energy_profiling"))
    parser.add_argument("--scaling-dir", type=Path,
                        default=Path("results/input_scaling"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/charts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(args.profile_dir)
    scaling = load_scaling(args.scaling_dir)

    print(f"Loaded {len(profiles)} kernel profiles, {len(scaling)} scaling results")
    print(f"Generating charts in {args.output_dir}/\n")

    if profiles:
        chart_energy_by_type(profiles, args.output_dir)
        chart_energy_vs_runtime(profiles, args.output_dir)
        chart_power_by_type(profiles, args.output_dir)
        chart_gpu_util_vs_energy(profiles, args.output_dir)
        chart_variance_within_type(profiles, args.output_dir)
        chart_compute_vs_memory(profiles, args.output_dir)

    if scaling:
        chart_scaling_energy(scaling, args.output_dir)
        chart_scaling_runtime(scaling, args.output_dir)
        chart_energy_efficiency_scaling(scaling, args.output_dir)

    print(f"\nDone! Charts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
