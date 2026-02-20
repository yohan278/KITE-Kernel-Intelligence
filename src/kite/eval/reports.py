"""Reporting helpers for KITE experiments: Pareto frontiers, plots, and markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from kite.utils.logging import get_logger
from kite.utils.serialization import save_json

logger = get_logger(__name__)


def pareto_frontier(
    points: Iterable[Dict[str, float]],
    x_key: str,
    y_key: str,
    higher_is_better: bool = True,
) -> List[Dict[str, float]]:
    """Compute the non-dominated Pareto frontier from a set of points."""
    rows = list(points)
    frontier: list[dict[str, float]] = []
    for candidate in rows:
        dominated = False
        cx = candidate.get(x_key, 0.0)
        cy = candidate.get(y_key, 0.0)
        for other in rows:
            if other is candidate:
                continue
            ox = other.get(x_key, 0.0)
            oy = other.get(y_key, 0.0)
            if higher_is_better:
                if ox >= cx and oy >= cy and (ox > cx or oy > cy):
                    dominated = True
                    break
            else:
                if ox <= cx and oy <= cy and (ox < cx or oy < cy):
                    dominated = True
                    break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: p.get(x_key, 0.0), reverse=higher_is_better)


def write_markdown_report(output_path: Path, title: str, sections: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for name, value in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(value))
        lines.append("")
    output_path.write_text("\n".join(lines))


def plot_pareto_frontiers(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate Pareto frontier plots if matplotlib is available."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logger.info("matplotlib not available; skipping plots")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: Dict[str, Path] = {}

    ids = [r.get("id", "?") for r in results]
    metrics_list = [r.get("metrics", {}) for r in results]

    # Plot 1: Throughput vs Energy/Token (lower energy is better)
    fig, ax = plt.subplots(figsize=(10, 7))
    throughputs = [m.get("throughput_tps", m.get("speedup", 1.0) * 100) for m in metrics_list]
    energies = [m.get("energy_per_token_j", 0.25) for m in metrics_list]

    scatter = ax.scatter(energies, throughputs, s=120, zorder=5)
    for i, label in enumerate(ids):
        ax.annotate(label, (energies[i], throughputs[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=10)

    points = [{"id": ids[i], "energy_per_token_j": energies[i], "throughput_tps": throughputs[i]}
              for i in range(len(ids))]
    frontier = pareto_frontier(points, "throughput_tps", "energy_per_token_j", higher_is_better=False)
    if len(frontier) >= 2:
        fx = [p["energy_per_token_j"] for p in frontier]
        fy = [p["throughput_tps"] for p in frontier]
        ax.plot(fx, fy, "r--", linewidth=1.5, alpha=0.7, label="Pareto frontier")
        ax.legend()

    ax.set_xlabel("Energy per Token (J)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_title("Throughput vs Energy Efficiency", fontsize=14)
    ax.grid(True, alpha=0.3)
    path1 = output_dir / "pareto_throughput_vs_energy.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["throughput_vs_energy"] = path1

    # Plot 2: APJ vs p95 Latency
    fig, ax = plt.subplots(figsize=(10, 7))
    apjs = [m.get("apj", 0.01) for m in metrics_list]
    latencies = [m.get("ttft_p95_s", 2.0) for m in metrics_list]

    ax.scatter(latencies, apjs, s=120, zorder=5)
    for i, label in enumerate(ids):
        ax.annotate(label, (latencies[i], apjs[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=10)

    points2 = [{"id": ids[i], "apj": apjs[i], "ttft_p95_s": latencies[i]}
               for i in range(len(ids))]
    frontier2 = pareto_frontier(points2, "apj", "ttft_p95_s", higher_is_better=False)
    if len(frontier2) >= 2:
        fx2 = [p["ttft_p95_s"] for p in frontier2]
        fy2 = [p["apj"] for p in frontier2]
        ax.plot(fx2, fy2, "r--", linewidth=1.5, alpha=0.7, label="Pareto frontier")
        ax.legend()

    ax.set_xlabel("TTFT p95 Latency (s)", fontsize=12)
    ax.set_ylabel("APJ (accuracy per joule)", fontsize=12)
    ax.set_title("Intelligence Per Joule vs Latency", fontsize=14)
    ax.grid(True, alpha=0.3)
    path2 = output_dir / "pareto_apj_vs_latency.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["apj_vs_latency"] = path2

    # Plot 3: APW vs APJ
    fig, ax = plt.subplots(figsize=(10, 7))
    apws = [m.get("apw", 0.001) for m in metrics_list]

    ax.scatter(apjs, apws, s=120, zorder=5)
    for i, label in enumerate(ids):
        ax.annotate(label, (apjs[i], apws[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=10)

    points3 = [{"id": ids[i], "apj": apjs[i], "apw": apws[i]} for i in range(len(ids))]
    frontier3 = pareto_frontier(points3, "apj", "apw")
    if len(frontier3) >= 2:
        fx3 = [p["apj"] for p in frontier3]
        fy3 = [p["apw"] for p in frontier3]
        ax.plot(fx3, fy3, "r--", linewidth=1.5, alpha=0.7, label="Pareto frontier")
        ax.legend()

    ax.set_xlabel("APJ", fontsize=12)
    ax.set_ylabel("APW", fontsize=12)
    ax.set_title("Intelligence Per Watt vs Intelligence Per Joule", fontsize=14)
    ax.grid(True, alpha=0.3)
    path3 = output_dir / "pareto_apj_vs_apw.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["apj_vs_apw"] = path3

    logger.info("Plots saved: %s", list(plots.values()))
    return plots


def save_suite_artifacts(output_dir: Path, suite: Dict[str, Any]) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "suite_results.json"
    save_json(json_path, suite)

    results = suite.get("results", [])

    points = []
    for row in results:
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            points.append({
                "id": row.get("id", "unknown"),
                "apj": float(metrics.get("apj", 0.0)),
                "apw": float(metrics.get("apw", 0.0)),
            })

    frontier = pareto_frontier(points, "apj", "apw")
    frontier_path = output_dir / "pareto_frontier.json"
    save_json(frontier_path, {"frontier": frontier})

    plot_paths = plot_pareto_frontiers(results, output_dir)

    sections: Dict[str, Any] = {
        "Summary": f"Evaluated {len(results)} experiments.",
        "Experiment Results": {},
        "Pareto Frontier (APJ vs APW)": [
            f"{p['id']}: apj={p['apj']:.4f}, apw={p['apw']:.4f}" for p in frontier
        ],
    }

    for row in results:
        metrics = row.get("metrics", {})
        exp_id = row.get("id", "?")
        sections["Experiment Results"][exp_id] = (
            f"correctness={metrics.get('correctness', 'N/A'):.2%}, "
            f"speedup={metrics.get('speedup', 'N/A'):.2f}x, "
            f"energy/tok={metrics.get('energy_per_token_j', 'N/A'):.3f}J, "
            f"apj={metrics.get('apj', 'N/A'):.4f}"
        ) if isinstance(metrics, dict) else str(metrics)

    if plot_paths:
        sections["Plots"] = [f"`{name}`: {path}" for name, path in plot_paths.items()]

    md_path = output_dir / "report.md"
    write_markdown_report(md_path, title="KITE Evaluation Report", sections=sections)

    artifacts = {
        "suite_json": json_path,
        "pareto_json": frontier_path,
        "report_md": md_path,
    }
    artifacts.update(plot_paths)
    return artifacts
