"""Plot profiling results and simulator roofline charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ipw.cli._console import error, info, success, warning


def _collect_options(ctx, param, values):
    """Parse key=value options into a dictionary."""
    collected: Dict[str, str] = {}
    for item in values:
        for piece in item.split(","):
            if not piece:
                continue
            key, _, raw = piece.partition("=")
            key = key.strip()
            if not key:
                continue
            collected[key] = raw.strip()
    return collected


@click.group(
    help="Generate plots from profiling results or simulator predictions.",
    invoke_without_command=True,
)
@click.pass_context
def plot(ctx: click.Context) -> None:
    """Plot command group. Run without subcommand for help."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@plot.command("results", help="Generate plots from profiling results directory.")
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--visualization",
    "--viz",
    "-v",
    "visualization_id",
    type=str,
    default="auto",
    help="Visualization provider to use (default: auto). Use 'all' to run every registered visualization.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    help="Directory to save plots (default: <results_dir>/plots).",
)
@click.option(
    "--option",
    "options",
    multiple=True,
    callback=_collect_options,
    help="Visualization-specific options (e.g., --option model=llama3.2:1b).",
)
def plot_results(
    results_dir: Path,
    visualization_id: str,
    output_dir: Path | None,
    options: Dict[str, Any],
) -> None:
    """Generate visualizations from profiling results."""
    from ipw.core.registry import VisualizationRegistry
    from ipw.visualization.base import VisualizationContext

    import ipw.visualization

    ipw.visualization.ensure_registered()

    if output_dir is None:
        output_dir = results_dir / "plots"

    context = VisualizationContext(
        results_dir=results_dir,
        output_dir=output_dir,
        options=options,
    )

    if visualization_id in {"auto", "all"}:
        results = []
        for viz_id, provider_cls in VisualizationRegistry.items():
            try:
                provider = provider_cls()
                results.append(provider.render(context))
            except Exception as exc:
                warning(f"Visualization '{viz_id}' failed: {exc}")
        if not results:
            warning("No visualizations generated")
            return
        for result in results:
            _print_result(result)
        return

    try:
        provider_cls = VisualizationRegistry.get(visualization_id)
    except KeyError:
        available = [key for key, _ in VisualizationRegistry.items()]
        error(f"Visualization '{visualization_id}' not found.")
        if available:
            info(f"Available visualizations: {', '.join(available)}")
        raise click.Abort()

    provider = provider_cls()
    result = provider.render(context)
    _print_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# ipw plot roofline
# ─────────────────────────────────────────────────────────────────────────────

from ipw.simulator.hardware_specs import HARDWARE_SPECS_REGISTRY

_ALL_GPUS = sorted(HARDWARE_SPECS_REGISTRY.keys())

_DEFAULT_MODELS = "4,8,14,32"


@plot.command("roofline", help="Generate roofline plots from simulator predictions.")
@click.option(
    "--gpus", "-g",
    type=str,
    default=",".join(_ALL_GPUS),
    help=f"Comma-separated GPU types. Default: all ({len(_ALL_GPUS)} GPUs)",
)
@click.option(
    "--models", "-m",
    "model_sizes",
    type=str,
    default=_DEFAULT_MODELS,
    help="Comma-separated model sizes in billions (active params). Default: 4,8,14,32",
)
@click.option(
    "--input-tokens", "-i",
    type=int,
    default=500,
    help="Input tokens per inference call. Default: 500",
)
@click.option(
    "--output-tokens", "-o",
    type=int,
    default=200,
    help="Output tokens per inference call. Default: 200",
)
@click.option(
    "--output-dir", "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save plots. Default: current directory",
)
@click.option(
    "--format", "-f",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output image format. Default: png",
)
@click.option(
    "--dpi",
    type=int,
    default=150,
    help="Image resolution in DPI. Default: 150",
)
@click.option(
    "--calibration",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to calibration JSON (adds calibrated data points to plots)",
)
def plot_roofline(
    gpus: str,
    model_sizes: str,
    input_tokens: int,
    output_tokens: int,
    output_dir: Optional[Path],
    fmt: str,
    dpi: int,
    calibration: Optional[Path],
) -> None:
    """Generate roofline plots comparing hardware and model configurations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ipw.simulator.hardware_specs import get_hardware_specs
    from ipw.simulator.inference_model import predict
    from ipw.simulator.calibration import CalibrationDB

    # Parse inputs
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    sizes = [float(s.strip()) for s in model_sizes.split(",") if s.strip()]
    sizes.sort()

    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate GPUs
    valid_gpus: List[str] = []
    for g in gpu_list:
        try:
            get_hardware_specs(g)
            valid_gpus.append(g)
        except KeyError:
            warning(f"Skipping unknown GPU: {g}")

    if not valid_gpus:
        error("No valid GPUs specified.")
        raise click.Abort()

    # Load calibration if provided
    cal_db = None
    if calibration:
        cal_db = CalibrationDB()
        cal_db.load(calibration)
        info(f"Loaded {len(cal_db)} calibration entries")

    # Vendor colors
    vendor_colors = {"apple": "#999999", "nvidia": "#76B900", "amd": "#ED1C24"}

    # Short labels
    gpu_labels = {
        "a100_80gb": "A100", "h100_80gb": "H100", "h200": "H200",
        "gh200": "GH200", "b200": "B200", "mi300x": "MI300X",
        "m4_max": "M4 Max", "m4_pro": "M4 Pro",
        "m3_max": "M3 Max", "m3_pro": "M3 Pro",
    }

    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    info(f"Generating roofline plots: {len(valid_gpus)} GPUs x {len(sizes)} model sizes")
    info(f"  Tokens: {input_tokens} input, {output_tokens} output")

    # ── Plot 1: Hardware Comparison (latency + energy side by side) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, size in enumerate(sizes):
        latencies = []
        energies = []
        for gpu_key in valid_gpus:
            hw = get_hardware_specs(gpu_key)
            bpp = hw.bytes_per_param
            r = predict(hw, size, input_tokens, output_tokens, bytes_per_param=bpp)
            latencies.append(r.total_time_seconds * 1000)
            energies.append(r.total_energy_joules)

        mk = markers[i % len(markers)]
        ax1.plot(range(len(valid_gpus)), latencies, marker=mk,
                 label=f"{size:.0f}B", linewidth=2, markersize=8, zorder=3)
        ax2.plot(range(len(valid_gpus)), energies, marker=mk,
                 label=f"{size:.0f}B", linewidth=2, markersize=8, zorder=3)

    tick_labels = [gpu_labels.get(g, g) for g in valid_gpus]
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(valid_gpus)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        for j, gpu_key in enumerate(valid_gpus):
            hw = get_hardware_specs(gpu_key)
            color = vendor_colors.get(hw.vendor, "black")
            ax.get_xticklabels()[j].set_color(color)

    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title(f"Predicted Latency by Hardware\n({input_tokens} input, {output_tokens} output tokens)", fontsize=13)
    ax2.set_ylabel("Energy (J)", fontsize=12)
    ax2.set_title(f"Predicted Energy by Hardware\n({input_tokens} input, {output_tokens} output tokens)", fontsize=13)

    plt.tight_layout()
    path1 = output_dir / f"roofline_hardware_comparison.{fmt}"
    plt.savefig(path1, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    success(f"  Saved: {path1}")

    # ── Plot 2: Throughput vs Model Size ──
    fig2, ax3 = plt.subplots(figsize=(10, 7))

    for gpu_key in valid_gpus:
        hw = get_hardware_specs(gpu_key)
        color = vendor_colors.get(hw.vendor, "black")
        tps_list = []

        for size in sizes:
            bpp = hw.bytes_per_param
            r = predict(hw, size, input_tokens, output_tokens, bytes_per_param=bpp)
            tps = output_tokens / r.total_time_seconds if r.total_time_seconds > 0 else 0
            tps_list.append(tps)

        ax3.plot(sizes, tps_list, marker="o", label=gpu_labels.get(gpu_key, gpu_key),
                 color=color, linewidth=2, markersize=7, alpha=0.85)

    ax3.set_xlabel("Active Parameters (B)", fontsize=12)
    ax3.set_ylabel("Output Tokens/s", fontsize=12)
    ax3.set_title(f"Roofline: Throughput vs Model Size\n({input_tokens} input, {output_tokens} output tokens)", fontsize=13)
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xticks(sizes)
    ax3.set_xticklabels([f"{s:.0f}B" for s in sizes])

    plt.tight_layout()
    path2 = output_dir / f"roofline_throughput.{fmt}"
    plt.savefig(path2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    success(f"  Saved: {path2}")

    # ── Plot 3: Energy-Latency Pareto Frontier ──
    fig3, ax4 = plt.subplots(figsize=(10, 7))

    for gpu_key in valid_gpus:
        hw = get_hardware_specs(gpu_key)
        color = vendor_colors.get(hw.vendor, "black")
        for size in sizes:
            bpp = hw.bytes_per_param
            r = predict(hw, size, input_tokens, output_tokens, bytes_per_param=bpp)
            dot_size = 40 + size * 8
            ax4.scatter(r.total_time_seconds * 1000, r.total_energy_joules,
                        s=dot_size, color=color, alpha=0.7,
                        edgecolors="white", linewidth=0.5)

    # Legend: vendors
    for vendor, color in vendor_colors.items():
        if any(get_hardware_specs(g).vendor == vendor for g in valid_gpus):
            ax4.scatter([], [], color=color, s=80, label=vendor.capitalize())
    # Legend: model sizes
    for size in sizes:
        ax4.scatter([], [], color="gray", s=40 + size * 8, label=f"{size:.0f}B")

    ax4.set_xlabel("Latency (ms)", fontsize=12)
    ax4.set_ylabel("Energy (J)", fontsize=12)
    ax4.set_title(f"Energy-Latency Tradeoff\n(each point = one GPU x model combo)", fontsize=13)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = output_dir / f"roofline_energy_latency.{fmt}"
    plt.savefig(path3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    success(f"  Saved: {path3}")

    # ── Plot 4: Energy Efficiency (J per output token) ──
    fig4, ax5 = plt.subplots(figsize=(10, 7))

    for gpu_key in valid_gpus:
        hw = get_hardware_specs(gpu_key)
        color = vendor_colors.get(hw.vendor, "black")
        jpt_list = []

        for size in sizes:
            bpp = hw.bytes_per_param
            r = predict(hw, size, input_tokens, output_tokens, bytes_per_param=bpp)
            jpt = r.total_energy_joules / output_tokens if output_tokens > 0 else 0
            jpt_list.append(jpt)

        ax5.plot(sizes, jpt_list, marker="o", label=gpu_labels.get(gpu_key, gpu_key),
                 color=color, linewidth=2, markersize=7, alpha=0.85)

    ax5.set_xlabel("Active Parameters (B)", fontsize=12)
    ax5.set_ylabel("Energy per Output Token (J/tok)", fontsize=12)
    ax5.set_title(f"Energy Efficiency: Joules per Output Token\n({input_tokens} input, {output_tokens} output tokens)", fontsize=13)
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale("log")
    ax5.set_yscale("log")
    ax5.set_xticks(sizes)
    ax5.set_xticklabels([f"{s:.0f}B" for s in sizes])

    plt.tight_layout()
    path4 = output_dir / f"roofline_energy_efficiency.{fmt}"
    plt.savefig(path4, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    success(f"  Saved: {path4}")

    info(f"\nAll plots saved to: {output_dir}")


def _print_result(result) -> None:
    info(f"Visualization: {result.visualization}")

    if result.artifacts:
        info("\nGenerated artifacts:")
        for name, path in result.artifacts.items():
            info(f"  {name}: {path}")
    else:
        warning("No artifacts generated")

    if result.warnings:
        info("\nWarnings:")
        for warn in result.warnings:
            warning(f"  {warn}")

    if result.metadata:
        info("\nMetadata:")
        for key, value in result.metadata.items():
            info(f"  {key}: {value}")


__all__ = ["plot"]
