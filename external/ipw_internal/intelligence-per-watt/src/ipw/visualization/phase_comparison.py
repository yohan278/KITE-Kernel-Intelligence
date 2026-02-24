"""Phase comparison visualization - stacked energy bar chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ipw.core.registry import VisualizationRegistry
from ipw.visualization.base import VisualizationContext, VisualizationProvider, VisualizationResult


def _load_dataset(results_dir: Path):
    from datasets import load_from_disk

    return load_from_disk(str(results_dir))


def _infer_model_name(dataset) -> Optional[str]:
    for record in dataset:
        model_metrics = record.get("model_metrics", {})
        if model_metrics:
            return next(iter(model_metrics.keys()))
    return None


def _infer_hardware_label(dataset, model_name: str) -> str:
    for record in dataset:
        model_metrics = record.get("model_metrics", {}).get(model_name)
        if not model_metrics:
            continue

        gpu_info = model_metrics.get("gpu_info", {})
        if isinstance(gpu_info, dict):
            gpu_name = gpu_info.get("name", "")
            if gpu_name:
                return gpu_name

        system_info = model_metrics.get("system_info", {})
        if isinstance(system_info, dict):
            cpu_brand = system_info.get("cpu_brand", "")
            if cpu_brand and cpu_brand != "Unknown CPU":
                return cpu_brand

    return "Unknown"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_phase_energy(dataset, model_name: str) -> tuple[Optional[float], Optional[float]]:
    prefill_values = []
    decode_values = []

    for record in dataset:
        phase_metrics = (
            record.get("model_metrics", {})
            .get(model_name, {})
            .get("phase_metrics", {})
        )
        prefill = _to_float(phase_metrics.get("prefill_energy_j"))
        decode = _to_float(phase_metrics.get("decode_energy_j"))
        if prefill is not None:
            prefill_values.append(prefill)
        if decode is not None:
            decode_values.append(decode)

    prefill_total = sum(prefill_values) if prefill_values else None
    decode_total = sum(decode_values) if decode_values else None
    return prefill_total, decode_total


@VisualizationRegistry.register("phase-comparison")
class PhaseComparisonVisualization(VisualizationProvider):
    """Stacked bar chart comparing prefill vs decode energy."""

    visualization_id = "phase-comparison"

    def render(self, context: VisualizationContext) -> VisualizationResult:
        dataset = _load_dataset(context.results_dir)

        model_name = context.options.get("model")
        if not model_name:
            model_name = _infer_model_name(dataset)
        if not model_name:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=("No model found in dataset. Specify --model in options.",),
            )

        hardware_label = _infer_hardware_label(dataset, model_name)

        prefill_total, decode_total = _sum_phase_energy(dataset, model_name)
        if prefill_total is None and decode_total is None:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=(
                    "No phase energy data found. Run profiling with --phased to generate phase metrics.",
                ),
            )

        prefill_total = prefill_total or 0.0
        decode_total = decode_total or 0.0
        total = prefill_total + decode_total

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Energy"], [prefill_total], color="#1f77b4", label="Prefill")
        ax.bar(
            ["Energy"],
            [decode_total],
            bottom=[prefill_total],
            color="#ff7f0e",
            label="Decode",
        )

        ax.set_ylabel("Energy (J)")
        subtitle = f"Model: {model_name} | Hardware: {hardware_label}"
        wrapped_subtitle = textwrap.fill(subtitle, width=52)
        ax.set_title(
            f"Phase Energy Breakdown\n{wrapped_subtitle}",
            fontsize=11,
            pad=10,
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(total * 1.1, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        output_path = context.output_dir / "phase_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=280)
        plt.close(fig)

        return VisualizationResult(
            visualization=self.visualization_id,
            artifacts={"phase_comparison": output_path},
            metadata={
                "model": model_name,
                "hardware": hardware_label,
                "prefill_total_energy_j": prefill_total,
                "decode_total_energy_j": decode_total,
            },
        )


__all__ = ["PhaseComparisonVisualization"]
