"""Phase power timeline visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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


@VisualizationRegistry.register("phase-power-timeline")
class PhasePowerTimelineVisualization(VisualizationProvider):
    """Plot phase power averages over time with phase boundaries."""

    visualization_id = "phase-power-timeline"

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

        segments = []
        boundaries = []
        cursor = 0.0

        for record in dataset:
            phase_metrics = (
                record.get("model_metrics", {})
                .get(model_name, {})
                .get("phase_metrics", {})
            )
            prefill_duration_ms = _to_float(phase_metrics.get("prefill_duration_ms"))
            decode_duration_ms = _to_float(phase_metrics.get("decode_duration_ms"))
            prefill_power = _to_float(phase_metrics.get("prefill_power_avg_w"))
            decode_power = _to_float(phase_metrics.get("decode_power_avg_w"))

            if prefill_duration_ms and prefill_power is not None:
                duration_s = max(prefill_duration_ms / 1000.0, 0.0)
                if duration_s > 0.0:
                    segments.append((cursor, cursor + duration_s, prefill_power, "prefill"))
                    cursor += duration_s
                    boundaries.append(cursor)

            if decode_duration_ms and decode_power is not None:
                duration_s = max(decode_duration_ms / 1000.0, 0.0)
                if duration_s > 0.0:
                    segments.append((cursor, cursor + duration_s, decode_power, "decode"))
                    cursor += duration_s
                    boundaries.append(cursor)

        if not segments:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=(
                    "No phase power timeline data found. Run profiling with --phased to generate phase metrics.",
                ),
            )

        fig, ax = plt.subplots(figsize=(10, 4))
        phase_colors = {"prefill": "#1f77b4", "decode": "#ff7f0e"}

        for start, end, power, phase in segments:
            color = phase_colors.get(phase, "#333333")
            ax.plot([start, end], [power, power], color=color, linewidth=2.2)
            ax.axvspan(start, end, color=color, alpha=0.08)

        for boundary in boundaries:
            ax.axvline(boundary, color="#999999", linewidth=0.6, alpha=0.4)

        ax.set_title(
            f"Phase Power Timeline\nModel: {model_name}  |  Hardware: {hardware_label}"
        )
        ax.set_xlabel("Elapsed time (seconds)")
        ax.set_ylabel("Average power (W)")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        output_path = context.output_dir / "phase_timeline.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=280)
        plt.close(fig)

        return VisualizationResult(
            visualization=self.visualization_id,
            artifacts={"phase_timeline": output_path},
            metadata={
                "model": model_name,
                "hardware": hardware_label,
                "segment_count": len(segments),
            },
        )


__all__ = ["PhasePowerTimelineVisualization"]
