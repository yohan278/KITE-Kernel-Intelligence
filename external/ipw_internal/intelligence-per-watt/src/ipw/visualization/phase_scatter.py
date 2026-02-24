"""Phase scatter visualization - energy vs tokens by phase."""

from __future__ import annotations

import math
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


@VisualizationRegistry.register("phase-scatter")
class PhaseScatterVisualization(VisualizationProvider):
    """Scatter plots for phase energy vs token counts."""

    visualization_id = "phase-scatter"

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

        prefill_x = []
        prefill_y = []
        decode_x = []
        decode_y = []
        ratio_x = []
        ratio_y = []

        for record in dataset:
            model_metrics = record.get("model_metrics", {}).get(model_name, {})
            if not model_metrics:
                continue

            token_metrics = model_metrics.get("token_metrics", {})
            phase_metrics = model_metrics.get("phase_metrics", {})

            input_tokens = _to_float(token_metrics.get("input"))
            output_tokens = _to_float(token_metrics.get("output"))
            total_tokens = None
            if input_tokens is not None or output_tokens is not None:
                total_tokens = (input_tokens or 0.0) + (output_tokens or 0.0)

            prefill_energy = _to_float(phase_metrics.get("prefill_energy_j"))
            decode_energy = _to_float(phase_metrics.get("decode_energy_j"))

            if input_tokens is not None and prefill_energy is not None:
                if math.isfinite(input_tokens) and math.isfinite(prefill_energy):
                    prefill_x.append(input_tokens)
                    prefill_y.append(prefill_energy)

            if output_tokens is not None and decode_energy is not None:
                if math.isfinite(output_tokens) and math.isfinite(decode_energy):
                    decode_x.append(output_tokens)
                    decode_y.append(decode_energy)

            if (
                total_tokens is not None
                and prefill_energy is not None
                and decode_energy is not None
                and prefill_energy > 0.0
            ):
                ratio = decode_energy / prefill_energy
                if math.isfinite(total_tokens) and math.isfinite(ratio):
                    ratio_x.append(total_tokens)
                    ratio_y.append(ratio)

        if not prefill_x and not decode_x and not ratio_x:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=(
                    "No phase scatter data found. Run profiling with --phased to generate phase metrics.",
                ),
            )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].scatter(prefill_x, prefill_y, alpha=0.35, s=12, color="#1f77b4")
        axes[0].set_title("Prefill Energy vs Input Tokens")
        axes[0].set_xlabel("Input tokens")
        axes[0].set_ylabel("Prefill energy (J)")
        axes[0].grid(alpha=0.3)

        axes[1].scatter(decode_x, decode_y, alpha=0.35, s=12, color="#ff7f0e")
        axes[1].set_title("Decode Energy vs Output Tokens")
        axes[1].set_xlabel("Output tokens")
        axes[1].set_ylabel("Decode energy (J)")
        axes[1].grid(alpha=0.3)

        axes[2].scatter(ratio_x, ratio_y, alpha=0.35, s=12, color="#2ca02c")
        axes[2].set_title("Decode/Prefill Energy Ratio")
        axes[2].set_xlabel("Total tokens")
        axes[2].set_ylabel("Energy ratio")
        axes[2].grid(alpha=0.3)

        fig.suptitle(f"Phase Scatter Plots\nModel: {model_name}  |  Hardware: {hardware_label}")
        fig.tight_layout()

        output_path = context.output_dir / "phase_scatter.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=280)
        plt.close(fig)

        return VisualizationResult(
            visualization=self.visualization_id,
            artifacts={"phase_scatter": output_path},
            metadata={
                "model": model_name,
                "hardware": hardware_label,
                "prefill_samples": len(prefill_x),
                "decode_samples": len(decode_x),
                "ratio_samples": len(ratio_x),
            },
        )


__all__ = ["PhaseScatterVisualization"]
