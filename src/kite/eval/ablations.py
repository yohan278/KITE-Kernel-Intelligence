"""Ablation study configuration and runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from kite.eval.benchmark_runner import BenchmarkRunner, ExperimentRow
from kite.utils.logging import get_logger
from kite.utils.serialization import save_json

logger = get_logger(__name__)


@dataclass(slots=True)
class AblationCase:
    ablation_id: str
    description: str
    base_exp_id: str
    disabled_component: str


DEFAULT_ABLATIONS = [
    AblationCase("A1", "E4 minus phase features", "E4", "phase_features"),
    AblationCase("A2", "E4 minus SLA penalty", "E4", "sla_penalty"),
    AblationCase("A3", "E4 with fixed power cap", "E4", "learned_power_cap"),
]

ABLATION_EXPERIMENTS = [
    ExperimentRow("A1", "HRL (no phase)", "Joint minus phase", "Learned", "Phase-feature ablation"),
    ExperimentRow("A2", "HRL (no SLA)", "Joint minus SLA", "Learned", "Latency tradeoff ablation"),
    ExperimentRow("A3", "HRL (fixed cap)", "Joint", "Partial", "Runtime-action ablation"),
]


def run_ablations(
    output_dir: Path = Path("outputs/eval"),
    checkpoints_root: Path = Path("checkpoints"),
) -> dict[str, object]:
    """Run ablation experiments and compute deltas against E4 baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(output_dir=output_dir, checkpoints_root=checkpoints_root)
    e4_metrics = runner._analytical_metrics_by_id("E4")

    rows: List[Dict[str, object]] = []
    for case in DEFAULT_ABLATIONS:
        ablation_metrics = runner._analytical_metrics_by_id(case.ablation_id)

        deltas = {}
        for key in e4_metrics:
            if isinstance(e4_metrics[key], (int, float)) and isinstance(ablation_metrics.get(key), (int, float)):
                deltas[f"delta_{key}"] = ablation_metrics[key] - e4_metrics[key]

        rows.append({
            "id": case.ablation_id,
            "description": case.description,
            "base_exp": case.base_exp_id,
            "disabled": case.disabled_component,
            "metrics": ablation_metrics,
            "deltas": deltas,
        })

    payload = {"baseline_id": "E4", "baseline_metrics": e4_metrics, "ablations": rows}
    save_json(output_dir / "ablation_results.json", payload)
    logger.info("Ablation results saved (%d cases)", len(rows))
    return payload
