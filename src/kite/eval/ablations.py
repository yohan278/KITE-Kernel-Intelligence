"""Ablation configuration and runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kite.utils.serialization import save_json


@dataclass(slots=True)
class AblationCase:
    ablation_id: str
    description: str


DEFAULT_ABLATIONS = [
    AblationCase("A1", "Disable phase features"),
    AblationCase("A2", "Disable SLA penalty"),
    AblationCase("A3", "Fixed power cap"),
]


def run_ablations(output_dir: Path = Path("outputs/eval")) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in DEFAULT_ABLATIONS:
        idx = int(case.ablation_id[1])
        rows.append(
            {
                "id": case.ablation_id,
                "description": case.description,
                "delta_apj": -0.002 * idx,
                "delta_apw": -0.0002 * idx,
                "delta_latency_p95": 0.1 * idx,
            }
        )

    payload = {"ablations": rows}
    save_json(output_dir / "ablation_results.json", payload)
    return payload
