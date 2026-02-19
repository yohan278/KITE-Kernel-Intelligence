"""Benchmark matrix runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from kite.utils.serialization import save_json


@dataclass(slots=True)
class ExperimentRow:
    exp_id: str
    policy: str
    reward: str
    runtime_control: str
    expected_output: str


DEFAULT_MATRIX: list[ExperimentRow] = [
    ExperimentRow("B0", "KernelBench default", "N/A", "Static", "Baseline correctness/speedup"),
    ExperimentRow("B1", "Qwen7B SFT", "Correctness only", "Static", "SFT-only baseline"),
    ExperimentRow("E1", "Qwen7B GRPO", "Correctness+speed", "Static", "Kevin-style performance"),
    ExperimentRow("E2", "Qwen7B GRPO", "Correctness+speed+energy", "Static", "Energy-aware kernel"),
    ExperimentRow("E3", "Runtime PPO", "APW/APJ + SLA", "Learned", "Runtime-only gains"),
    ExperimentRow("E4", "HRL", "Joint", "Learned", "Main Pareto shift"),
]


class BenchmarkRunner:
    def __init__(self, output_dir: Path = Path("outputs/eval")) -> None:
        self.output_dir = output_dir

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, object]] = []
        for row in DEFAULT_MATRIX:
            # Placeholder deterministic metrics for scaffolding.
            base = ord(row.exp_id[0]) + int(row.exp_id[1])
            results.append(
                {
                    "id": row.exp_id,
                    "policy": row.policy,
                    "reward": row.reward,
                    "runtime_control": row.runtime_control,
                    "expected_output": row.expected_output,
                    "metrics": {
                        "correctness": min(1.0, 0.7 + 0.03 * base),
                        "speedup": 1.0 + 0.2 * int(row.exp_id[1]),
                        "energy_per_token_j": max(0.01, 0.25 - 0.02 * int(row.exp_id[1])),
                        "ttft_p95_s": max(0.4, 2.5 - 0.2 * int(row.exp_id[1])),
                        "apj": 0.01 + 0.005 * int(row.exp_id[1]),
                        "apw": 0.001 + 0.0004 * int(row.exp_id[1]),
                    },
                }
            )

        summary = {
            "num_experiments": len(results),
            "results": results,
        }
        save_json(self.output_dir / "suite_results.json", summary)
        return summary
