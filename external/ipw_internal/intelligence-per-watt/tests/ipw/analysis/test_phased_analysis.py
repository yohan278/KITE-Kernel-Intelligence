"""Tests for phased analysis outputs."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from ipw.analysis.base import AnalysisContext
from ipw.analysis.phased import PhasedAnalysis


def _write_dataset(tmp_path: Path, records: list[dict]) -> None:
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(tmp_path))


def test_phased_analysis_summarizes_energy(tmp_path: Path) -> None:
    records = [
        {
            "model_metrics": {
                "test-model": {
                    "phase_metrics": {
                        "prefill_energy_j": 10.0,
                        "decode_energy_j": 30.0,
                        "prefill_power_avg_w": 20.0,
                        "decode_power_avg_w": 15.0,
                        "prefill_duration_ms": 500.0,
                        "decode_duration_ms": 1500.0,
                        "prefill_energy_per_input_token_j": 1.0,
                        "decode_energy_per_output_token_j": 6.0,
                    }
                }
            }
        },
        {
            "model_metrics": {
                "test-model": {
                    "phase_metrics": {
                        "prefill_energy_j": 5.0,
                        "decode_energy_j": 15.0,
                        "prefill_power_avg_w": 10.0,
                        "decode_power_avg_w": 12.0,
                        "prefill_duration_ms": 400.0,
                        "decode_duration_ms": 1200.0,
                        "prefill_energy_per_input_token_j": 0.5,
                        "decode_energy_per_output_token_j": 3.0,
                    }
                }
            }
        },
    ]
    _write_dataset(tmp_path, records)

    analysis = PhasedAnalysis()
    result = analysis.run(AnalysisContext(results_dir=tmp_path))

    assert result.summary["prefill_total_energy_j"] == 15.0
    assert result.summary["decode_total_energy_j"] == 45.0
    assert "report" in result.artifacts
