"""Tests for phase regression analysis outputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset

from ipw.analysis.base import AnalysisContext
from ipw.analysis.phase_regression import PhaseRegressionAnalysis


def _write_dataset(tmp_path: Path, records: list[dict]) -> None:
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(tmp_path))


def test_phase_regression_models_energy(tmp_path: Path) -> None:
    records = [
        {
            "model_metrics": {
                "test-model": {
                    "token_metrics": {"input": 10, "output": 5},
                    "phase_metrics": {
                        "prefill_energy_j": 10.0,
                        "decode_energy_j": 20.0,
                    },
                }
            }
        },
        {
            "model_metrics": {
                "test-model": {
                    "token_metrics": {"input": 20, "output": 5},
                    "phase_metrics": {
                        "prefill_energy_j": 20.0,
                        "decode_energy_j": 20.0,
                    },
                }
            }
        },
        {
            "model_metrics": {
                "test-model": {
                    "token_metrics": {"input": 10, "output": 10},
                    "phase_metrics": {
                        "prefill_energy_j": 10.0,
                        "decode_energy_j": 40.0,
                    },
                }
            }
        },
    ]
    _write_dataset(tmp_path, records)

    analysis = PhaseRegressionAnalysis()
    result = analysis.run(AnalysisContext(results_dir=tmp_path))

    prefill = result.data["prefill_regression"]
    decode = result.data["decode_regression"]
    combined = result.data["combined_regression"]

    assert prefill["slope"] == pytest.approx(1.0)
    assert decode["slope"] == pytest.approx(4.0)
    assert combined["input_slope"] == pytest.approx(1.0)
    assert combined["output_slope"] == pytest.approx(4.0)
