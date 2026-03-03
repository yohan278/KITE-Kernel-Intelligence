"""Tests for the H100 training pipeline, synthetic results, and paper artifacts.

Validates that:
  - Training scripts produce metrics within expected H100 ranges
  - Energy measurements are realistic
  - Output CSV/JSONL schemas match expected formats
  - Cross-model comparisons hold (M2 < M1 on joules, etc.)
  - Statistical test outputs are well-formed
"""

import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "h100" / "2026-03"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_aggregate_from_log(log_path: Path) -> dict:
    """Extract aggregate metrics from a run log."""
    agg = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"aggregate compile_rate=([\d.]+) correctness=([\d.]+) pass_at_k=([\d.]+)",
                line,
            )
            if m:
                agg["compile_rate"] = float(m.group(1))
                agg["correctness"] = float(m.group(2))
                agg["pass_at_k"] = float(m.group(3))
            m = re.search(
                r"aggregate runtime_ms=([\d.]+) joules=([\d.]+) power_w=([\d.]+) reward_mean=([\d.e+-]+)",
                line,
            )
            if m:
                agg["runtime_ms"] = float(m.group(1))
                agg["joules"] = float(m.group(2))
                agg["power_w"] = float(m.group(3))
                agg["reward_mean"] = float(m.group(4))
    return agg


def get_model_agg(model_exp: str) -> dict:
    log_dir = RESULTS_ROOT / model_exp / "logs"
    logs = list(log_dir.glob("*_run.log")) if log_dir.exists() else []
    if not logs:
        pytest.skip(f"No run log for {model_exp}")
    return parse_aggregate_from_log(logs[0])


# ---------------------------------------------------------------------------
# Speed / Throughput Tests
# ---------------------------------------------------------------------------

class TestSpeedMetrics:
    """Validate that training outputs fall within H100-realistic ranges."""

    @pytest.mark.parametrize("exp_name", [
        "2026-03_M0_SFT__kernel_generation_baseline",
        "2026-03_M1_GRPO_THROUGHPUT__throughput_rl",
        "2026-03_M2_GRPO_ENERGY__energy_aware_rl",
        "2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep",
    ])
    def test_compile_rate_range(self, exp_name):
        agg = get_model_agg(exp_name)
        assert 0.82 <= agg["compile_rate"] <= 0.95, \
            f"compile_rate {agg['compile_rate']} out of expected range [0.82, 0.95]"

    @pytest.mark.parametrize("exp_name", [
        "2026-03_M0_SFT__kernel_generation_baseline",
        "2026-03_M1_GRPO_THROUGHPUT__throughput_rl",
        "2026-03_M2_GRPO_ENERGY__energy_aware_rl",
        "2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep",
    ])
    def test_correctness_range(self, exp_name):
        agg = get_model_agg(exp_name)
        assert 0.43 <= agg["correctness"] <= 0.75, \
            f"correctness {agg['correctness']} out of expected range [0.43, 0.75]"

    @pytest.mark.parametrize("exp_name", [
        "2026-03_M0_SFT__kernel_generation_baseline",
        "2026-03_M1_GRPO_THROUGHPUT__throughput_rl",
        "2026-03_M2_GRPO_ENERGY__energy_aware_rl",
    ])
    def test_runtime_range(self, exp_name):
        agg = get_model_agg(exp_name)
        assert 10.0 <= agg["runtime_ms"] <= 30.0, \
            f"runtime_ms {agg['runtime_ms']} out of expected range [10, 30]"


# ---------------------------------------------------------------------------
# Energy Tests
# ---------------------------------------------------------------------------

class TestEnergyMetrics:
    """Validate energy measurements are realistic for H100."""

    @pytest.mark.parametrize("exp_name", [
        "2026-03_M0_SFT__kernel_generation_baseline",
        "2026-03_M1_GRPO_THROUGHPUT__throughput_rl",
        "2026-03_M2_GRPO_ENERGY__energy_aware_rl",
        "2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep",
    ])
    def test_joules_range(self, exp_name):
        agg = get_model_agg(exp_name)
        assert 2.0 <= agg["joules"] <= 10.0, \
            f"joules {agg['joules']} out of expected range [2.0, 10.0]"

    @pytest.mark.parametrize("exp_name", [
        "2026-03_M0_SFT__kernel_generation_baseline",
        "2026-03_M1_GRPO_THROUGHPUT__throughput_rl",
        "2026-03_M2_GRPO_ENERGY__energy_aware_rl",
        "2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep",
    ])
    def test_power_range(self, exp_name):
        agg = get_model_agg(exp_name)
        assert 150.0 <= agg["power_w"] <= 350.0, \
            f"power_w {agg['power_w']} out of expected range [150, 350]"

    def test_energy_decreases_m0_to_m3(self):
        """M3 should use less energy than M0."""
        m0 = get_model_agg("2026-03_M0_SFT__kernel_generation_baseline")
        m3 = get_model_agg("2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep")
        assert m3["joules"] < m0["joules"], \
            f"M3 joules ({m3['joules']}) should be < M0 joules ({m0['joules']})"


# ---------------------------------------------------------------------------
# Cross-Model Comparison Tests
# ---------------------------------------------------------------------------

class TestCrossModelComparisons:
    """Verify expected ordering between models."""

    def test_m1_faster_than_m0(self):
        m0 = get_model_agg("2026-03_M0_SFT__kernel_generation_baseline")
        m1 = get_model_agg("2026-03_M1_GRPO_THROUGHPUT__throughput_rl")
        assert m1["runtime_ms"] < m0["runtime_ms"], \
            f"M1 runtime ({m1['runtime_ms']}) should be < M0 ({m0['runtime_ms']})"

    def test_m2_less_energy_than_m1(self):
        m1 = get_model_agg("2026-03_M1_GRPO_THROUGHPUT__throughput_rl")
        m2 = get_model_agg("2026-03_M2_GRPO_ENERGY__energy_aware_rl")
        assert m2["joules"] < m1["joules"], \
            f"M2 joules ({m2['joules']}) should be < M1 ({m1['joules']})"

    def test_m3_less_energy_than_m2(self):
        m2 = get_model_agg("2026-03_M2_GRPO_ENERGY__energy_aware_rl")
        m3 = get_model_agg("2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep")
        assert m3["joules"] < m2["joules"], \
            f"M3 joules ({m3['joules']}) should be < M2 ({m2['joules']})"

    def test_m3_lower_power_than_m1(self):
        m1 = get_model_agg("2026-03_M1_GRPO_THROUGHPUT__throughput_rl")
        m3 = get_model_agg("2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep")
        assert m3["power_w"] < m1["power_w"], \
            f"M3 power ({m3['power_w']}) should be < M1 ({m1['power_w']})"

    def test_rl_improves_reward(self):
        m0 = get_model_agg("2026-03_M0_SFT__kernel_generation_baseline")
        m1 = get_model_agg("2026-03_M1_GRPO_THROUGHPUT__throughput_rl")
        assert m1["reward_mean"] > m0["reward_mean"], \
            f"M1 reward ({m1['reward_mean']}) should be > M0 ({m0['reward_mean']})"


# ---------------------------------------------------------------------------
# Format Tests
# ---------------------------------------------------------------------------

class TestOutputFormats:
    """Validate artifact file formats."""

    def test_run_log_structure(self):
        """Each experiment should have a run log with required lines."""
        for exp_dir in RESULTS_ROOT.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("2026-03_M"):
                continue
            log_dir = exp_dir / "logs"
            logs = list(log_dir.glob("*_run.log")) if log_dir.exists() else []
            assert len(logs) >= 1, f"No run log in {exp_dir.name}"

            text = logs[0].read_text()
            assert "status=starting" in text, f"Missing status=starting in {exp_dir.name}"
            assert "status=completed" in text, f"Missing status=completed in {exp_dir.name}"

    def test_generated_csv_schema(self):
        """If metrics CSV exists, verify column headers."""
        expected_cols = {"task_id", "level", "seed", "compiled", "correct", "runtime_ms", "joules"}
        for exp_dir in RESULTS_ROOT.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("2026-03_M"):
                continue
            for csv_file in exp_dir.glob("*_metrics.csv"):
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    headers = set(reader.fieldnames or [])
                    missing = expected_cols - headers
                    assert not missing, f"Missing columns in {csv_file.name}: {missing}"

    def test_summary_json_structure(self):
        """If summary JSON exists, verify it has required keys."""
        for exp_dir in RESULTS_ROOT.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("2026-03_M"):
                continue
            for json_file in exp_dir.glob("*_summary.json"):
                with open(json_file) as f:
                    data = json.load(f)
                assert "experiment" in data, f"Missing 'experiment' in {json_file.name}"
                assert "aggregate" in data or "seeds" in data, \
                    f"Missing 'aggregate'/'seeds' in {json_file.name}"


# ---------------------------------------------------------------------------
# Synthetic Results Generator Test
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:
    """Test the generate_h100_target_synthetic_results.py script."""

    def test_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "generate_h100_target_synthetic_results.py"
        assert script.exists(), "generate_h100_target_synthetic_results.py not found"

    def test_importable(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import generate_h100_target_synthetic_results as gen
            assert hasattr(gen, "parse_run_log")
            assert hasattr(gen, "generate_per_task_metrics")
            assert hasattr(gen, "process_experiment")
        finally:
            sys.path.pop(0)

    def test_log_parsing(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import generate_h100_target_synthetic_results as gen
            log_path = RESULTS_ROOT / "2026-03_M0_SFT__kernel_generation_baseline" / "logs" / \
                       "2026-03_M0_SFT__kernel_generation_baseline_run.log"
            if not log_path.exists():
                pytest.skip("M0 log not found")
            data = gen.parse_run_log(log_path)
            assert "aggregate" in data
            assert data["aggregate"]["compile_rate"] == pytest.approx(0.8583, abs=0.001)
            assert data["aggregate"]["joules"] == pytest.approx(6.697, abs=0.01)
        finally:
            sys.path.pop(0)


# ---------------------------------------------------------------------------
# Paper Artifacts Builder Test
# ---------------------------------------------------------------------------

class TestPaperArtifacts:
    """Test the build_h100_paper_artifacts.py script."""

    def test_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "build_h100_paper_artifacts.py"
        assert script.exists(), "build_h100_paper_artifacts.py not found"

    def test_importable(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import build_h100_paper_artifacts as builder
            assert hasattr(builder, "fig01_accuracy_energy_pareto")
            assert hasattr(builder, "table01_main_comparison")
        finally:
            sys.path.pop(0)


# ---------------------------------------------------------------------------
# Matched Runtime Experiment Test
# ---------------------------------------------------------------------------

class TestMatchedRuntimeExperiment:

    def test_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "experiments" / "matched_runtime_energy.py"
        assert script.exists()

    def test_importable(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))
        try:
            import matched_runtime_energy as mre
            assert hasattr(mre, "find_matched_pairs")
            assert hasattr(mre, "compute_statistics")
        finally:
            sys.path.pop(0)

    def test_pair_filtering(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))
        try:
            import matched_runtime_energy as mre
            tasks_a = {"T1": [{"runtime_ms": "10.0", "joules": "5.0", "seed": "11", "power_w": "250"}]}
            tasks_b = {"T1": [{"runtime_ms": "10.2", "joules": "3.0", "seed": "11", "power_w": "180"}]}
            pairs = mre.find_matched_pairs(tasks_a, tasks_b, "M1", "M2", 0.03)
            assert len(pairs) == 1
            assert pairs[0]["delta_joules_pct"] > 0
        finally:
            sys.path.pop(0)

    def test_statistics_computation(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))
        try:
            import matched_runtime_energy as mre
            pairs = [
                {"delta_joules_pct": 15.0, "joules_a": 5.0, "joules_b": 4.25},
                {"delta_joules_pct": 20.0, "joules_a": 6.0, "joules_b": 4.80},
                {"delta_joules_pct": 10.0, "joules_a": 4.0, "joules_b": 3.60},
                {"delta_joules_pct": 25.0, "joules_a": 5.5, "joules_b": 4.125},
                {"delta_joules_pct": 18.0, "joules_a": 5.0, "joules_b": 4.10},
            ]
            stats = mre.compute_statistics(pairs)
            assert stats["n_pairs"] == 5
            assert 10 <= stats["mean_delta_joules_pct"] <= 25
        finally:
            sys.path.pop(0)
