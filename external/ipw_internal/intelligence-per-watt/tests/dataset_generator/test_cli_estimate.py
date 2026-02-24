"""Tests for Pipeline #1b CLI commands: estimate, compare-estimators, generate-lut."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    ModelSpec,
)
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_measurements() -> list[OperatorMeasurement]:
    """Generate synthetic profiling data for testing estimator training."""
    measurements = []
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [128, 256, 512, 1024, 2048]:
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6

            measurements.append(
                OperatorMeasurement(
                    operator_name="linear_qkv",
                    category=OperatorCategory.LINEAR,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.0,
                    energy_j=base_time * 400,
                    power_w=400.0,
                    flops=int(tokens * 2 * 4096 * 6144),
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_prefill",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.5,
                    energy_j=base_time * 1.5 * 400,
                    power_w=420.0,
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_decode",
                    category=OperatorCategory.ATTENTION_DECODE,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 0.3,
                    energy_j=base_time * 0.3 * 350,
                    power_w=350.0,
                )
            )
    return measurements


def _write_category_csvs(measurements: list[OperatorMeasurement], output_dir: Path) -> None:
    """Write measurements to per-category CSV files matching csv_category_map keys.

    The LUTGenerator.generate_full_bundle expects files named:
      linear.csv, attention_prefill.csv, attention_decode.csv, etc.
    """
    category_rows: dict[str, list[dict]] = {}
    for m in measurements:
        cat_name = m.category.value  # e.g., "linear", "attention_prefill"
        if cat_name not in category_rows:
            category_rows[cat_name] = []
        category_rows[cat_name].append({
            "operator_name": m.operator_name,
            "batch_size": m.batch_size,
            "seq_len": m.seq_len,
            "time_s": m.time_s,
            "energy_j": m.energy_j if m.energy_j is not None else "",
            "power_w": m.power_w if m.power_w is not None else "",
            "flops": m.flops if m.flops is not None else "",
        })

    fieldnames = [
        "operator_name", "batch_size", "seq_len",
        "time_s", "energy_j", "power_w", "flops",
    ]
    for cat_name, rows in category_rows.items():
        csv_path = output_dir / f"{cat_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_measurements():
    return _make_synthetic_measurements()


@pytest.fixture
def qwen3_8b_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen3-8B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=36,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=12288,
        vocab_size=151936,
    )


@pytest.fixture
def a100_spec() -> HardwareSpec:
    return HardwareSpec(
        name="A100-SXM4-80GB",
        vendor="nvidia",
        memory_gb=80,
        tdp_watts=400,
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        hbm_bandwidth_gb_s=2039.0,
        nvlink_bandwidth_gb_s=600.0,
        price_per_hour_usd=3.50,
    )


@pytest.fixture
def _skip_no_sklearn():
    pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# 4A: Test estimate command / LUT generation
# ---------------------------------------------------------------------------

class TestEstimateCLI:
    """Tests for the 'estimate' CLI command and LUT bundle generation."""

    def test_lut_generator_produces_bundle(
        self, _skip_no_sklearn, synthetic_measurements, qwen3_8b_spec, a100_spec
    ):
        """LUTGenerator.generate_full_bundle creates all required .npz files."""
        from inference_simulator.estimator.lut_generator import LUTGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_measurements, profiling_dir)
            assert (profiling_dir / "linear.csv").exists()
            assert (profiling_dir / "attention_prefill.csv").exists()
            assert (profiling_dir / "attention_decode.csv").exists()

            generator = LUTGenerator()
            bundle = generator.generate_full_bundle(
                profiling_dir, lut_dir, qwen3_8b_spec, a100_spec
            )

            assert bundle.exists()
            assert bundle.gpu_token_ops_lut.exists()
            assert bundle.gpu_attention_prefill_lut.exists()
            assert bundle.gpu_attention_decode_lut.exists()
            assert bundle.base_dir == lut_dir

    def test_lut_bundle_metadata(
        self, _skip_no_sklearn, synthetic_measurements, qwen3_8b_spec, a100_spec
    ):
        """LUT bundle metadata contains estimator class and training info."""
        from inference_simulator.estimator.lut_generator import LUTGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_measurements, profiling_dir)

            generator = LUTGenerator()
            bundle = generator.generate_full_bundle(
                profiling_dir, lut_dir, qwen3_8b_spec, a100_spec
            )

            assert "estimator_class" in bundle.metadata
            assert "training_scores" in bundle.metadata
            assert "num_measurements" in bundle.metadata
            assert bundle.metadata["num_measurements"] > 0

    def test_lut_npz_contents_valid(
        self, _skip_no_sklearn, synthetic_measurements, qwen3_8b_spec, a100_spec
    ):
        """Generated .npz files have correct structure and non-negative values."""
        import numpy as np
        from inference_simulator.estimator.lut_generator import LUTGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_measurements, profiling_dir)

            generator = LUTGenerator()
            bundle = generator.generate_full_bundle(
                profiling_dir, lut_dir, qwen3_8b_spec, a100_spec
            )

            # Check token ops LUT
            data = np.load(bundle.gpu_token_ops_lut, allow_pickle=True)
            assert "grid" in data
            assert "axis_names" in data
            grid = data["grid"]
            assert grid.ndim == 4
            assert grid.shape[-1] == 2  # (time_s, energy_j)
            assert np.all(grid[:, :, :, 0] >= 0)  # times non-negative

    def test_estimate_cli_via_runner(
        self, _skip_no_sklearn, synthetic_measurements, qwen3_8b_spec, a100_spec
    ):
        """Test the estimate CLI command via Click's CliRunner."""
        from click.testing import CliRunner
        from dataset_generator.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_measurements, profiling_dir)

            runner = CliRunner()
            result = runner.invoke(cli, [
                "estimate",
                "--profiling-dir", str(profiling_dir),
                "--output-dir", str(lut_dir),
                "--model", "Qwen/Qwen3-8B",
                "--hardware", "a100_80gb",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "LUT bundle generated" in result.output
            assert (lut_dir / "gpu_token_ops.npz").exists()

    def test_generate_lut_cli(
        self, _skip_no_sklearn, synthetic_measurements, qwen3_8b_spec, a100_spec
    ):
        """Test the generate-lut CLI command via Click's CliRunner."""
        from click.testing import CliRunner
        from dataset_generator.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_measurements, profiling_dir)

            runner = CliRunner()
            result = runner.invoke(cli, [
                "generate-lut",
                "--profiling-dir", str(profiling_dir),
                "--output-dir", str(lut_dir),
                "--model", "Qwen/Qwen3-8B",
                "--hardware", "a100_80gb",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "LUT bundle generated" in result.output


# ---------------------------------------------------------------------------
# 4B: Test compare-estimators command
# ---------------------------------------------------------------------------

class TestCompareEstimatorsCLI:
    """Tests for the 'compare-estimators' CLI command."""

    def test_compare_estimators_produces_results(
        self, _skip_no_sklearn, synthetic_measurements
    ):
        """compare_estimators returns results for RF, Ridge, KNN."""
        from inference_simulator.estimator.model_comparison import compare_estimators
        from inference_simulator.estimator.random_forest import RandomForestEstimator
        from inference_simulator.estimator.ridge import RidgeRegressionEstimator
        from inference_simulator.estimator.knn import KNNEstimator

        comparison = compare_estimators(
            synthetic_measurements, None,
            [RandomForestEstimator, RidgeRegressionEstimator, KNNEstimator],
        )

        assert len(comparison) == 3
        names = {entry["estimator"] for entry in comparison}
        assert "RandomForestEstimator" in names
        assert "RidgeRegressionEstimator" in names
        assert "KNNEstimator" in names

        # All should have time_r2 scores (no errors)
        for entry in comparison:
            assert "error" not in entry, f"{entry['estimator']} errored: {entry.get('error')}"
            assert "time_r2" in entry
            assert entry["time_r2"] > 0  # Should fit synthetic data well

    def test_pick_best_estimator_selects_highest_r2(
        self, _skip_no_sklearn, synthetic_measurements
    ):
        """pick_best_estimator selects the estimator with highest R^2."""
        from inference_simulator.estimator.model_comparison import (
            compare_estimators,
            pick_best_estimator,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator
        from inference_simulator.estimator.ridge import RidgeRegressionEstimator
        from inference_simulator.estimator.knn import KNNEstimator

        comparison = compare_estimators(
            synthetic_measurements, None,
            [RandomForestEstimator, RidgeRegressionEstimator, KNNEstimator],
        )

        best_name = pick_best_estimator(comparison, "time_r2")
        assert best_name  # non-empty string

        # Verify the best actually has the highest score
        best_r2 = max(
            entry["time_r2"]
            for entry in comparison
            if "time_r2" in entry
        )
        best_entry = next(e for e in comparison if e["estimator"] == best_name)
        assert best_entry["time_r2"] == best_r2

    def test_compare_estimators_cli(
        self, _skip_no_sklearn, synthetic_measurements
    ):
        """Test compare-estimators CLI command via CliRunner."""
        from click.testing import CliRunner
        from dataset_generator.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()

            _write_category_csvs(synthetic_measurements, profiling_dir)

            runner = CliRunner()
            result = runner.invoke(cli, [
                "compare-estimators",
                "--profiling-dir", str(profiling_dir),
                "--output-dir", str(Path(tmpdir) / "comparison"),
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "Comparing" in result.output
            assert "Best estimator" in result.output

    def test_compare_estimators_energy_metrics(
        self, _skip_no_sklearn, synthetic_measurements
    ):
        """compare_estimators includes energy R^2 when energy data present."""
        from inference_simulator.estimator.model_comparison import compare_estimators
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        comparison = compare_estimators(
            synthetic_measurements, None, [RandomForestEstimator],
        )

        assert len(comparison) == 1
        entry = comparison[0]
        # Synthetic data has energy_j values, so energy metrics should be present
        assert "energy_r2" in entry
