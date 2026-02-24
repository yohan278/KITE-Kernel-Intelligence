"""Tests for PerOperatorEstimator pipeline integration.

Covers:
- compare_estimators_by_category with synthetic measurements
- PerOperatorEstimator used in LUT generation
- CLI --per-category flag output format
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from click.testing import CliRunner

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


@pytest.fixture
def synthetic_measurements() -> List[OperatorMeasurement]:
    """Generate synthetic profiling measurements with multiple categories."""
    measurements: List[OperatorMeasurement] = []
    for batch_size in [1, 2, 4, 8, 16]:
        for seq_len in [128, 256, 512, 1024]:
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6

            measurements.append(OperatorMeasurement(
                operator_name="linear_qkv",
                category=OperatorCategory.LINEAR,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time,
                energy_j=base_time * 400, power_w=400.0,
            ))
            measurements.append(OperatorMeasurement(
                operator_name="attention_prefill",
                category=OperatorCategory.ATTENTION_PREFILL,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time * 1.5 * (seq_len / 128),
                energy_j=base_time * 1.5 * 400,
            ))
            measurements.append(OperatorMeasurement(
                operator_name="attention_decode",
                category=OperatorCategory.ATTENTION_DECODE,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time * 0.3,
                energy_j=base_time * 0.3 * 350,
            ))
            measurements.append(OperatorMeasurement(
                operator_name="rmsnorm",
                category=OperatorCategory.NORMALIZATION,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time * 0.1,
                energy_j=base_time * 0.1 * 300,
            ))
            measurements.append(OperatorMeasurement(
                operator_name="silu_activation",
                category=OperatorCategory.ACTIVATION,
                batch_size=batch_size, seq_len=seq_len,
                time_s=base_time * 0.05,
                energy_j=base_time * 0.05 * 280,
            ))
    return measurements


class TestCompareEstimatorsByCategory:
    @pytest.fixture(autouse=True)
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_returns_per_category_results(self, synthetic_measurements):
        from inference_simulator.estimator.model_comparison import (
            compare_estimators_by_category,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        results = compare_estimators_by_category(
            synthetic_measurements, None,
            [RandomForestEstimator],
            include_per_operator=True,
        )

        # Should have results for categories with >= 5 measurements
        assert len(results) > 0
        assert "linear" in results
        assert "attention_prefill" in results
        assert "attention_decode" in results

    def test_contains_per_operator_estimator(self, synthetic_measurements):
        from inference_simulator.estimator.model_comparison import (
            compare_estimators_by_category,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        results = compare_estimators_by_category(
            synthetic_measurements, None,
            [RandomForestEstimator],
            include_per_operator=True,
        )

        # Each category should have both RandomForest and PerOperatorEstimator
        for cat_name, estimator_metrics in results.items():
            assert "RandomForestEstimator" in estimator_metrics, (
                f"Missing RandomForestEstimator for {cat_name}"
            )
            assert "PerOperatorEstimator" in estimator_metrics, (
                f"Missing PerOperatorEstimator for {cat_name}"
            )

    def test_metrics_have_expected_keys(self, synthetic_measurements):
        from inference_simulator.estimator.model_comparison import (
            compare_estimators_by_category,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        results = compare_estimators_by_category(
            synthetic_measurements, None,
            [RandomForestEstimator],
            include_per_operator=True,
        )

        for cat_name, estimator_metrics in results.items():
            for est_name, metrics in estimator_metrics.items():
                if "error" not in metrics:
                    assert "time_r2" in metrics, f"Missing time_r2 for {est_name} in {cat_name}"
                    assert "time_mae" in metrics, f"Missing time_mae for {est_name} in {cat_name}"
                    assert "time_rmse" in metrics, f"Missing time_rmse for {est_name} in {cat_name}"

    def test_exclude_per_operator(self, synthetic_measurements):
        from inference_simulator.estimator.model_comparison import (
            compare_estimators_by_category,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        results = compare_estimators_by_category(
            synthetic_measurements, None,
            [RandomForestEstimator],
            include_per_operator=False,
        )

        for cat_name, estimator_metrics in results.items():
            assert "PerOperatorEstimator" not in estimator_metrics

    def test_skips_categories_with_few_measurements(self):
        from inference_simulator.estimator.model_comparison import (
            compare_estimators_by_category,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        # Only 3 measurements for LINEAR -- should be skipped
        sparse = [
            OperatorMeasurement(
                operator_name="linear_qkv",
                category=OperatorCategory.LINEAR,
                batch_size=bs, seq_len=128,
                time_s=bs * 128 * 1e-6,
            )
            for bs in [1, 2, 4]
        ]

        results = compare_estimators_by_category(
            sparse, None, [RandomForestEstimator],
        )
        assert "linear" not in results


class TestPerOperatorEstimatorInLUTGeneration:
    @pytest.fixture(autouse=True)
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_lut_generator_accepts_per_operator_estimator(self, synthetic_measurements):
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator

        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        gen = LUTGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Token ops LUT
            token_path = gen.generate_gpu_token_ops_lut(
                estimator=est,
                operators=[OperatorCategory.LINEAR],
                token_counts=[128, 256, 512],
                tp_sizes=[1],
                output_path=Path(tmpdir) / "token_ops.npz",
            )
            assert token_path.exists()
            data = np.load(token_path, allow_pickle=True)
            assert data["grid"].size > 0
            assert np.all(data["grid"][:, :, :, 0] > 0)

    def test_attention_prefill_lut_with_per_operator(self, synthetic_measurements):
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator

        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        gen = LUTGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.generate_attention_prefill_lut(
                estimator=est,
                seq_lens=[128, 256, 512],
                batch_tokens=[1, 2, 4],
                tp_sizes=[1],
                output_path=Path(tmpdir) / "attn_prefill.npz",
            )
            assert path.exists()
            data = np.load(path, allow_pickle=True)
            assert data["grid"].size > 0

    def test_attention_decode_lut_with_per_operator(self, synthetic_measurements):
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator

        est = PerOperatorEstimator()
        est.fit(synthetic_measurements)

        gen = LUTGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.generate_attention_decode_lut(
                estimator=est,
                kv_cache_sizes=[128, 256, 512],
                batch_sizes=[1, 2, 4],
                tp_sizes=[1],
                output_path=Path(tmpdir) / "attn_decode.npz",
            )
            assert path.exists()
            data = np.load(path, allow_pickle=True)
            assert data["grid"].size > 0


class TestCLIPerCategoryFlag:
    @pytest.fixture(autouse=True)
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_per_category_flag_accepted(self, synthetic_measurements, tmp_path):
        """Test that --per-category flag is accepted and produces output."""
        import csv
        from dataset_generator.cli import cli

        # Write synthetic measurements as CSVs
        for cat_name, cat_enum in [
            ("linear", OperatorCategory.LINEAR),
            ("attention_prefill", OperatorCategory.ATTENTION_PREFILL),
            ("attention_decode", OperatorCategory.ATTENTION_DECODE),
            ("normalization", OperatorCategory.NORMALIZATION),
            ("activation", OperatorCategory.ACTIVATION),
        ]:
            cat_ms = [m for m in synthetic_measurements if m.category == cat_enum]
            csv_path = tmp_path / f"{cat_name}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "operator_name", "batch_size", "seq_len",
                    "time_s", "energy_j", "power_w",
                ])
                writer.writeheader()
                for m in cat_ms:
                    writer.writerow({
                        "operator_name": m.operator_name,
                        "batch_size": m.batch_size,
                        "seq_len": m.seq_len,
                        "time_s": m.time_s,
                        "energy_j": m.energy_j if m.energy_j is not None else "",
                        "power_w": m.power_w if m.power_w is not None else "",
                    })

        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare-estimators",
            "--profiling-dir", str(tmp_path),
            "--per-category",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Per-Category Breakdown" in result.output
        assert "linear" in result.output.lower()
        assert "R²" in result.output or "R2" in result.output

    def test_without_per_category_flag(self, synthetic_measurements, tmp_path):
        """Test that without --per-category, no category breakdown is shown."""
        import csv
        from dataset_generator.cli import cli

        # Write a minimal CSV
        csv_path = tmp_path / "linear.csv"
        cat_ms = [m for m in synthetic_measurements if m.category == OperatorCategory.LINEAR]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "operator_name", "batch_size", "seq_len",
                "time_s", "energy_j", "power_w",
            ])
            writer.writeheader()
            for m in cat_ms:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                })

        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare-estimators",
            "--profiling-dir", str(tmp_path),
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Per-Category Breakdown" not in result.output
