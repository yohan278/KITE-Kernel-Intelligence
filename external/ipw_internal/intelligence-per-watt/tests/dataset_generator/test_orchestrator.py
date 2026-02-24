"""Tests for PipelineOrchestrator: Pipeline #1b -> #2 -> #3."""
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
    """Generate synthetic profiling data for testing."""
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
    """Write measurements to per-category CSV files."""
    category_rows: dict[str, list[dict]] = {}
    for m in measurements:
        cat_name = m.category.value
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
def _skip_no_sklearn():
    pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# 4D: Test PipelineOrchestrator
# ---------------------------------------------------------------------------

class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator with synthetic data."""

    def test_run_pipeline_1b(self, _skip_no_sklearn):
        """Pipeline #1b: profiling CSVs -> LUT bundle."""
        from dataset_generator.pipeline.orchestrator import (
            PipelineConfig,
            PipelineOrchestrator,
        )

        measurements = _make_synthetic_measurements()

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(measurements, profiling_dir)

            config = PipelineConfig(
                model_id="Qwen/Qwen3-8B",
                hardware_key="a100_80gb",
                profiling_dir=profiling_dir,
                lut_dir=lut_dir,
            )

            model_spec = ModelSpec(
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
            hw_spec = HardwareSpec.from_registry("a100_80gb")

            orchestrator = PipelineOrchestrator()
            bundle = orchestrator.run_pipeline_1b(config, model_spec, hw_spec)

            assert bundle.exists()
            assert bundle.gpu_token_ops_lut.exists()
            assert bundle.gpu_attention_prefill_lut.exists()
            assert bundle.gpu_attention_decode_lut.exists()

    def test_run_pipeline_3(self, _skip_no_sklearn):
        """Pipeline #3: search with ML-backed oracle from LUT bundle."""
        from dataset_generator.pipeline.orchestrator import (
            PipelineConfig,
            PipelineOrchestrator,
        )

        measurements = _make_synthetic_measurements()

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(measurements, profiling_dir)

            config = PipelineConfig(
                model_id="Qwen/Qwen3-8B",
                hardware_key="a100_80gb",
                profiling_dir=profiling_dir,
                lut_dir=lut_dir,
                max_ttft=5.0,
                accuracy_score=0.85,
                price_per_gpu_hour_usd=3.50,
            )

            model_spec = ModelSpec(
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
            hw_spec = HardwareSpec.from_registry("a100_80gb")

            orchestrator = PipelineOrchestrator()

            # First generate LUT bundle (needed for pipeline 3)
            orchestrator.run_pipeline_1b(config, model_spec, hw_spec)

            # Then run search
            search_result = orchestrator.run_pipeline_3(config, model_spec, hw_spec)

            assert len(search_result.all_results) >= 1

    def test_run_all(self, _skip_no_sklearn):
        """PipelineOrchestrator.run_all() chains all three pipelines."""
        from dataset_generator.pipeline.orchestrator import (
            PipelineConfig,
            PipelineOrchestrator,
        )

        measurements = _make_synthetic_measurements()

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(measurements, profiling_dir)

            config = PipelineConfig(
                model_id="Qwen/Qwen3-8B",
                hardware_key="a100_80gb",
                profiling_dir=profiling_dir,
                lut_dir=lut_dir,
                accuracy_score=0.9,
                price_per_gpu_hour_usd=3.50,
                max_ttft=5.0,  # SLA needed so QPS search converges
            )

            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_all(config)

            # Verify all pipeline outputs present
            assert "lut_bundle" in result
            assert result["lut_bundle"].exists()
            assert "simulation_metrics" in result
            assert "search_result" in result
            assert len(result["search_result"].all_results) >= 1

    def test_pipeline_1b_missing_csvs(self, _skip_no_sklearn):
        """Pipeline #1b fails gracefully with empty profiling directory."""
        from dataset_generator.pipeline.orchestrator import (
            PipelineConfig,
            PipelineOrchestrator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            config = PipelineConfig(
                model_id="Qwen/Qwen3-8B",
                hardware_key="a100_80gb",
                profiling_dir=profiling_dir,
                lut_dir=lut_dir,
            )

            model_spec = ModelSpec(
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
            hw_spec = HardwareSpec.from_registry("a100_80gb")

            orchestrator = PipelineOrchestrator()

            with pytest.raises(FileNotFoundError):
                orchestrator.run_pipeline_1b(config, model_spec, hw_spec)
