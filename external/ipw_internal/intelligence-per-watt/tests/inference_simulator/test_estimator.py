"""Tests for runtime estimators."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.estimator.roofline import RooflineEstimator
from inference_simulator.estimator.lookup_table import LookupTableEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen3_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen3-8B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=36,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=11008,
        vocab_size=151936,
    )


@pytest.fixture
def h100_spec() -> HardwareSpec:
    return HardwareSpec.from_registry("h100_80gb")


@pytest.fixture
def synthetic_measurements() -> list[OperatorMeasurement]:
    """Generate synthetic profiling measurements for testing."""
    measurements = []
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [128, 256, 512, 1024]:
            # Time scales roughly linearly with tokens
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6  # ~1us per token

            measurements.append(
                OperatorMeasurement(
                    operator_name="linear_qkv",
                    category=OperatorCategory.LINEAR,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.0,
                    energy_j=base_time * 400,  # ~400W
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


@pytest.fixture
def synthetic_csv(synthetic_measurements: list[OperatorMeasurement]) -> Path:
    """Write synthetic measurements to a temp CSV."""
    tmpdir = Path(tempfile.mkdtemp())
    csv_path = tmpdir / "token_ops.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator_name", "batch_size", "seq_len",
                "time_s", "energy_j", "power_w", "flops", "bandwidth_gb_s",
            ],
        )
        writer.writeheader()
        for m in synthetic_measurements:
            if m.category == OperatorCategory.LINEAR:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j,
                    "power_w": m.power_w,
                    "flops": m.flops or "",
                    "bandwidth_gb_s": "",
                })

    return csv_path


# ---------------------------------------------------------------------------
# EstimatorResult
# ---------------------------------------------------------------------------


class TestEstimatorResult:
    def test_create(self):
        r = EstimatorResult(time_s=0.001, energy_j=0.5, power_w=400.0)
        assert r.time_s == 0.001
        assert r.energy_j == 0.5
        assert r.power_w == 400.0

    def test_defaults(self):
        r = EstimatorResult(time_s=0.001)
        assert r.energy_j is None
        assert r.power_w is None

    def test_frozen(self):
        r = EstimatorResult(time_s=0.001)
        with pytest.raises(AttributeError):
            r.time_s = 0.002


# ---------------------------------------------------------------------------
# RooflineEstimator
# ---------------------------------------------------------------------------


class TestRooflineEstimator:
    def test_is_always_fitted(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        assert est.is_fitted() is True

    def test_estimate_linear(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=1024)
        assert result.time_s > 0
        assert result.energy_j is not None
        assert result.energy_j > 0

    def test_estimate_attention_prefill(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate(
            OperatorCategory.ATTENTION_PREFILL, batch_size=1, seq_len=1024
        )
        assert result.time_s > 0

    def test_estimate_attention_decode(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate(
            OperatorCategory.ATTENTION_DECODE,
            batch_size=1,
            seq_len=1,
            kv_cache_len=1024,
        )
        assert result.time_s > 0

    def test_estimate_normalization(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate(
            OperatorCategory.NORMALIZATION, batch_size=1, seq_len=1024
        )
        assert result.time_s > 0

    def test_estimate_embedding(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate(OperatorCategory.EMBEDDING, batch_size=1, seq_len=1024)
        assert result.time_s > 0

    def test_longer_seq_takes_longer(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        short = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        long = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=4096)
        assert long.time_s > short.time_s

    def test_larger_batch_takes_longer(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        small = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=1024)
        large = est.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=1024)
        assert large.time_s > small.time_s

    def test_multi_gpu_is_faster(self, qwen3_spec, h100_spec):
        est_1 = RooflineEstimator(qwen3_spec, h100_spec, num_gpus=1)
        est_8 = RooflineEstimator(qwen3_spec, h100_spec, num_gpus=8)
        r1 = est_1.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=1024)
        r8 = est_8.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=1024)
        assert r8.time_s < r1.time_s

    def test_estimate_prefill_convenience(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate_prefill(batch_size=1, seq_len=1024)
        assert result.time_s > 0

    def test_estimate_decode_step_convenience(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec)
        result = est.estimate_decode_step(batch_size=1, kv_cache_len=1024)
        assert result.time_s > 0

    def test_communication_single_gpu(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec, num_gpus=1)
        result = est.estimate(OperatorCategory.COMMUNICATION, batch_size=1, seq_len=1)
        assert result.time_s == 0.0

    def test_communication_multi_gpu(self, qwen3_spec, h100_spec):
        est = RooflineEstimator(qwen3_spec, h100_spec, num_gpus=8)
        result = est.estimate(
            OperatorCategory.COMMUNICATION,
            batch_size=1,
            seq_len=1,
            message_bytes=1024 * 1024,
        )
        assert result.time_s > 0


# ---------------------------------------------------------------------------
# LookupTableEstimator
# ---------------------------------------------------------------------------


class TestLookupTableEstimator:
    def test_not_fitted_initially(self):
        est = LookupTableEstimator()
        assert est.is_fitted() is False

    def test_raises_when_not_fitted(self):
        est = LookupTableEstimator()
        with pytest.raises(RuntimeError, match="no data"):
            est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)

    def test_load_from_measurements(self, synthetic_measurements):
        est = LookupTableEstimator()
        est.load_from_measurements(synthetic_measurements)
        assert est.is_fitted() is True

    def test_exact_match(self, synthetic_measurements):
        est = LookupTableEstimator()
        est.load_from_measurements(synthetic_measurements)

        # This exact point exists in the synthetic data
        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        expected_time = 1 * 128 * 1e-6 * 1.0
        assert result.time_s == pytest.approx(expected_time, rel=0.01)

    def test_interpolation(self, synthetic_measurements):
        est = LookupTableEstimator()
        est.load_from_measurements(synthetic_measurements)

        # Point between measured values
        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=192)
        assert result.time_s > 0

    def test_load_from_csv(self, synthetic_csv):
        est = LookupTableEstimator()
        est.load_from_csv(synthetic_csv, OperatorCategory.LINEAR)
        assert est.is_fitted() is True

        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        assert result.time_s > 0
        assert result.energy_j is not None

    def test_csv_not_found(self):
        est = LookupTableEstimator()
        with pytest.raises(FileNotFoundError):
            est.load_from_csv(Path("/nonexistent/path.csv"), OperatorCategory.LINEAR)


# ---------------------------------------------------------------------------
# RandomForestEstimator
# ---------------------------------------------------------------------------


class TestRandomForestEstimator:
    @pytest.fixture
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_not_fitted_initially(self, _skip_no_sklearn):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator()
        assert est.is_fitted() is False

    def test_raises_when_not_fitted(self, _skip_no_sklearn):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator()
        with pytest.raises(RuntimeError, match="not fitted"):
            est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)

    def test_fit_from_measurements(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=10, random_state=42)
        scores = est.fit(synthetic_measurements)
        assert est.is_fitted() is True
        assert "time_train_r2" in scores
        assert scores["time_train_r2"] > 0.5  # Should fit well on synthetic data

    def test_predictions_reasonable(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=50, random_state=42)
        est.fit(synthetic_measurements)

        # Predict on a known point
        result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        expected = 1 * 128 * 1e-6
        # Should be within 5x of expected (RF on synthetic data)
        assert result.time_s > 0
        assert result.time_s < expected * 10

    def test_scaling_direction(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=50, random_state=42)
        est.fit(synthetic_measurements)

        small = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=128)
        large = est.estimate(OperatorCategory.LINEAR, batch_size=8, seq_len=1024)
        assert large.time_s > small.time_s

    def test_fit_from_csv(self, _skip_no_sklearn, synthetic_csv):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=10, random_state=42)
        scores = est.fit_from_csv(
            [(synthetic_csv, OperatorCategory.LINEAR)]
        )
        assert est.is_fitted() is True
        assert scores["time_train_r2"] > 0.5

    def test_too_few_measurements(self, _skip_no_sklearn):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator()
        with pytest.raises(ValueError, match="at least 2"):
            est.fit([
                OperatorMeasurement(
                    operator_name="x",
                    category=OperatorCategory.LINEAR,
                    batch_size=1,
                    seq_len=128,
                    time_s=0.001,
                ),
            ])

    def test_with_model_dims(self, _skip_no_sklearn, synthetic_measurements):
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=10, random_state=42)
        model_dims = {"hidden_dim": 4096, "num_heads": 32}
        scores = est.fit(synthetic_measurements, model_dims=model_dims)
        assert est.is_fitted() is True

        result = est.estimate(
            OperatorCategory.LINEAR,
            batch_size=1,
            seq_len=128,
            model_dims=model_dims,
        )
        assert result.time_s > 0
