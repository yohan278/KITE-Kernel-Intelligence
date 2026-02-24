"""Tests for the MTPProfiler."""

from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.mtp import MTPProfiler
from dataset_generator.profiler.sweep import SweepConfig

torch = pytest.importorskip("torch")


@pytest.fixture
def small_model_spec() -> ModelSpec:
    return ModelSpec(
        model_id="test/mtp-model",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=2,
        hidden_dim=64,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=16,
        intermediate_dim=128,
        vocab_size=1000,
    )


@pytest.fixture
def hw_spec() -> HardwareSpec:
    return HardwareSpec(
        name="test-gpu", vendor="nvidia", memory_gb=80,
        tdp_watts=400, peak_fp16_tflops=312.0,
    )


@pytest.fixture
def small_sweep() -> SweepConfig:
    return SweepConfig(
        batch_sizes=[1, 2],
        prefill_seq_lengths=[64, 128],
        warmup_iterations=1,
        measurement_iterations=2,
    )


EXPECTED_OPS = {
    "mtp_head_forward",
    "mtp_loss",
    "mtp_token_merge",
    "speculative_draft",
    "speculative_verify",
    "draft_accept_reject",
}


class TestMTPProfiler:
    def test_category(self):
        profiler = MTPProfiler()
        assert profiler.category == OperatorCategory.MTP

    def test_sweep_dimensions(self):
        profiler = MTPProfiler()
        dims = profiler.get_sweep_dimensions()
        assert dims == ["batch_sizes"]

    def test_profile_returns_measurements(
        self, small_model_spec, hw_spec, small_sweep
    ):
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        assert isinstance(measurements, list)
        assert len(measurements) > 0
        assert all(isinstance(m, OperatorMeasurement) for m in measurements)

    def test_operator_names(self, small_model_spec, hw_spec, small_sweep):
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        names = {m.operator_name for m in measurements}
        assert names == EXPECTED_OPS

    def test_all_measurements_have_positive_time(
        self, small_model_spec, hw_spec, small_sweep
    ):
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.time_s > 0, f"{m.operator_name} has non-positive time_s"

    def test_categories_correct(self, small_model_spec, hw_spec, small_sweep):
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.category == OperatorCategory.MTP

    def test_metadata_has_num_draft_tokens(
        self, small_model_spec, hw_spec, small_sweep
    ):
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert "num_draft_tokens" in m.metadata
            assert m.metadata["num_draft_tokens"] in [1, 2, 4, 8]

    def test_measurement_count(self, small_model_spec, hw_spec, small_sweep):
        """Should have 6 operators * 2 batch_sizes * 4 num_draft_tokens = 48 measurements."""
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        num_draft_values = len(MTPProfiler.DEFAULT_NUM_DRAFT_TOKENS)
        expected = (
            len(EXPECTED_OPS) * len(small_sweep.batch_sizes) * num_draft_values
        )
        assert len(measurements) == expected

    def test_draft_token_sweep(self, small_model_spec, hw_spec, small_sweep):
        """Verify internal sweep over num_draft_tokens."""
        profiler = MTPProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        draft_values = {m.metadata["num_draft_tokens"] for m in measurements}
        assert draft_values == {1, 2, 4, 8}
