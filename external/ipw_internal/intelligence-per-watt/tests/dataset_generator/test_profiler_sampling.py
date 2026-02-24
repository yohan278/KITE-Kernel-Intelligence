"""Tests for the SamplingProfiler."""

from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.sampling import SamplingProfiler
from dataset_generator.profiler.sweep import SweepConfig

torch = pytest.importorskip("torch")


@pytest.fixture
def small_model_spec() -> ModelSpec:
    return ModelSpec(
        model_id="test/sampling-model",
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
    "temperature_scaling",
    "top_p_filter",
    "top_k_filter",
    "repetition_penalty",
    "logit_processor_chain",
    "multinomial_sample",
    "beam_search_step",
}


class TestSamplingProfiler:
    def test_category(self):
        profiler = SamplingProfiler()
        assert profiler.category == OperatorCategory.SAMPLING

    def test_sweep_dimensions(self):
        profiler = SamplingProfiler()
        dims = profiler.get_sweep_dimensions()
        assert dims == ["batch_sizes"]

    def test_profile_returns_measurements(self, small_model_spec, hw_spec, small_sweep):
        profiler = SamplingProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        assert isinstance(measurements, list)
        assert len(measurements) > 0
        assert all(isinstance(m, OperatorMeasurement) for m in measurements)

    def test_operator_names(self, small_model_spec, hw_spec, small_sweep):
        profiler = SamplingProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        names = {m.operator_name for m in measurements}
        assert names == EXPECTED_OPS

    def test_all_measurements_have_positive_time(
        self, small_model_spec, hw_spec, small_sweep
    ):
        profiler = SamplingProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.time_s > 0, f"{m.operator_name} has non-positive time_s"

    def test_seq_len_is_one(self, small_model_spec, hw_spec, small_sweep):
        """Sampling happens at decode time, so seq_len should always be 1."""
        profiler = SamplingProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.seq_len == 1

    def test_categories_correct(self, small_model_spec, hw_spec, small_sweep):
        profiler = SamplingProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.category == OperatorCategory.SAMPLING

    def test_measurement_count(self, small_model_spec, hw_spec, small_sweep):
        """Should have 7 operators * 2 batch sizes = 14 measurements."""
        profiler = SamplingProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        assert len(measurements) == len(EXPECTED_OPS) * len(small_sweep.batch_sizes)
