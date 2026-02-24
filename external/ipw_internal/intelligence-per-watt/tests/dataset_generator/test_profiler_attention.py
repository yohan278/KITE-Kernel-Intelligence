"""Tests for dataset_generator attention profiler."""

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.attention import AttentionProfiler


@pytest.fixture
def qwen3_spec():
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
def hw_spec():
    return HardwareSpec(
        name="test-gpu", vendor="nvidia", memory_gb=80,
        tdp_watts=400, peak_fp16_tflops=312.0,
    )


@pytest.fixture
def small_sweep():
    return SweepConfig(
        batch_sizes=[1],
        prefill_seq_lengths=[128],
        kv_cache_sizes=[256],
        warmup_iterations=1,
        measurement_iterations=2,
    )


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


class TestAttentionProfiler:
    def test_category(self):
        profiler = AttentionProfiler()
        assert profiler.category == OperatorCategory.ATTENTION_PREFILL

    def test_sweep_dimensions(self):
        profiler = AttentionProfiler()
        dims = profiler.get_sweep_dimensions()
        assert "batch_sizes" in dims
        assert "prefill_seq_lengths" in dims

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_profile_returns_measurements(self, qwen3_spec, hw_spec, small_sweep):
        profiler = AttentionProfiler()
        measurements = profiler.profile(qwen3_spec, hw_spec, small_sweep)
        assert isinstance(measurements, list)
        assert len(measurements) > 0

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_prefill_and_decode_present(self, qwen3_spec, hw_spec, small_sweep):
        profiler = AttentionProfiler()
        measurements = profiler.profile(qwen3_spec, hw_spec, small_sweep)
        categories = {m.category for m in measurements}
        assert OperatorCategory.ATTENTION_PREFILL in categories
        assert OperatorCategory.ATTENTION_DECODE in categories

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_decode_has_kv_cache_metadata(self, qwen3_spec, hw_spec, small_sweep):
        profiler = AttentionProfiler()
        measurements = profiler.profile(qwen3_spec, hw_spec, small_sweep)
        decode_measurements = [
            m for m in measurements
            if m.category == OperatorCategory.ATTENTION_DECODE
        ]
        assert len(decode_measurements) > 0
        for m in decode_measurements:
            assert "kv_cache_size" in m.metadata
