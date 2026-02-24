"""Tests for dataset_generator token ops profiler."""

import pytest
from unittest.mock import patch, MagicMock

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.token_ops import TokenOpProfiler


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


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
        warmup_iterations=1,
        measurement_iterations=2,
    )


class TestTokenOpProfiler:
    def test_category(self):
        profiler = TokenOpProfiler()
        assert profiler.category == OperatorCategory.LINEAR

    def test_sweep_dimensions(self):
        profiler = TokenOpProfiler()
        dims = profiler.get_sweep_dimensions()
        assert "batch_sizes" in dims
        assert "prefill_seq_lengths" in dims

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_profile_returns_measurements(self, qwen3_spec, hw_spec, small_sweep):
        profiler = TokenOpProfiler()
        measurements = profiler.profile(qwen3_spec, hw_spec, small_sweep)
        assert isinstance(measurements, list)
        assert len(measurements) > 0
        assert all(isinstance(m, OperatorMeasurement) for m in measurements)

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_operator_names(self, qwen3_spec, hw_spec, small_sweep):
        profiler = TokenOpProfiler()
        measurements = profiler.profile(qwen3_spec, hw_spec, small_sweep)
        names = {m.operator_name for m in measurements}
        expected_names = {
            "linear_qkv", "linear_o", "mlp_up", "mlp_gate", "mlp_down",
            "rmsnorm", "silu_activation", "embedding", "lm_head",
            "layernorm", "gelu_activation", "residual_add",
            "rotary_embedding", "softmax", "dropout", "cross_entropy_loss",
        }
        assert names == expected_names

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_flops_computed(self, qwen3_spec, hw_spec, small_sweep):
        profiler = TokenOpProfiler()
        measurements = profiler.profile(qwen3_spec, hw_spec, small_sweep)
        linear_measurements = [m for m in measurements if m.operator_name == "linear_qkv"]
        assert len(linear_measurements) == 1
        m = linear_measurements[0]
        # FLOPs for QKV: 2 * batch * seq * hidden * (num_heads + 2*num_kv)*head_dim
        expected_flops = 2 * 1 * 128 * 4096 * (32 + 2 * 8) * 128
        assert m.flops == expected_flops
