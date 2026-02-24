"""Tests for dataset_generator CPU host profiler."""

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.cpu_host import CPUHostProfiler


@pytest.fixture
def small_model_spec():
    return ModelSpec(
        model_id="test/small-model",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=4,
        hidden_dim=256,
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=32,
        intermediate_dim=512,
        vocab_size=1000,
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
        prefill_seq_lengths=[64],
        warmup_iterations=1,
        measurement_iterations=2,
    )


EXPECTED_OPS = [
    "cpu_offload_transfer",
    "gpu_mem_alloc",
    "pcie_h2d_copy",
    "pcie_d2h_copy",
    "scheduler_overhead",
    "tokenizer_encode",
    "tokenizer_decode",
    "dynamic_batching_overhead",
]


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


class TestCPUHostProfiler:
    def test_category(self):
        profiler = CPUHostProfiler()
        assert profiler.category == OperatorCategory.CPU_HOST

    def test_sweep_dimensions(self):
        profiler = CPUHostProfiler()
        dims = profiler.get_sweep_dimensions()
        assert "batch_sizes" in dims
        assert "prefill_seq_lengths" in dims

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_profile_returns_measurements(self, small_model_spec, hw_spec, small_sweep):
        profiler = CPUHostProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        assert isinstance(measurements, list)
        assert len(measurements) > 0

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_all_operators_present(self, small_model_spec, hw_spec, small_sweep):
        profiler = CPUHostProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        op_names = {m.operator_name for m in measurements}
        for expected_op in EXPECTED_OPS:
            assert expected_op in op_names, f"Missing operator: {expected_op}"

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_positive_time(self, small_model_spec, hw_spec, small_sweep):
        profiler = CPUHostProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.time_s > 0, f"{m.operator_name} has non-positive time: {m.time_s}"

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available"
    )
    def test_all_cpu_host_category(self, small_model_spec, hw_spec, small_sweep):
        profiler = CPUHostProfiler()
        measurements = profiler.profile(small_model_spec, hw_spec, small_sweep)
        for m in measurements:
            assert m.category == OperatorCategory.CPU_HOST, (
                f"{m.operator_name} has wrong category: {m.category}"
            )
