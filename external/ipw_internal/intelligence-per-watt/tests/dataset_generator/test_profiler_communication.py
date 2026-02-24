"""Tests for the communication profiler."""

from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.communication import CommunicationProfiler
from dataset_generator.profiler.sweep import SweepConfig


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
def small_sweep() -> SweepConfig:
    return SweepConfig(
        message_sizes_bytes=[1024, 65536],
        gpu_topologies=[2, 4],
        warmup_iterations=1,
        measurement_iterations=2,
    )


class TestCommunicationProfiler:
    def test_category(self):
        profiler = CommunicationProfiler()
        assert profiler.category == OperatorCategory.COMMUNICATION

    def test_sweep_dimensions(self):
        profiler = CommunicationProfiler()
        dims = profiler.get_sweep_dimensions()
        assert "message_sizes_bytes" in dims
        assert "gpu_topologies" in dims

    def test_analytical_fallback(self, qwen3_spec, h100_spec, small_sweep):
        """Test that profiling works with analytical fallback (no multi-GPU)."""
        profiler = CommunicationProfiler()
        measurements = profiler._profile_analytical(qwen3_spec, h100_spec, small_sweep)

        # Should have measurements for allreduce and allgather at each (msg_size, num_gpus > 1)
        assert len(measurements) > 0

        for m in measurements:
            assert m.category == OperatorCategory.COMMUNICATION
            assert m.time_s > 0
            assert m.metadata.get("analytical") is True

    def test_analytical_allreduce_vs_allgather(self, qwen3_spec, h100_spec):
        """AllReduce should take ~2x AllGather for same message size."""
        profiler = CommunicationProfiler()
        ar = profiler._analytical_measurement("allreduce", 1024 * 1024, 8, h100_spec)
        ag = profiler._analytical_measurement("allgather", 1024 * 1024, 8, h100_spec)
        assert ar is not None and ag is not None
        assert ar.time_s > ag.time_s
        assert ar.time_s == pytest.approx(ag.time_s * 2, rel=0.01)

    def test_larger_message_takes_longer(self, qwen3_spec, h100_spec):
        profiler = CommunicationProfiler()
        small = profiler._analytical_measurement("allreduce", 1024, 4, h100_spec)
        large = profiler._analytical_measurement("allreduce", 1024 * 1024, 4, h100_spec)
        assert small is not None and large is not None
        assert large.time_s > small.time_s

    def test_more_gpus_changes_time(self, qwen3_spec, h100_spec):
        profiler = CommunicationProfiler()
        r2 = profiler._analytical_measurement("allreduce", 65536, 2, h100_spec)
        r8 = profiler._analytical_measurement("allreduce", 65536, 8, h100_spec)
        assert r2 is not None and r8 is not None
        # 8 GPUs: ring_factor = 2*7/8 = 1.75, 2 GPUs: ring_factor = 2*1/2 = 1.0
        assert r8.time_s > r2.time_s

    def test_single_gpu_skipped(self, qwen3_spec, h100_spec):
        """Single GPU topology should produce no measurements."""
        sweep = SweepConfig(
            message_sizes_bytes=[1024],
            gpu_topologies=[1],
            warmup_iterations=1,
            measurement_iterations=2,
        )
        profiler = CommunicationProfiler()
        measurements = profiler._profile_analytical(qwen3_spec, h100_spec, sweep)
        assert len(measurements) == 0

    def test_bandwidth_recorded(self, qwen3_spec, h100_spec):
        profiler = CommunicationProfiler()
        m = profiler._analytical_measurement("allreduce", 1024 * 1024, 4, h100_spec)
        assert m is not None
        assert m.bandwidth_gb_s is not None
        assert m.bandwidth_gb_s > 0
