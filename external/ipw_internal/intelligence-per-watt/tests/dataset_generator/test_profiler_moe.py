"""Tests for the MoE profiler."""

from __future__ import annotations

import pytest

from inference_simulator.types.operators import OperatorCategory
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.moe import MoEProfiler
from dataset_generator.profiler.sweep import SweepConfig

torch = pytest.importorskip("torch")


@pytest.fixture
def moe_spec() -> ModelSpec:
    """Mixtral-style MoE model spec."""
    return ModelSpec(
        model_id="test/moe-model",
        architecture_type=ArchitectureType.MOE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=14336,
        vocab_size=32000,
        num_experts=8,
        experts_per_token=2,
    )


@pytest.fixture
def h100_spec() -> HardwareSpec:
    return HardwareSpec.from_registry("h100_80gb")


@pytest.fixture
def small_sweep() -> SweepConfig:
    return SweepConfig(
        batch_sizes=[1, 2],
        prefill_seq_lengths=[128, 256],
        warmup_iterations=1,
        measurement_iterations=2,
    )


class TestMoEProfiler:
    def test_category(self):
        profiler = MoEProfiler()
        assert profiler.category == OperatorCategory.MOE_ROUTING

    def test_sweep_dimensions(self):
        profiler = MoEProfiler()
        dims = profiler.get_sweep_dimensions()
        assert "batch_sizes" in dims
        assert "prefill_seq_lengths" in dims

    def test_profile_returns_measurements(self, moe_spec, h100_spec, small_sweep):
        profiler = MoEProfiler()
        measurements = profiler.profile(moe_spec, h100_spec, small_sweep)

        assert len(measurements) > 0

        # Should have router, expert, and combine measurements
        op_names = {m.operator_name for m in measurements}
        assert "moe_router" in op_names
        assert "moe_expert_mlp" in op_names
        assert "moe_combine" in op_names

    def test_categories_correct(self, moe_spec, h100_spec, small_sweep):
        profiler = MoEProfiler()
        measurements = profiler.profile(moe_spec, h100_spec, small_sweep)

        for m in measurements:
            if m.operator_name == "moe_router":
                assert m.category == OperatorCategory.MOE_ROUTING
            elif m.operator_name in ("moe_expert_mlp", "moe_combine"):
                assert m.category == OperatorCategory.MOE_EXPERT

    def test_measurements_have_timing(self, moe_spec, h100_spec, small_sweep):
        profiler = MoEProfiler()
        measurements = profiler.profile(moe_spec, h100_spec, small_sweep)

        for m in measurements:
            assert m.time_s > 0

    def test_metadata_includes_expert_info(self, moe_spec, h100_spec, small_sweep):
        profiler = MoEProfiler()
        measurements = profiler.profile(moe_spec, h100_spec, small_sweep)

        router_ms = [m for m in measurements if m.operator_name == "moe_router"]
        assert len(router_ms) > 0
        assert router_ms[0].metadata.get("num_experts") == 8
        assert router_ms[0].metadata.get("experts_per_token") == 2

    def test_dense_model_uses_defaults(self, h100_spec, small_sweep):
        """A dense model (no MoE fields) should use default expert counts."""
        dense_spec = ModelSpec(
            model_id="test/dense",
            architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.GQA,
            num_layers=32,
            hidden_dim=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=11008,
            vocab_size=32000,
        )
        profiler = MoEProfiler()
        measurements = profiler.profile(dense_spec, h100_spec, small_sweep)
        # Should still work with default num_experts=8
        assert len(measurements) > 0
