"""Tests for EventDrivenOracle in inference_search."""

import pytest

from inference_search.oracle import EventDrivenOracle, RooflineOracle, SimulatorOracle


class TestEventDrivenOracleCreation:
    def test_creation_no_lut(self):
        oracle = EventDrivenOracle()
        assert oracle.lut_bundle_dir is None
        assert oracle.accuracy_score == 1.0
        assert oracle.price_per_hour_usd == 0.0
        assert oracle.simulation_duration_s == 30.0

    def test_creation_with_params(self):
        oracle = EventDrivenOracle(
            accuracy_score=0.85,
            price_per_hour_usd=3.50,
            simulation_duration_s=10.0,
        )
        assert oracle.accuracy_score == 0.85
        assert oracle.price_per_hour_usd == 3.50
        assert oracle.simulation_duration_s == 10.0

    def test_implements_protocol(self):
        """EventDrivenOracle should match the SimulatorOracle protocol."""
        oracle = EventDrivenOracle()
        assert isinstance(oracle, SimulatorOracle)

    def test_lut_cache_initialized_empty(self):
        oracle = EventDrivenOracle()
        assert oracle._lut_cache == {}


class TestRooflineOracleProtocol:
    def test_implements_protocol(self):
        oracle = RooflineOracle()
        assert isinstance(oracle, SimulatorOracle)


class TestEventDrivenOracleLUTLoading:
    def test_load_missing_dir_returns_none(self, tmp_path):
        oracle = EventDrivenOracle(lut_bundle_dir=tmp_path / "nonexistent")
        from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec, WorkloadSpec
        from inference_simulator.types.model_spec import ArchitectureType, AttentionType

        model_spec = ModelSpec(
            model_id="test/test-7b",
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
        hardware_spec = HardwareSpec.from_registry("h100_80gb")
        inference_spec = InferenceSpec(num_gpus=1, precision="fp16")

        bundle = oracle._load_lut_bundle(model_spec, hardware_spec, inference_spec)
        assert bundle is None

    def test_lut_cache_reuses_result(self, tmp_path):
        oracle = EventDrivenOracle(lut_bundle_dir=tmp_path / "nonexistent")
        from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec
        from inference_simulator.types.model_spec import ArchitectureType, AttentionType

        model_spec = ModelSpec(
            model_id="test/test-7b",
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
        hardware_spec = HardwareSpec.from_registry("h100_80gb")
        inference_spec = InferenceSpec(num_gpus=1, precision="fp16")

        # First call populates cache
        oracle._load_lut_bundle(model_spec, hardware_spec, inference_spec)
        assert len(oracle._lut_cache) == 1
        # Second call hits cache
        oracle._load_lut_bundle(model_spec, hardware_spec, inference_spec)
        assert len(oracle._lut_cache) == 1
