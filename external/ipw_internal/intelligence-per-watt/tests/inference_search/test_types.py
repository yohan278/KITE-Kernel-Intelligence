"""Tests for inference_search.types."""

from __future__ import annotations

import pytest

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)

from inference_search.types import (
    ConfigurationResult,
    SLAConstraint,
    SearchConfig,
    SearchResult,
)


def _make_model() -> ModelSpec:
    return ModelSpec(
        model_id="test/model-1b",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=24,
        hidden_dim=2048,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_dim=5504,
        vocab_size=32000,
    )


def _make_hardware() -> HardwareSpec:
    return HardwareSpec(
        name="Test GPU",
        vendor="test",
        memory_gb=80,
        tdp_watts=400,
        peak_fp16_tflops=312.0,
        hbm_bandwidth_gb_s=2000.0,
    )


class TestSLAConstraint:
    def test_max_constraint(self) -> None:
        c = SLAConstraint(metric_name="ttft_s", threshold=1.0, direction="max")
        assert c.metric_name == "ttft_s"
        assert c.threshold == 1.0
        assert c.direction == "max"

    def test_min_constraint(self) -> None:
        c = SLAConstraint(metric_name="throughput_tps", threshold=10.0, direction="min")
        assert c.direction == "min"

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            SLAConstraint(metric_name="ttft_s", threshold=1.0, direction="bad")

    def test_frozen(self) -> None:
        c = SLAConstraint(metric_name="ttft_s", threshold=1.0, direction="max")
        with pytest.raises(AttributeError):
            c.threshold = 2.0  # type: ignore[misc]


class TestSearchConfig:
    def test_creation(self) -> None:
        config = SearchConfig(
            model_specs=[_make_model()],
            hardware_specs=[_make_hardware()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
        )
        assert len(config.model_specs) == 1
        assert len(config.hardware_specs) == 1
        assert config.duration_s == 60.0
        assert config.sla_constraints == []
        assert config.optimization_targets == ["ipj", "ipw", "cost_per_query_usd"]

    def test_with_sla(self) -> None:
        config = SearchConfig(
            model_specs=[_make_model()],
            hardware_specs=[_make_hardware()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[
                SLAConstraint("ttft_s", 1.0, "max"),
                SLAConstraint("throughput_tps", 10.0, "min"),
            ],
        )
        assert len(config.sla_constraints) == 2


class TestConfigurationResult:
    def test_creation(self) -> None:
        result = ConfigurationResult(
            model_spec=_make_model(),
            hardware_spec=_make_hardware(),
            inference_spec=InferenceSpec(),
            max_qps=42.5,
            metrics={"ttft_s": 0.5, "throughput_tps": 100.0},
        )
        assert result.max_qps == 42.5
        assert result.metrics["ttft_s"] == 0.5
        assert result.sla_violations == []

    def test_with_violations(self) -> None:
        result = ConfigurationResult(
            model_spec=_make_model(),
            hardware_spec=_make_hardware(),
            inference_spec=InferenceSpec(),
            max_qps=0.0,
            sla_violations=["ttft_s=2.0 exceeds max threshold 1.0"],
        )
        assert len(result.sla_violations) == 1


class TestSearchResult:
    def test_creation(self) -> None:
        config = SearchConfig(
            model_specs=[_make_model()],
            hardware_specs=[_make_hardware()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
        )
        result = SearchResult(
            all_results=[],
            pareto_frontier=[],
            search_config=config,
            total_simulations=100,
            elapsed_seconds=1.5,
        )
        assert result.total_simulations == 100
        assert result.elapsed_seconds == 1.5
