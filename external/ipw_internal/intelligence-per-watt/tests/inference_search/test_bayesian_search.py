"""Tests for Bayesian search optimization."""

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

from inference_search.oracle import RooflineOracle
from inference_search.types import SLAConstraint, SearchConfig


def _make_model(model_id: str = "test/1b") -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
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


def _make_hw() -> HardwareSpec:
    return HardwareSpec(
        name="H100",
        vendor="nvidia",
        memory_gb=80,
        tdp_watts=700,
        peak_fp16_tflops=989.4,
        peak_fp8_tflops=1978.9,
        peak_bf16_tflops=989.4,
        hbm_bandwidth_gb_s=3352.0,
        nvlink_bandwidth_gb_s=900.0,
    )


class TestBayesianSearch:
    @pytest.fixture
    def _skip_no_bayes_opt(self):
        pytest.importorskip("bayes_opt")

    def test_basic_search(self, _skip_no_bayes_opt):
        from inference_search.bayesian_search import BayesianSearcher

        config = SearchConfig(
            model_specs=[_make_model()],
            hardware_specs=[_make_hw()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[SLAConstraint("ttft_s", 2.0, "max")],
            optimization_targets=["throughput_tps"],
            search_method="bayesian",
        )
        oracle = RooflineOracle()
        searcher = BayesianSearcher(oracle, config)
        results = searcher.search()

        assert len(results) >= 1

    def test_multi_model_search(self, _skip_no_bayes_opt):
        from inference_search.bayesian_search import BayesianSearcher

        config = SearchConfig(
            model_specs=[_make_model("test/small"), _make_model("test/large")],
            hardware_specs=[_make_hw()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[],
            optimization_targets=["throughput_tps", "avg_power_w"],
            search_method="bayesian",
        )
        oracle = RooflineOracle()
        searcher = BayesianSearcher(oracle, config)
        results = searcher.search()

        assert len(results) >= 1
