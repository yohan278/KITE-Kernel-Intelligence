"""End-to-end tests for inference search with roofline oracle."""

from __future__ import annotations

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)

from inference_search.cli import run_search
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


def _make_hw(name: str = "H100", memory_gb: float = 80) -> HardwareSpec:
    return HardwareSpec(
        name=name,
        vendor="nvidia",
        memory_gb=memory_gb,
        tdp_watts=700,
        peak_fp16_tflops=989.4,
        peak_fp8_tflops=1978.9,
        peak_bf16_tflops=989.4,
        hbm_bandwidth_gb_s=3352.0,
        nvlink_bandwidth_gb_s=900.0,
    )


class TestRooflineOracle:
    def test_returns_all_metrics(self) -> None:
        oracle = RooflineOracle()
        metrics = oracle.simulate(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(qps=1.0),
        )
        expected_keys = {"ttft_s", "tbt_s", "e2e_latency_s", "throughput_tps",
                         "throughput_rps", "total_energy_j", "avg_power_w"}
        assert expected_keys.issubset(metrics.keys())

    def test_latency_increases_with_qps(self) -> None:
        oracle = RooflineOracle()
        model = _make_model()
        hw = _make_hw()
        inf = InferenceSpec()

        low = oracle.simulate(model, hw, inf, WorkloadSpec(qps=1.0))
        high = oracle.simulate(model, hw, inf, WorkloadSpec(qps=50.0))

        assert high["ttft_s"] > low["ttft_s"]
        assert high["e2e_latency_s"] > low["e2e_latency_s"]

    def test_positive_metrics(self) -> None:
        oracle = RooflineOracle()
        metrics = oracle.simulate(
            _make_model(), _make_hw(), InferenceSpec(),
            WorkloadSpec(qps=10.0),
        )
        for key, value in metrics.items():
            assert value >= 0, f"{key} should be non-negative, got {value}"


class TestEndToEnd:
    def test_single_config_search(self) -> None:
        config = SearchConfig(
            model_specs=[_make_model()],
            hardware_specs=[_make_hw()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[
                SLAConstraint("ttft_s", 1.0, "max"),
            ],
            optimization_targets=["throughput_tps"],
        )
        result = run_search(config)

        assert len(result.all_results) == 1
        assert result.all_results[0].max_qps > 0
        assert result.elapsed_seconds > 0

    def test_multi_config_search(self) -> None:
        """Search across multiple models and hardware."""
        model_small = _make_model("test/small")
        model_large = ModelSpec(
            model_id="test/large",
            architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.GQA,
            num_layers=60,
            hidden_dim=5120,
            num_attention_heads=40,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=13824,
            vocab_size=32000,
        )

        config = SearchConfig(
            model_specs=[model_small, model_large],
            hardware_specs=[_make_hw("H100", 80)],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[SLAConstraint("ttft_s", 2.0, "max")],
            optimization_targets=["throughput_tps", "avg_power_w"],
        )
        result = run_search(config)

        assert len(result.all_results) == 2
        assert len(result.pareto_frontier) >= 1

    def test_infeasible_filtered(self) -> None:
        """Infeasible configs should not appear in results."""
        # Make a model too big for the GPU
        huge_model = ModelSpec(
            model_id="huge/200b",
            architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.MHA,
            num_layers=120,
            hidden_dim=12288,
            num_attention_heads=96,
            num_kv_heads=96,
            head_dim=128,
            intermediate_dim=32768,
            vocab_size=100000,
        )

        config = SearchConfig(
            model_specs=[huge_model],
            hardware_specs=[_make_hw("Small", 24)],
            inference_specs=[InferenceSpec(num_gpus=1)],
            workload_spec=WorkloadSpec(),
            sla_constraints=[],
        )
        result = run_search(config)
        assert len(result.all_results) == 0
