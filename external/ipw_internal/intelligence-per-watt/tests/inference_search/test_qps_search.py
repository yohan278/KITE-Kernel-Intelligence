"""Tests for inference_search.qps_search with mock simulator."""

from __future__ import annotations

from typing import Dict

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)

from inference_search.qps_search import search
from inference_search.types import SLAConstraint


def _make_model() -> ModelSpec:
    return ModelSpec(
        model_id="test/1b",
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
        name="GPU",
        vendor="test",
        memory_gb=80,
        tdp_watts=400,
        peak_fp16_tflops=312.0,
        hbm_bandwidth_gb_s=2000.0,
    )


class MockSimulator:
    """Mock simulator where latency scales linearly with QPS.

    TTFT = base_ttft * (1 + qps / max_capacity) to simulate queuing.
    """

    def __init__(self, base_ttft: float = 0.1, max_capacity: float = 100.0) -> None:
        self.base_ttft = base_ttft
        self.max_capacity = max_capacity
        self.call_count = 0

    def simulate(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        workload_spec: WorkloadSpec,
    ) -> Dict[str, float]:
        self.call_count += 1
        qps = workload_spec.qps
        rho = qps / self.max_capacity

        if rho >= 1.0:
            ttft = 100.0  # system overloaded
        else:
            ttft = self.base_ttft * (1.0 + rho / (1.0 - rho))

        return {
            "ttft_s": ttft,
            "tbt_s": 0.01,
            "e2e_latency_s": ttft + 0.01 * 200,
            "throughput_tps": min(qps, self.max_capacity) * 200,
            "throughput_rps": min(qps, self.max_capacity),
            "avg_power_w": 300.0,
        }


class TestQPSSearch:
    def test_convergence(self) -> None:
        """Search should converge to a max QPS under SLA."""
        sim = MockSimulator(base_ttft=0.1, max_capacity=100.0)
        constraints = [SLAConstraint("ttft_s", 1.0, "max")]

        result = search(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
            sla_constraints=constraints,
            simulator=sim,
            min_qps=0.1,
            max_qps=200.0,
            tolerance=0.5,
        )

        assert result.max_qps > 0
        assert result.sla_violations == []
        assert result.metrics["ttft_s"] <= 1.0

    def test_finds_max_qps(self) -> None:
        """Should find max QPS close to the theoretical limit."""
        sim = MockSimulator(base_ttft=0.1, max_capacity=100.0)
        # ttft <= 0.5 => 0.1 * (1 + rho/(1-rho)) <= 0.5 => rho <= 0.8 => qps <= 80
        constraints = [SLAConstraint("ttft_s", 0.5, "max")]

        result = search(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
            sla_constraints=constraints,
            simulator=sim,
            min_qps=0.1,
            max_qps=200.0,
            tolerance=1.0,
        )

        # Should be close to 80 QPS (within tolerance)
        assert 70.0 <= result.max_qps <= 85.0

    def test_min_qps_fails(self) -> None:
        """If even min QPS fails SLA, return max_qps=0."""
        sim = MockSimulator(base_ttft=10.0, max_capacity=100.0)
        constraints = [SLAConstraint("ttft_s", 1.0, "max")]

        result = search(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
            sla_constraints=constraints,
            simulator=sim,
        )

        assert result.max_qps == 0.0
        assert len(result.sla_violations) > 0

    def test_no_constraints(self) -> None:
        """With no SLA constraints, should reach max QPS."""
        sim = MockSimulator(base_ttft=0.1, max_capacity=100.0)

        result = search(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
            sla_constraints=[],
            simulator=sim,
            min_qps=0.1,
            max_qps=1000.0,
            tolerance=1.0,
        )

        # With no constraints, everything passes => max_qps should be near max_qps
        assert result.max_qps >= 999.0

    def test_multiple_constraints(self) -> None:
        """Multiple SLA constraints are all checked."""
        sim = MockSimulator(base_ttft=0.1, max_capacity=100.0)
        # throughput_tps = qps * 200, so min 10.0 => qps >= 0.05
        constraints = [
            SLAConstraint("ttft_s", 0.5, "max"),
            SLAConstraint("throughput_tps", 10.0, "min"),
        ]

        result = search(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
            sla_constraints=constraints,
            simulator=sim,
            min_qps=0.1,
            max_qps=200.0,
            tolerance=1.0,
        )

        assert result.max_qps > 0
        assert result.metrics["ttft_s"] <= 0.5
        assert result.metrics["throughput_tps"] >= 10.0

    def test_simulator_call_count(self) -> None:
        """Binary search should make a bounded number of calls."""
        sim = MockSimulator(base_ttft=0.1, max_capacity=100.0)
        constraints = [SLAConstraint("ttft_s", 1.0, "max")]

        search(
            model_spec=_make_model(),
            hardware_spec=_make_hw(),
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
            sla_constraints=constraints,
            simulator=sim,
            min_qps=0.1,
            max_qps=1000.0,
            tolerance=0.5,
        )

        # log2(1000/0.5) ~ 11 + 1 for initial check = ~12
        assert sim.call_count <= 25
