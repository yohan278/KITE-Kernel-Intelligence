"""Tests for inference_search.pareto."""

from __future__ import annotations

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
)

from inference_search.pareto import compute, compute_2d
from inference_search.types import ConfigurationResult


def _make_result(metrics: dict[str, float], model_id: str = "test") -> ConfigurationResult:
    return ConfigurationResult(
        model_spec=ModelSpec(
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
        ),
        hardware_spec=HardwareSpec(
            name="GPU",
            vendor="test",
            memory_gb=80,
            tdp_watts=400,
            peak_fp16_tflops=312.0,
        ),
        inference_spec=InferenceSpec(),
        max_qps=10.0,
        metrics=metrics,
    )


class TestParetoFrontier:
    def test_single_result(self) -> None:
        """Single result is always on the frontier."""
        results = [_make_result({"cost": 10.0, "latency": 1.0})]
        frontier = compute(results, [("cost", "minimize"), ("latency", "minimize")])
        assert len(frontier) == 1

    def test_dominated_filtered(self) -> None:
        """A result dominated on all objectives should be filtered."""
        a = _make_result({"cost": 5.0, "latency": 0.5}, "a")
        b = _make_result({"cost": 10.0, "latency": 1.0}, "b")  # dominated by a
        frontier = compute([a, b], [("cost", "minimize"), ("latency", "minimize")])
        assert len(frontier) == 1
        assert frontier[0].model_spec.model_id == "a"

    def test_non_dominated_kept(self) -> None:
        """Results that trade off objectives should both be kept."""
        a = _make_result({"cost": 5.0, "latency": 2.0}, "a")   # cheap but slow
        b = _make_result({"cost": 10.0, "latency": 0.5}, "b")  # expensive but fast
        frontier = compute([a, b], [("cost", "minimize"), ("latency", "minimize")])
        assert len(frontier) == 2

    def test_three_points_with_one_dominated(self) -> None:
        """Three points where one is dominated by another."""
        a = _make_result({"cost": 5.0, "latency": 1.0}, "a")
        b = _make_result({"cost": 10.0, "latency": 2.0}, "b")  # dominated by a
        c = _make_result({"cost": 3.0, "latency": 3.0}, "c")   # tradeoff with a
        frontier = compute([a, b, c], [("cost", "minimize"), ("latency", "minimize")])
        assert len(frontier) == 2
        ids = {r.model_spec.model_id for r in frontier}
        assert ids == {"a", "c"}

    def test_maximize_direction(self) -> None:
        """Test maximize objective direction."""
        a = _make_result({"throughput": 100.0, "power": 300.0}, "a")
        b = _make_result({"throughput": 50.0, "power": 200.0}, "b")
        # maximize throughput, minimize power
        frontier = compute(
            [a, b],
            [("throughput", "maximize"), ("power", "minimize")],
        )
        # Both are non-dominated: a has more throughput, b uses less power
        assert len(frontier) == 2

    def test_identical_results(self) -> None:
        """Identical results are not dominated by each other."""
        a = _make_result({"cost": 5.0, "latency": 1.0}, "a")
        b = _make_result({"cost": 5.0, "latency": 1.0}, "b")
        frontier = compute([a, b], [("cost", "minimize"), ("latency", "minimize")])
        assert len(frontier) == 2

    def test_empty_results(self) -> None:
        frontier = compute([], [("cost", "minimize")])
        assert frontier == []

    def test_empty_objectives(self) -> None:
        results = [_make_result({"cost": 5.0})]
        frontier = compute(results, [])
        assert len(frontier) == 1

    def test_sorted_by_first_objective(self) -> None:
        a = _make_result({"cost": 10.0, "latency": 0.5}, "a")
        b = _make_result({"cost": 3.0, "latency": 3.0}, "b")
        c = _make_result({"cost": 7.0, "latency": 1.0}, "c")
        frontier = compute(
            [a, b, c],
            [("cost", "minimize"), ("latency", "minimize")],
        )
        costs = [r.metrics["cost"] for r in frontier]
        assert costs == sorted(costs)


class TestPareto2D:
    def test_basic(self) -> None:
        a = _make_result({"x": 1.0, "y": 5.0}, "a")
        b = _make_result({"x": 5.0, "y": 1.0}, "b")
        c = _make_result({"x": 3.0, "y": 3.0}, "c")
        frontier = compute_2d([a, b, c], "x", "y")
        assert len(frontier) == 3  # All non-dominated in 2D

    def test_with_dominated(self) -> None:
        a = _make_result({"x": 1.0, "y": 1.0}, "a")
        b = _make_result({"x": 2.0, "y": 2.0}, "b")
        frontier = compute_2d([a, b], "x", "y")
        assert len(frontier) == 1
        assert frontier[0].model_spec.model_id == "a"
