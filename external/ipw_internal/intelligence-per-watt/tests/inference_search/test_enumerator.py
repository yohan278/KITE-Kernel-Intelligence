"""Tests for inference_search.enumerator."""

from __future__ import annotations

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)

from inference_search.enumerator import enumerate_configurations
from inference_search.types import SearchConfig


def _make_model(
    model_id: str = "test/1b",
    num_layers: int = 24,
    hidden_dim: int = 2048,
    intermediate_dim: int = 5504,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_dim=intermediate_dim,
        vocab_size=32000,
    )


def _make_hw(name: str = "GPU-A", memory_gb: float = 80) -> HardwareSpec:
    return HardwareSpec(
        name=name,
        vendor="test",
        memory_gb=memory_gb,
        tdp_watts=400,
        peak_fp16_tflops=312.0,
        hbm_bandwidth_gb_s=2000.0,
    )


class TestEnumerator:
    def test_cartesian_product(self) -> None:
        """2 models x 2 hardware x 1 inference = 4 configs."""
        config = SearchConfig(
            model_specs=[_make_model("m1"), _make_model("m2")],
            hardware_specs=[_make_hw("A"), _make_hw("B")],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
        )
        results = enumerate_configurations(config)
        assert len(results) == 4

    def test_filters_oversized_model(self) -> None:
        """Model that doesn't fit in GPU memory should be filtered."""
        # Create a large model: ~70B params with fp16 => ~140 GB
        large_model = _make_model(
            model_id="big/70b",
            num_layers=80,
            hidden_dim=8192,
            intermediate_dim=28672,
        )
        small_gpu = _make_hw("Small GPU", memory_gb=24)

        config = SearchConfig(
            model_specs=[large_model],
            hardware_specs=[small_gpu],
            inference_specs=[InferenceSpec(num_gpus=1)],
            workload_spec=WorkloadSpec(),
        )
        results = enumerate_configurations(config)
        assert len(results) == 0

    def test_model_fits_with_multi_gpu(self) -> None:
        """Large model should fit with enough GPUs."""
        large_model = _make_model(
            model_id="big/70b",
            num_layers=80,
            hidden_dim=8192,
            intermediate_dim=28672,
        )
        gpu = _make_hw("GPU", memory_gb=80)

        config = SearchConfig(
            model_specs=[large_model],
            hardware_specs=[gpu],
            inference_specs=[InferenceSpec(num_gpus=4, tensor_parallel=4)],
            workload_spec=WorkloadSpec(),
        )
        results = enumerate_configurations(config)
        # With 4x80GB = 320GB, the ~140GB model should fit
        assert len(results) == 1

    def test_filters_invalid_tensor_parallel(self) -> None:
        """tensor_parallel > num_gpus should be filtered."""
        config = SearchConfig(
            model_specs=[_make_model()],
            hardware_specs=[_make_hw()],
            inference_specs=[InferenceSpec(num_gpus=1, tensor_parallel=4)],
            workload_spec=WorkloadSpec(),
        )
        results = enumerate_configurations(config)
        assert len(results) == 0

    def test_empty_inputs(self) -> None:
        """Empty spec lists produce empty results."""
        config = SearchConfig(
            model_specs=[],
            hardware_specs=[_make_hw()],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
        )
        assert enumerate_configurations(config) == []

    def test_mixed_feasibility(self) -> None:
        """Mix of feasible and infeasible configs."""
        small = _make_model("small/1b")
        large = _make_model("big/70b", num_layers=80, hidden_dim=8192, intermediate_dim=28672)
        gpu = _make_hw("GPU", memory_gb=80)

        config = SearchConfig(
            model_specs=[small, large],
            hardware_specs=[gpu],
            inference_specs=[InferenceSpec(num_gpus=1)],
            workload_spec=WorkloadSpec(),
        )
        results = enumerate_configurations(config)
        # small should fit, large should not
        assert len(results) == 1
        assert results[0][0].model_id == "small/1b"
