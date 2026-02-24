"""Tests for ML-backed oracle integration with search."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)
from inference_search.types import SLAConstraint, SearchConfig


@pytest.fixture
def qwen3_8b_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen3-8B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=36,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=12288,
        vocab_size=151936,
    )


@pytest.fixture
def a100_spec() -> HardwareSpec:
    return HardwareSpec(
        name="A100-SXM4-80GB",
        vendor="nvidia",
        memory_gb=80,
        tdp_watts=400,
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        hbm_bandwidth_gb_s=2039.0,
        nvlink_bandwidth_gb_s=600.0,
        price_per_hour_usd=3.50,
    )


def _create_synthetic_lut_bundle(bundle_dir: Path) -> None:
    """Create minimal synthetic LUT .npz files."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Token ops LUT: [operators x token_counts x tp_sizes x 2]
    op_names = np.array(["linear"])
    token_counts = np.array([128, 256, 512, 1024], dtype=np.float64)
    tp_sizes = np.array([1], dtype=np.float64)
    grid = np.zeros((1, 4, 1, 2))
    for j, tc in enumerate(token_counts):
        grid[0, j, 0, 0] = tc * 1e-6  # time_s
        grid[0, j, 0, 1] = tc * 1e-6 * 400  # energy_j
    np.savez(
        bundle_dir / "gpu_token_ops.npz",
        grid=grid, axis_0=op_names, axis_1=token_counts, axis_2=tp_sizes,
        axis_names=np.array(["operator", "token_count", "tp_size"]),
    )

    # Attention prefill LUT: [seq_lens x batch_tokens x tp_sizes x 2]
    seq_lens = np.array([128, 256, 512, 1024], dtype=np.float64)
    batch_tokens = np.array([1, 4, 8], dtype=np.float64)
    grid = np.zeros((4, 3, 1, 2))
    for i, sl in enumerate(seq_lens):
        for j, bt in enumerate(batch_tokens):
            grid[i, j, 0, 0] = sl * bt * 1e-6
            grid[i, j, 0, 1] = sl * bt * 1e-6 * 400
    np.savez(
        bundle_dir / "gpu_attention_prefill.npz",
        grid=grid, axis_0=seq_lens, axis_1=batch_tokens, axis_2=tp_sizes,
        axis_names=np.array(["seq_len", "batch_tokens", "tp_size"]),
    )

    # Attention decode LUT: [kv_cache x batch_sizes x tp_sizes x 2]
    kv_sizes = np.array([128, 256, 512, 1024], dtype=np.float64)
    batch_sizes = np.array([1, 4, 8], dtype=np.float64)
    grid = np.zeros((4, 3, 1, 2))
    for i, kv in enumerate(kv_sizes):
        for j, bs in enumerate(batch_sizes):
            grid[i, j, 0, 0] = kv * 1e-7  # decode is faster
            grid[i, j, 0, 1] = kv * 1e-7 * 350
    np.savez(
        bundle_dir / "gpu_attention_decode.npz",
        grid=grid, axis_0=kv_sizes, axis_1=batch_sizes, axis_2=tp_sizes,
        axis_names=np.array(["kv_cache_size", "batch_size", "tp_size"]),
    )


class TestMLSearch:
    def test_run_search_with_roofline_oracle_default(self, qwen3_8b_spec, a100_spec):
        """run_search() still works with no oracle (backward compatible)."""
        from inference_search.cli import run_search

        config = SearchConfig(
            model_specs=[qwen3_8b_spec],
            hardware_specs=[a100_spec],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[SLAConstraint("ttft_s", 2.0, "max")],
            optimization_targets=["throughput_tps"],
        )
        result = run_search(config)
        assert len(result.all_results) >= 1
        assert result.pareto_frontier

    def test_run_search_accepts_oracle_param(self, qwen3_8b_spec, a100_spec):
        """run_search() accepts an explicit oracle parameter."""
        from inference_search.cli import run_search
        from inference_search.oracle import RooflineOracle

        oracle = RooflineOracle(accuracy_score=0.9, price_per_hour_usd=3.50)
        config = SearchConfig(
            model_specs=[qwen3_8b_spec],
            hardware_specs=[a100_spec],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            optimization_targets=["throughput_tps"],
        )
        result = run_search(config, oracle=oracle)
        assert len(result.all_results) >= 1

    def test_ml_oracle_with_synthetic_bundle(self, qwen3_8b_spec, a100_spec):
        """MLBackedOracle loads synthetic LUT bundle and returns metrics."""
        from inference_search.ml_oracle import MLBackedOracle

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "luts"
            _create_synthetic_lut_bundle(bundle_dir)

            oracle = MLBackedOracle(
                lut_bundle_dir=bundle_dir,
                accuracy_score=0.85,
                price_per_hour_usd=3.50,
            )
            metrics = oracle.simulate(
                model_spec=qwen3_8b_spec,
                hardware_spec=a100_spec,
                inference_spec=InferenceSpec(),
                workload_spec=WorkloadSpec(qps=1.0, avg_input_tokens=64, avg_output_tokens=16),
            )
            # Should have basic metrics
            assert "throughput_tps" in metrics or "throughput_rps" in metrics
            assert "ttft_s" in metrics or "ttft_p50" in metrics

    def test_ml_oracle_fallback_on_missing_bundle(self, qwen3_8b_spec, a100_spec):
        """MLBackedOracle falls back to roofline when bundle dir doesn't exist."""
        from inference_search.ml_oracle import MLBackedOracle

        oracle = MLBackedOracle(
            lut_bundle_dir=Path("/nonexistent/path"),
            accuracy_score=1.0,
        )
        # Should use roofline fallback
        metrics = oracle.simulate(
            model_spec=qwen3_8b_spec,
            hardware_spec=a100_spec,
            inference_spec=InferenceSpec(),
            workload_spec=WorkloadSpec(),
        )
        assert "throughput_tps" in metrics
        assert metrics["throughput_tps"] > 0

    def test_search_with_ml_oracle(self, qwen3_8b_spec, a100_spec):
        """Full search with ML-backed oracle from synthetic LUT bundle."""
        from inference_search.cli import run_search
        from inference_search.ml_oracle import MLBackedOracle

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "luts"
            _create_synthetic_lut_bundle(bundle_dir)

            oracle = MLBackedOracle(
                lut_bundle_dir=bundle_dir,
                accuracy_score=0.85,
                price_per_hour_usd=3.50,
            )
            config = SearchConfig(
                model_specs=[qwen3_8b_spec],
                hardware_specs=[a100_spec],
                inference_specs=[InferenceSpec()],
                workload_spec=WorkloadSpec(qps=1.0, avg_input_tokens=64, avg_output_tokens=16),
                sla_constraints=[SLAConstraint("ttft_s", 5.0, "max")],
                optimization_targets=["throughput_tps"],
            )
            result = run_search(config, oracle=oracle)
            assert len(result.all_results) >= 1
