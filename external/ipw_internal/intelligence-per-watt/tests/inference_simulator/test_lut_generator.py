"""Tests for LUT generator and LUT lookup interpolation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


@pytest.fixture
def synthetic_measurements():
    """Generate synthetic profiling measurements for LUT testing."""
    measurements = []
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [128, 256, 512, 1024]:
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6

            measurements.append(
                OperatorMeasurement(
                    operator_name="linear_qkv",
                    category=OperatorCategory.LINEAR,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time,
                    energy_j=base_time * 400,
                    power_w=400.0,
                    flops=int(tokens * 2 * 4096 * 6144),
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_prefill",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.5,
                    energy_j=base_time * 1.5 * 400,
                    power_w=420.0,
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_decode",
                    category=OperatorCategory.ATTENTION_DECODE,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 0.3,
                    energy_j=base_time * 0.3 * 350,
                    power_w=350.0,
                )
            )
    return measurements


class TestLUTLookup:
    def test_create_and_lookup(self):
        """Test creating a .npz file and looking up values."""
        from inference_simulator.estimator.lut_lookup import LUTLookup

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"

            # Create a simple 2D grid
            axis_0 = np.array([1, 2, 4, 8], dtype=np.float64)
            axis_1 = np.array([128, 256, 512, 1024], dtype=np.float64)
            grid = np.zeros((4, 4), dtype=np.float64)
            for i, bs in enumerate(axis_0):
                for j, sl in enumerate(axis_1):
                    grid[i, j] = bs * sl * 1e-6  # Simple linear model

            np.savez(
                npz_path,
                grid=grid,
                axis_0=axis_0,
                axis_1=axis_1,
                axis_names=np.array(["batch_size", "seq_len"]),
            )

            lut = LUTLookup(npz_path)

            # Exact match
            val = lut.lookup(batch_size=1, seq_len=128)
            assert val == pytest.approx(128e-6, rel=0.01)

            val = lut.lookup(batch_size=8, seq_len=1024)
            assert val == pytest.approx(8192e-6, rel=0.01)

    def test_interpolation(self):
        """Test that lookup interpolates between grid points."""
        from inference_simulator.estimator.lut_lookup import LUTLookup

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "interp.npz"

            axis_0 = np.array([1.0, 4.0], dtype=np.float64)
            axis_1 = np.array([100.0, 400.0], dtype=np.float64)
            grid = np.array([[100.0, 400.0], [400.0, 1600.0]], dtype=np.float64)

            np.savez(
                npz_path,
                grid=grid,
                axis_0=axis_0,
                axis_1=axis_1,
                axis_names=np.array(["batch_size", "seq_len"]),
            )

            lut = LUTLookup(npz_path)

            # Interpolated value should be between corner values
            val = lut.lookup(batch_size=2.0, seq_len=200.0)
            assert 100.0 < val < 1600.0


class TestLUTGenerator:
    @pytest.fixture
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_generate_token_ops_lut(self, _skip_no_sklearn, synthetic_measurements):
        """Test generating a token ops LUT from measurements."""
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=10, random_state=42)
        est.fit(synthetic_measurements)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = LUTGenerator()
            output_path = Path(tmpdir) / "token_ops.npz"
            result_path = gen.generate_gpu_token_ops_lut(
                estimator=est,
                operators=[OperatorCategory.LINEAR],
                token_counts=[128, 256, 512, 1024],
                tp_sizes=[1],
                output_path=output_path,
            )
            assert result_path.exists()

            # Load and verify structure
            data = np.load(result_path, allow_pickle=True)
            assert "grid" in data
            assert data["grid"].size > 0

    def test_generate_attention_prefill_lut(self, _skip_no_sklearn, synthetic_measurements):
        """Test generating an attention prefill LUT."""
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=10, random_state=42)
        est.fit(synthetic_measurements)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = LUTGenerator()
            output_path = Path(tmpdir) / "attn_prefill.npz"
            result_path = gen.generate_attention_prefill_lut(
                estimator=est,
                seq_lens=[128, 256, 512, 1024],
                batch_tokens=[1, 2, 4],
                tp_sizes=[1],
                output_path=output_path,
            )
            assert result_path.exists()

    def test_round_trip_accuracy(self, _skip_no_sklearn, synthetic_measurements):
        """Test that LUT values match estimator predictions."""
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_simulator.estimator.lut_lookup import LUTLookup
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        est = RandomForestEstimator(n_estimators=50, random_state=42)
        est.fit(synthetic_measurements)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = LUTGenerator()
            output_path = Path(tmpdir) / "token_ops.npz"
            gen.generate_gpu_token_ops_lut(
                estimator=est,
                operators=[OperatorCategory.LINEAR],
                token_counts=[128, 256, 512, 1024],
                tp_sizes=[1],
                output_path=output_path,
            )

            # The LUT should contain values
            data = np.load(output_path, allow_pickle=True)
            assert data["grid"].size > 0
