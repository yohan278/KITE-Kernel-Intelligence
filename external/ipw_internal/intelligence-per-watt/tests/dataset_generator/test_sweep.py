"""Tests for dataset_generator sweep configuration."""

import pytest

from dataset_generator.profiler.sweep import SweepConfig


class TestSweepConfig:
    def test_defaults(self):
        config = SweepConfig()
        assert config.batch_sizes == [1, 2, 4, 8, 16, 32, 64]
        assert config.prefill_seq_lengths[0] == 128
        assert config.prefill_seq_lengths[-1] == 131072
        assert len(config.prefill_seq_lengths) == 11
        assert config.kv_cache_sizes[0] == 128
        assert config.kv_cache_sizes[-1] == 262144
        assert config.warmup_iterations == 5
        assert config.measurement_iterations == 20

    def test_custom_config(self):
        config = SweepConfig(
            batch_sizes=[1, 2],
            prefill_seq_lengths=[128, 256],
            warmup_iterations=3,
            measurement_iterations=10,
        )
        assert config.batch_sizes == [1, 2]
        assert config.prefill_seq_lengths == [128, 256]
        assert config.warmup_iterations == 3
        assert config.measurement_iterations == 10

    def test_get_sweep_points_single_dim(self):
        config = SweepConfig(batch_sizes=[1, 2, 4])
        points = list(config.get_sweep_points(["batch_sizes"]))
        assert len(points) == 3
        assert points[0] == {"batch_size": 1}
        assert points[1] == {"batch_size": 2}
        assert points[2] == {"batch_size": 4}

    def test_get_sweep_points_two_dims(self):
        config = SweepConfig(
            batch_sizes=[1, 2],
            prefill_seq_lengths=[128, 256],
        )
        points = list(config.get_sweep_points(["batch_sizes", "prefill_seq_lengths"]))
        assert len(points) == 4  # 2 x 2
        assert {"batch_size": 1, "seq_len": 128} in points
        assert {"batch_size": 2, "seq_len": 256} in points

    def test_get_sweep_points_empty_dims(self):
        config = SweepConfig()
        points = list(config.get_sweep_points([]))
        assert len(points) == 1
        assert points[0] == {}

    def test_get_sweep_points_invalid_dim(self):
        config = SweepConfig()
        with pytest.raises((ValueError, AttributeError)):
            list(config.get_sweep_points(["nonexistent_dim"]))
