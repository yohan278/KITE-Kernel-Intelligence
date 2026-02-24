"""Tests for Markov tool latency model (container pools and DNS warm-up)."""

import numpy as np
import pytest

from inference_simulator.engine.tool_sampler import (
    ContainerPoolState,
    ToolLatencySampler,
)


class TestContainerPoolState:
    def test_first_call_is_cold_start(self):
        pool = ContainerPoolState()
        rng = np.random.default_rng(42)
        latency = pool.sample_latency(0.0, rng)
        # Cold start: mean ~2.0s
        assert latency >= 0.001
        assert pool.warm_containers == 1

    def test_second_call_within_cooldown_is_warm(self):
        pool = ContainerPoolState(cooldown_time_s=60.0)
        rng = np.random.default_rng(42)
        # First call: cold start
        pool.sample_latency(0.0, rng)
        assert pool.warm_containers == 1
        # Second call within cooldown: warm hit
        latency = pool.sample_latency(10.0, rng)
        assert latency >= 0.001

    def test_call_after_cooldown_is_cold(self):
        pool = ContainerPoolState(cooldown_time_s=60.0)
        rng = np.random.default_rng(42)
        pool.sample_latency(0.0, rng)
        # Call after cooldown expires
        latency = pool.sample_latency(100.0, rng)
        assert latency >= 0.001
        assert pool.warm_containers == 2  # Another warm container added

    def test_max_pool_size_caps_warm_containers(self):
        pool = ContainerPoolState(max_pool_size=2, cooldown_time_s=60.0)
        rng = np.random.default_rng(42)
        # Make 5 cold starts (each separated by > cooldown)
        for i in range(5):
            pool.sample_latency(i * 100.0, rng)
        assert pool.warm_containers == 2  # Capped at max_pool_size

    def test_warm_latency_lower_than_cold(self):
        """Statistically, warm latencies should be lower than cold."""
        rng = np.random.default_rng(42)
        cold_latencies = []
        warm_latencies = []
        for _ in range(100):
            pool = ContainerPoolState()
            cold_latencies.append(pool.sample_latency(0.0, rng))
            warm_latencies.append(pool.sample_latency(1.0, rng))

        assert np.mean(warm_latencies) < np.mean(cold_latencies)

    def test_minimum_latency_enforced(self):
        pool = ContainerPoolState(warm_mean_s=0.0, warm_std_s=0.0)
        rng = np.random.default_rng(42)
        pool.sample_latency(0.0, rng)  # cold start
        latency = pool.sample_latency(1.0, rng)  # warm
        assert latency >= 0.001


class TestToolLatencySampler:
    def test_container_tool_uses_markov_model(self):
        sampler = ToolLatencySampler()
        rng = np.random.default_rng(42)
        # First call: cold start
        lat1 = sampler.sample_latency("code_interpreter", "default", rng, 0.0)
        assert lat1 >= 0.001
        # Second call within cooldown: should create pool state
        lat2 = sampler.sample_latency("code_interpreter", "default", rng, 1.0)
        assert lat2 >= 0.001

    def test_docker_exec_uses_markov_model(self):
        sampler = ToolLatencySampler()
        rng = np.random.default_rng(42)
        lat = sampler.sample_latency("docker_exec", "default", rng, 0.0)
        assert lat >= 0.001

    def test_web_search_dns_warmup(self):
        sampler = ToolLatencySampler()
        rng = np.random.default_rng(42)
        # First call: cold DNS
        lat1 = sampler.sample_latency("web_search", "google", rng, 0.0)
        assert lat1 >= 0.001
        # Second call: warm DNS
        lat2 = sampler.sample_latency("web_search", "google", rng, 10.0)
        assert lat2 >= 0.001

    def test_simple_tool_iid(self):
        sampler = ToolLatencySampler()
        rng = np.random.default_rng(42)
        lat = sampler.sample_latency("calculator", "default", rng, 0.0)
        assert lat >= 0.001

    def test_different_configs_get_separate_pools(self):
        sampler = ToolLatencySampler()
        rng = np.random.default_rng(42)
        # Two different configs should have independent pool states
        sampler.sample_latency("code_interpreter", "python", rng, 0.0)
        sampler.sample_latency("code_interpreter", "node", rng, 0.0)
        assert len(sampler._container_pools) == 2

    def test_sample_result_tokens_default(self):
        sampler = ToolLatencySampler()
        rng = np.random.default_rng(42)
        tokens = sampler.sample_result_tokens("calculator", "default", rng)
        assert tokens == 512  # Default fallback

    def test_has_distribution_false_by_default(self):
        sampler = ToolLatencySampler()
        assert not sampler.has_distribution("calculator", "default")
