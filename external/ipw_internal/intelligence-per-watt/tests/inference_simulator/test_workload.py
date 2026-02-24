"""Tests for WorkloadGenerator Poisson arrivals and token distributions."""

import numpy as np
import pytest

from inference_simulator.request.request import RequestState
from inference_simulator.types.workload_spec import WorkloadSpec
from inference_simulator.workload.generator import WorkloadGenerator


class TestWorkloadGenerator:
    def test_generate_basic(self):
        spec = WorkloadSpec(
            qps=10.0,
            avg_input_tokens=500,
            avg_output_tokens=200,
        )
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=1.0, seed=42)

        # With QPS=10 and 1s, expect roughly 10 requests
        assert 5 <= len(requests) <= 20

        # All requests should be in WAITING state
        for r in requests:
            assert r.state == RequestState.WAITING

    def test_sorted_by_arrival(self):
        spec = WorkloadSpec(qps=100.0)
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=1.0, seed=42)

        for i in range(1, len(requests)):
            assert requests[i].arrival_time_ns >= requests[i - 1].arrival_time_ns

    def test_arrivals_within_duration(self):
        spec = WorkloadSpec(qps=50.0)
        gen = WorkloadGenerator()
        duration_s = 2.0
        requests = gen.generate(spec, duration_s=duration_s, seed=42)

        for r in requests:
            assert 0 <= r.arrival_time_ns < int(duration_s * 1e9)

    def test_token_distributions(self):
        spec = WorkloadSpec(
            qps=100.0,
            avg_input_tokens=500,
            avg_output_tokens=200,
            input_token_std=100.0,
            output_token_std=50.0,
        )
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=10.0, seed=42)

        input_tokens = [r.input_tokens for r in requests]
        output_tokens = [r.max_output_tokens for r in requests]

        # Mean should be close to specified
        assert abs(np.mean(input_tokens) - 500) < 50
        assert abs(np.mean(output_tokens) - 200) < 30

        # All should be positive
        assert all(t >= 1 for t in input_tokens)
        assert all(t >= 1 for t in output_tokens)

    def test_reproducibility(self):
        spec = WorkloadSpec(qps=10.0)
        gen = WorkloadGenerator()

        requests1 = gen.generate(spec, duration_s=1.0, seed=42)
        requests2 = gen.generate(spec, duration_s=1.0, seed=42)

        assert len(requests1) == len(requests2)
        for r1, r2 in zip(requests1, requests2):
            assert r1.arrival_time_ns == r2.arrival_time_ns
            assert r1.input_tokens == r2.input_tokens
            assert r1.max_output_tokens == r2.max_output_tokens

    def test_different_seeds_different_results(self):
        spec = WorkloadSpec(qps=10.0)
        gen = WorkloadGenerator()

        requests1 = gen.generate(spec, duration_s=1.0, seed=42)
        requests2 = gen.generate(spec, duration_s=1.0, seed=99)

        # Very unlikely to be identical
        if len(requests1) == len(requests2):
            tokens1 = [r.input_tokens for r in requests1]
            tokens2 = [r.input_tokens for r in requests2]
            assert tokens1 != tokens2

    def test_zero_qps(self):
        spec = WorkloadSpec(qps=0.0)
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=1.0, seed=42)
        assert len(requests) == 0

    def test_zero_duration(self):
        spec = WorkloadSpec(qps=10.0)
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=0.0, seed=42)
        assert len(requests) == 0

    def test_max_seq_len_clamping(self):
        spec = WorkloadSpec(
            qps=10.0,
            avg_input_tokens=1000,
            input_token_std=500.0,
        )
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=1.0, seed=42, max_seq_len=512)

        for r in requests:
            assert r.input_tokens <= 512

    def test_unique_request_ids(self):
        spec = WorkloadSpec(qps=100.0)
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=1.0, seed=42)

        ids = [r.request_id for r in requests]
        assert len(ids) == len(set(ids))

    def test_poisson_inter_arrivals(self):
        """Verify inter-arrival times are roughly exponentially distributed."""
        spec = WorkloadSpec(qps=100.0)
        gen = WorkloadGenerator()
        requests = gen.generate(spec, duration_s=10.0, seed=42)

        if len(requests) < 10:
            pytest.skip("Too few requests for distribution test")

        inter_arrivals_ns = []
        for i in range(1, len(requests)):
            inter_arrivals_ns.append(
                requests[i].arrival_time_ns - requests[i - 1].arrival_time_ns
            )

        inter_arrivals_s = np.array(inter_arrivals_ns) / 1e9
        # Mean inter-arrival should be close to 1/QPS = 0.01s
        mean_ia = np.mean(inter_arrivals_s)
        assert abs(mean_ia - 0.01) < 0.005

    def test_burstiness_preserves_mean_rate(self):
        """Bursty arrivals should have same mean rate as Poisson."""
        gen = WorkloadGenerator()
        for burstiness in [0.5, 1.0, 2.0]:
            spec = WorkloadSpec(qps=100.0, burstiness=burstiness)
            requests = gen.generate(spec, duration_s=10.0, seed=42)
            mean_rate = len(requests) / 10.0
            assert abs(mean_rate - 100.0) < 20.0  # within 20%

    def test_burstiness_default_matches_poisson(self):
        """Default burstiness=1.0 should produce identical results to current behavior."""
        gen = WorkloadGenerator()
        spec = WorkloadSpec(qps=100.0, burstiness=1.0)
        requests = gen.generate(spec, duration_s=10.0, seed=42)
        inter_arrivals_s = np.diff([r.arrival_time_ns for r in requests]) / 1e9
        mean_ia = np.mean(inter_arrivals_s)
        assert abs(mean_ia - 0.01) < 0.005
