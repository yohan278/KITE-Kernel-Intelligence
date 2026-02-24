"""Tests for MetricsCollector percentile computation."""

import pytest

from inference_simulator.metrics.collector import MetricsCollector, SimulationMetrics
from inference_simulator.request.request import Request, RequestState


def _make_completed_request(
    request_id: int,
    arrival_ns: int,
    first_token_ns: int,
    completion_ns: int,
    tokens_generated: int = 50,
    input_tokens: int = 100,
) -> Request:
    r = Request(
        request_id=request_id,
        arrival_time_ns=arrival_ns,
        input_tokens=input_tokens,
        max_output_tokens=tokens_generated,
        state=RequestState.COMPLETED,
        tokens_generated=tokens_generated,
        first_token_ns=first_token_ns,
        completion_ns=completion_ns,
    )
    return r


class TestSimulationMetrics:
    def test_defaults(self):
        m = SimulationMetrics()
        assert m.ttft_p50 == 0.0
        assert m.throughput_rps == 0.0
        assert m.total_requests == 0

    def test_frozen(self):
        m = SimulationMetrics()
        with pytest.raises(AttributeError):
            m.ttft_p50 = 1.0


class TestMetricsCollector:
    def test_empty(self):
        collector = MetricsCollector()
        metrics = collector.compute()
        assert metrics.total_requests == 0
        assert metrics.throughput_rps == 0.0

    def test_single_request(self):
        collector = MetricsCollector()
        req = _make_completed_request(
            request_id=0,
            arrival_ns=0,
            first_token_ns=100_000_000,  # 100ms
            completion_ns=1_000_000_000,  # 1s
            tokens_generated=50,
        )
        collector.record_request(req)
        collector.set_total_time(1.0)

        metrics = collector.compute()
        assert metrics.total_requests == 1
        assert metrics.total_tokens_generated == 50
        assert metrics.throughput_rps == pytest.approx(1.0)
        assert metrics.throughput_tps == pytest.approx(50.0)

        # TTFT should be 100ms
        assert metrics.ttft_p50 == pytest.approx(0.1)

        # E2E should be 1s
        assert metrics.e2e_p50 == pytest.approx(1.0)

    def test_multiple_requests_percentiles(self):
        collector = MetricsCollector()

        # Create 100 requests with varying TTFT
        for i in range(100):
            ttft_ns = (i + 1) * 10_000_000  # 10ms to 1000ms
            e2e_ns = ttft_ns + 500_000_000  # +500ms decode
            req = _make_completed_request(
                request_id=i,
                arrival_ns=0,
                first_token_ns=ttft_ns,
                completion_ns=e2e_ns,
                tokens_generated=50,
            )
            collector.record_request(req)

        collector.set_total_time(10.0)
        metrics = collector.compute()

        assert metrics.total_requests == 100
        # P50 TTFT should be around 500ms (50th percentile of 10-1000ms)
        assert 0.4 < metrics.ttft_p50 < 0.6
        # P99 TTFT should be close to 1000ms
        assert metrics.ttft_p99 > 0.9

    def test_decode_step_tbt(self):
        collector = MetricsCollector()

        # Record decode steps with varying durations
        for i in range(100):
            collector.record_decode_step(10_000_000 + i * 100_000)  # ~10ms

        req = _make_completed_request(
            request_id=0,
            arrival_ns=0,
            first_token_ns=100_000_000,
            completion_ns=1_000_000_000,
        )
        collector.record_request(req)
        collector.set_total_time(1.0)

        metrics = collector.compute()
        # TBT should be around 10ms
        assert 0.009 < metrics.tbt_p50 < 0.015

    def test_energy_and_power(self):
        collector = MetricsCollector()
        req = _make_completed_request(
            request_id=0,
            arrival_ns=0,
            first_token_ns=100_000_000,
            completion_ns=1_000_000_000,
        )
        collector.record_request(req)
        collector.set_energy(300.0)  # 300 joules
        collector.set_total_time(1.0)  # 1 second

        metrics = collector.compute()
        assert metrics.total_energy_j == pytest.approx(300.0)
        assert metrics.avg_power_w == pytest.approx(300.0)

    def test_throughput_calculation(self):
        """Throughput uses the steady-state window (first arrival → last completion)."""
        collector = MetricsCollector()

        # 10 requests arriving every 500ms, each completing 500ms after arrival
        for i in range(10):
            arrival = i * 500_000_000  # 0, 0.5s, 1s, ...
            req = _make_completed_request(
                request_id=i,
                arrival_ns=arrival,
                first_token_ns=arrival + 100_000_000,
                completion_ns=arrival + 500_000_000,
                tokens_generated=100,
            )
            collector.record_request(req)

        collector.set_total_time(10.0)
        metrics = collector.compute()

        # Steady window = last_completion(4.5s) - first_arrival(0s) = 5.0s
        # (not total_time_s=10.0)
        assert metrics.throughput_rps == pytest.approx(2.0, rel=0.01)  # 10 reqs / 5s
        assert metrics.throughput_tps == pytest.approx(200.0, rel=0.01)  # 1000 tokens / 5s

    def test_warmup_exclusion(self):
        """Warmup requests should be excluded from latency percentiles."""
        collector = MetricsCollector(warmup_requests=2)

        # 2 warmup requests with artificially low TTFT (10ms)
        for i in range(2):
            collector.record_request(_make_completed_request(
                request_id=i, arrival_ns=0,
                first_token_ns=10_000_000, completion_ns=500_000_000,
            ))

        # 6 steady-state requests with higher TTFT (100ms)
        for i in range(2, 8):
            collector.record_request(_make_completed_request(
                request_id=i, arrival_ns=0,
                first_token_ns=100_000_000, completion_ns=1_000_000_000,
            ))

        # 2 drain requests with low TTFT again
        for i in range(8, 10):
            collector.record_request(_make_completed_request(
                request_id=i, arrival_ns=0,
                first_token_ns=10_000_000, completion_ns=500_000_000,
            ))

        collector.set_total_time(10.0)
        metrics = collector.compute()

        # Total should include all requests
        assert metrics.total_requests == 10

        # TTFT should reflect steady-state (100ms), not warmup (10ms)
        assert metrics.ttft_p50 == pytest.approx(0.1)

    def test_warmup_zero_default(self):
        """With warmup_requests=0 (default), all requests are included."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_request(_make_completed_request(
                request_id=i, arrival_ns=0,
                first_token_ns=100_000_000, completion_ns=1_000_000_000,
            ))

        collector.set_total_time(10.0)
        metrics = collector.compute()
        assert metrics.total_requests == 10
        assert metrics.ttft_p50 == pytest.approx(0.1)
