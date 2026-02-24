"""Tests for WorkloadGenerator.generate_from_profile() and profile_to_workload_spec()."""

import numpy as np
import pytest

from inference_simulator.request.request import Request, RequestState
from inference_simulator.types.execution import LLMStep, ToolCall
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile
from inference_simulator.types.workload_spec import WorkloadSpec, WorkloadType
from inference_simulator.workload.generator import WorkloadGenerator


def _make_normal_dist(mean: float, std: float, n: int = 1000) -> FittedDistribution:
    """Helper to create an empirical FittedDistribution that approximates a normal.

    Uses empirical samples to avoid scipy dependency issues in CI.
    """
    rng = np.random.default_rng(12345)
    samples = rng.normal(mean, std, size=n).tolist()
    return FittedDistribution(
        dist_name="empirical",
        n_samples=n,
        mean=mean,
        std=std,
        empirical_samples=samples,
    )


def _make_empirical_dist(samples: list) -> FittedDistribution:
    """Helper to create an empirical FittedDistribution."""
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    return FittedDistribution(
        dist_name="empirical",
        n_samples=len(samples),
        mean=mean,
        std=std,
        empirical_samples=samples,
    )


class TestGenerateFromProfileChat:
    """Test generate_from_profile with a chat-style profile."""

    def _make_chat_profile(self) -> WorkloadProfile:
        return WorkloadProfile(
            workload_type="chat",
            source_dataset="wildchat",
            n_samples=1000,
            turns_or_steps_dist=_make_empirical_dist([1.0, 1.0, 1.0, 2.0, 1.0]),
            input_tokens_dist=_make_normal_dist(500.0, 200.0),
            answer_tokens_dist=_make_normal_dist(300.0, 100.0),
        )

    def test_basic_generation(self):
        profile = self._make_chat_profile()
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)

        assert 5 <= len(requests) <= 20
        for r in requests:
            assert r.state == RequestState.WAITING
            assert r.workload_type == "chat"
            assert r.input_tokens >= 1
            assert r.max_output_tokens >= 1

    def test_requests_have_steps(self):
        profile = self._make_chat_profile()
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)

        for r in requests:
            assert len(r.steps) >= 1
            assert r.input_tokens == r.steps[0].input_tokens
            assert r.max_output_tokens == r.steps[0].output_tokens

    def test_sorted_by_arrival(self):
        profile = self._make_chat_profile()
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=50.0, duration_s=1.0, seed=42)

        for i in range(1, len(requests)):
            assert requests[i].arrival_time_ns >= requests[i - 1].arrival_time_ns

    def test_arrivals_within_duration(self):
        profile = self._make_chat_profile()
        gen = WorkloadGenerator()
        duration_s = 2.0
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=duration_s, seed=42)

        for r in requests:
            assert 0 <= r.arrival_time_ns < int(duration_s * 1e9)

    def test_reproducibility(self):
        profile = self._make_chat_profile()
        gen = WorkloadGenerator()

        r1 = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)
        r2 = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.arrival_time_ns == b.arrival_time_ns
            assert a.input_tokens == b.input_tokens
            assert a.max_output_tokens == b.max_output_tokens

    def test_unique_request_ids(self):
        profile = self._make_chat_profile()
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=50.0, duration_s=1.0, seed=42)

        ids = [r.request_id for r in requests]
        assert len(ids) == len(set(ids))


class TestGenerateFromProfileReasoning:
    """Test generate_from_profile with a reasoning-style profile (thinking tokens)."""

    def test_thinking_tokens_added(self):
        profile = WorkloadProfile(
            workload_type="reasoning",
            source_dataset="openthoughts",
            n_samples=500,
            input_tokens_dist=_make_normal_dist(800.0, 200.0),
            answer_tokens_dist=_make_normal_dist(200.0, 50.0),
            thinking_tokens_dist=_make_normal_dist(5000.0, 1000.0),
        )
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)

        assert len(requests) > 0
        # First step output should include thinking tokens (much larger than answer alone)
        output_tokens = [r.steps[0].output_tokens for r in requests]
        mean_output = np.mean(output_tokens)
        # With answer ~200 + thinking ~5000, mean should be well above 1000
        assert mean_output > 1000


class TestGenerateFromProfileAgentic:
    """Test generate_from_profile with an agentic-style profile (tool calls)."""

    def test_tool_calls_present(self):
        profile = WorkloadProfile(
            workload_type="agentic",
            source_dataset="agentdata",
            n_samples=200,
            turns_or_steps_dist=_make_empirical_dist([3.0, 4.0, 5.0, 3.0, 4.0,
                                                       3.0, 5.0, 4.0, 3.0, 4.0]),
            input_tokens_dist=_make_normal_dist(500.0, 200.0),
            answer_tokens_dist=_make_normal_dist(300.0, 100.0),
            tool_call_probability=0.8,
            tool_type_distribution={"web_search": 0.5, "code_interpreter": 0.3, "api_call": 0.2},
        )
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=2.0, seed=42)

        assert len(requests) > 0

        # At least some requests should have tool calls
        requests_with_tools = [
            r for r in requests
            if any(s.tool_call is not None for s in r.steps)
        ]
        assert len(requests_with_tools) > 0

        # Tool types should come from the distribution
        all_tool_types = set()
        for r in requests:
            for s in r.steps:
                if s.tool_call is not None:
                    all_tool_types.add(s.tool_call.tool_type)

        # With 80% probability and multiple steps, we should see tool calls
        assert len(all_tool_types) > 0
        for tt in all_tool_types:
            assert tt in {"web_search", "code_interpreter", "api_call"}

    def test_last_step_no_tool_call(self):
        """Tool calls should only appear on non-last steps."""
        profile = WorkloadProfile(
            workload_type="agentic",
            source_dataset="agentdata",
            n_samples=100,
            turns_or_steps_dist=_make_empirical_dist([3.0, 4.0, 5.0, 3.0, 4.0,
                                                       3.0, 5.0, 4.0, 3.0, 4.0]),
            input_tokens_dist=_make_normal_dist(500.0, 200.0),
            answer_tokens_dist=_make_normal_dist(300.0, 100.0),
            tool_call_probability=1.0,
            tool_type_distribution={"web_search": 1.0},
        )
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)

        for r in requests:
            if len(r.steps) > 1:
                # Last step must not have a tool call
                assert r.steps[-1].tool_call is None


class TestGenerateFromProfileEmpty:
    """Test edge cases that should return empty lists."""

    def test_zero_qps(self):
        profile = WorkloadProfile(
            workload_type="chat", source_dataset="test", n_samples=10,
        )
        gen = WorkloadGenerator()
        assert gen.generate_from_profile(profile, qps=0.0, duration_s=1.0, seed=42) == []

    def test_negative_qps(self):
        profile = WorkloadProfile(
            workload_type="chat", source_dataset="test", n_samples=10,
        )
        gen = WorkloadGenerator()
        assert gen.generate_from_profile(profile, qps=-1.0, duration_s=1.0, seed=42) == []

    def test_zero_duration(self):
        profile = WorkloadProfile(
            workload_type="chat", source_dataset="test", n_samples=10,
        )
        gen = WorkloadGenerator()
        assert gen.generate_from_profile(profile, qps=10.0, duration_s=0.0, seed=42) == []


class TestGenerateFromProfilePositionConditioned:
    """Test position-conditioned input/output distributions."""

    def test_position_specific_distributions(self):
        profile = WorkloadProfile(
            workload_type="chat",
            source_dataset="wildchat",
            n_samples=500,
            turns_or_steps_dist=_make_empirical_dist([3.0, 3.0, 3.0, 3.0, 3.0,
                                                       3.0, 3.0, 3.0, 3.0, 3.0]),
            input_tokens_dist=_make_normal_dist(500.0, 100.0),
            answer_tokens_dist=_make_normal_dist(300.0, 50.0),
            input_tokens_by_position={
                0: _make_normal_dist(1000.0, 50.0),  # First turn: long
                1: _make_normal_dist(100.0, 20.0),   # Second turn: short
                2: _make_normal_dist(200.0, 30.0),   # Third turn: medium
            },
            output_tokens_by_position={
                0: _make_normal_dist(800.0, 50.0),
                1: _make_normal_dist(150.0, 30.0),
                2: _make_normal_dist(250.0, 40.0),
            },
        )
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=20.0, duration_s=2.0, seed=42)

        assert len(requests) > 0

        # Collect per-position tokens
        pos0_input = [r.steps[0].input_tokens for r in requests if len(r.steps) >= 1]
        pos1_input = [r.steps[1].input_tokens for r in requests if len(r.steps) >= 2]

        # Position 0 should have larger input tokens (mean ~1000) vs position 1 (mean ~100)
        if pos0_input and pos1_input:
            assert np.mean(pos0_input) > np.mean(pos1_input)


class TestProfileToWorkloadSpec:
    """Test conversion of WorkloadProfile to WorkloadSpec."""

    def test_basic_conversion(self):
        profile = WorkloadProfile(
            workload_type="chat",
            source_dataset="wildchat",
            n_samples=1000,
            input_tokens_dist=_make_normal_dist(500.0, 200.0),
            answer_tokens_dist=_make_normal_dist(300.0, 100.0),
        )

        spec = WorkloadGenerator.profile_to_workload_spec(profile, qps=5.0)

        assert spec.qps == 5.0
        assert spec.avg_input_tokens == 500
        assert spec.avg_output_tokens == 300
        assert spec.input_token_std == 200.0
        assert spec.output_token_std == 100.0
        assert spec.workload_type == WorkloadType.CHAT

    def test_reasoning_type_mapping(self):
        profile = WorkloadProfile(
            workload_type="reasoning",
            source_dataset="openthoughts",
            n_samples=500,
            input_tokens_dist=_make_normal_dist(800.0, 300.0),
            answer_tokens_dist=_make_normal_dist(200.0, 50.0),
        )

        spec = WorkloadGenerator.profile_to_workload_spec(profile)
        assert spec.workload_type == WorkloadType.REASONING

    def test_missing_distributions_fallback(self):
        profile = WorkloadProfile(
            workload_type="chat",
            source_dataset="test",
            n_samples=10,
        )

        spec = WorkloadGenerator.profile_to_workload_spec(profile)
        assert spec.avg_input_tokens == 500
        assert spec.avg_output_tokens == 200
        assert spec.input_token_std == 200.0
        assert spec.output_token_std == 100.0

    def test_all_workload_types(self):
        for wt_name, wt_enum in [
            ("chat", WorkloadType.CHAT),
            ("reasoning", WorkloadType.REASONING),
            ("agentic", WorkloadType.AGENTIC),
            ("rag", WorkloadType.RAG),
            ("coding", WorkloadType.CODING),
        ]:
            profile = WorkloadProfile(
                workload_type=wt_name,
                source_dataset="test",
                n_samples=10,
                input_tokens_dist=_make_normal_dist(100.0, 10.0),
                answer_tokens_dist=_make_normal_dist(50.0, 5.0),
            )
            spec = WorkloadGenerator.profile_to_workload_spec(profile)
            assert spec.workload_type == wt_enum

    def test_unknown_workload_type(self):
        profile = WorkloadProfile(
            workload_type="unknown",
            source_dataset="test",
            n_samples=10,
            input_tokens_dist=_make_normal_dist(100.0, 10.0),
            answer_tokens_dist=_make_normal_dist(50.0, 5.0),
        )
        spec = WorkloadGenerator.profile_to_workload_spec(profile)
        assert spec.workload_type is None


class TestTokenCountsStatistical:
    """Statistical tests on generated token distributions."""

    def test_means_within_bounds(self):
        """With enough samples, means should be within 2 sigma of profile means."""
        profile = WorkloadProfile(
            workload_type="chat",
            source_dataset="wildchat",
            n_samples=1000,
            input_tokens_dist=_make_normal_dist(500.0, 100.0),
            answer_tokens_dist=_make_normal_dist(300.0, 80.0),
        )
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=100.0, duration_s=10.0, seed=42)

        assert len(requests) > 100

        input_tokens = [r.steps[0].input_tokens for r in requests]
        output_tokens = [r.steps[0].output_tokens for r in requests]

        n = len(input_tokens)
        input_mean = np.mean(input_tokens)
        output_mean = np.mean(output_tokens)

        # Mean should be within 2 sigma / sqrt(n) of target
        input_tolerance = 2 * 100.0 / np.sqrt(n) + 10  # small buffer for clamping
        output_tolerance = 2 * 80.0 / np.sqrt(n) + 10

        assert abs(input_mean - 500.0) < input_tolerance, (
            f"Input mean {input_mean:.1f} too far from 500.0 (tolerance {input_tolerance:.1f})"
        )
        assert abs(output_mean - 300.0) < output_tolerance, (
            f"Output mean {output_mean:.1f} too far from 300.0 (tolerance {output_tolerance:.1f})"
        )

    def test_cumulative_context_grows(self):
        """Cumulative context should increase with each step."""
        profile = WorkloadProfile(
            workload_type="agentic",
            source_dataset="agentdata",
            n_samples=100,
            turns_or_steps_dist=_make_empirical_dist([4.0, 4.0, 4.0, 4.0, 4.0,
                                                       4.0, 4.0, 4.0, 4.0, 4.0]),
            input_tokens_dist=_make_normal_dist(500.0, 100.0),
            answer_tokens_dist=_make_normal_dist(300.0, 50.0),
        )
        gen = WorkloadGenerator()
        requests = gen.generate_from_profile(profile, qps=10.0, duration_s=1.0, seed=42)

        for r in requests:
            if len(r.steps) > 1:
                for i in range(1, len(r.steps)):
                    assert r.steps[i].cumulative_context > r.steps[i - 1].cumulative_context

    def test_max_seq_len_clamping(self):
        """Token counts should respect max_seq_len."""
        profile = WorkloadProfile(
            workload_type="chat",
            source_dataset="test",
            n_samples=100,
            input_tokens_dist=_make_normal_dist(1000.0, 200.0),
            answer_tokens_dist=_make_normal_dist(800.0, 150.0),
        )
        gen = WorkloadGenerator()
        max_len = 512
        requests = gen.generate_from_profile(
            profile, qps=20.0, duration_s=1.0, seed=42, max_seq_len=max_len,
        )

        for r in requests:
            for s in r.steps:
                assert s.input_tokens <= max_len
                assert s.output_tokens <= max_len
