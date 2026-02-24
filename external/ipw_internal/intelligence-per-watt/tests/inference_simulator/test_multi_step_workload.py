"""Tests for multi-step workload generation."""

from __future__ import annotations

import pytest

from inference_simulator.types.workload_spec import WorkloadSpec, WorkloadType
from inference_simulator.workload.generator import WorkloadGenerator


class TestMultiStepWorkloadGeneration:
    def test_chat_workload(self):
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_chat(qps=10.0)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            # Chat: single step
            if req.steps:
                assert len(req.steps) == 1
                assert req.steps[0].tool_call is None

    def test_reasoning_workload(self):
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_reasoning(qps=5.0)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            # Reasoning: single step, large output
            if req.steps:
                assert len(req.steps) == 1
                assert req.steps[0].tool_call is None

    def test_agentic_workload(self):
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_agentic(qps=5.0)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            if req.steps:
                # Agentic: 2+ steps (1 base + 1-8 tool calls)
                assert len(req.steps) >= 2
                # All but last step should have tool_call
                for step in req.steps[:-1]:
                    assert step.tool_call is not None
                # Last step has no tool_call
                assert req.steps[-1].tool_call is None

    def test_rag_workload(self):
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_rag(qps=5.0)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            if req.steps:
                # RAG: 2 steps (query → retrieval → generate)
                assert len(req.steps) == 2
                assert req.steps[0].tool_call is not None
                assert req.steps[0].tool_call.tool_type == "faiss_retrieval"
                assert req.steps[1].tool_call is None

    def test_coding_workload(self):
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_coding(qps=5.0)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            if req.steps:
                assert len(req.steps) >= 2

    def test_default_workload_fallback(self):
        """WorkloadSpec without workload_type falls back to single-step."""
        gen = WorkloadGenerator()
        spec = WorkloadSpec(qps=10.0, avg_input_tokens=500, avg_output_tokens=200)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            if req.steps:
                assert len(req.steps) == 1

    def test_backward_compat_generate(self):
        """The original generate() method still works."""
        gen = WorkloadGenerator()
        spec = WorkloadSpec(qps=10.0, avg_input_tokens=500, avg_output_tokens=200)
        requests = gen.generate(spec, duration_s=1.0, seed=42)

        assert len(requests) > 0
        for req in requests:
            assert req.input_tokens > 0
            assert req.max_output_tokens > 0

    def test_arrival_times_ordered(self):
        """All requests should have non-decreasing arrival times."""
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_agentic(qps=20.0)
        requests = gen.generate_multi_step(spec, duration_s=2.0, seed=42)

        for i in range(1, len(requests)):
            assert requests[i].arrival_time_ns >= requests[i - 1].arrival_time_ns

    def test_cumulative_context_grows(self):
        """For agentic workloads, cumulative context should increase per step."""
        gen = WorkloadGenerator()
        spec = WorkloadSpec.for_agentic(qps=5.0)
        requests = gen.generate_multi_step(spec, duration_s=1.0, seed=42)

        for req in requests:
            if req.steps and len(req.steps) > 1:
                for i in range(1, len(req.steps)):
                    assert req.steps[i].cumulative_context >= req.steps[i - 1].cumulative_context
