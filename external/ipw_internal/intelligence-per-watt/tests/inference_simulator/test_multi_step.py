"""Tests for multi-step execution types and request state transitions."""

from __future__ import annotations

import pytest

from inference_simulator.types.execution import LLMStep, MultiStepRequest, ToolCall


class TestToolCall:
    def test_create(self):
        tc = ToolCall(tool_type="web_search", tool_config="default")
        assert tc.tool_type == "web_search"
        assert tc.tool_config == "default"

    def test_default_config(self):
        tc = ToolCall(tool_type="calculator")
        assert tc.tool_config == "default"


class TestLLMStep:
    def test_create(self):
        step = LLMStep(
            input_tokens=1024,
            output_tokens=512,
            cumulative_context=0,
            tool_call=None,
        )
        assert step.input_tokens == 1024
        assert step.output_tokens == 512
        assert step.cumulative_context == 0
        assert step.tool_call is None

    def test_with_tool_call(self):
        tc = ToolCall(tool_type="code_interpreter", tool_config="sandbox")
        step = LLMStep(
            input_tokens=1024,
            output_tokens=512,
            cumulative_context=1024,
            tool_call=tc,
        )
        assert step.tool_call is not None
        assert step.tool_call.tool_type == "code_interpreter"

    def test_defaults(self):
        step = LLMStep(input_tokens=100, output_tokens=50)
        assert step.cumulative_context == 0
        assert step.tool_call is None


class TestMultiStepRequest:
    def test_create_empty(self):
        req = MultiStepRequest(
            request_id="test-1",
            arrival_time_ns=1000,
            workload_type="chat",
        )
        assert req.num_steps == 0
        assert req.is_complete
        assert req.current_llm_step is None

    def test_single_step(self):
        step = LLMStep(input_tokens=1024, output_tokens=512)
        req = MultiStepRequest(
            request_id="test-2",
            arrival_time_ns=0,
            workload_type="chat",
            steps=[step],
        )
        assert req.num_steps == 1
        assert not req.is_complete
        assert req.current_llm_step is step
        assert req.total_input_tokens == 1024
        assert req.total_output_tokens == 512

    def test_multi_step_agentic(self):
        steps = [
            LLMStep(
                input_tokens=1024,
                output_tokens=512,
                cumulative_context=0,
                tool_call=ToolCall("web_search"),
            ),
            LLMStep(
                input_tokens=512,
                output_tokens=512,
                cumulative_context=1536,
                tool_call=ToolCall("calculator"),
            ),
            LLMStep(
                input_tokens=512,
                output_tokens=1024,
                cumulative_context=2560,
                tool_call=None,
            ),
        ]
        req = MultiStepRequest(
            request_id="agent-1",
            arrival_time_ns=0,
            workload_type="agentic",
            steps=steps,
        )
        assert req.num_steps == 3
        assert not req.is_complete
        assert req.total_input_tokens == 2048
        assert req.total_output_tokens == 2048
        assert req.current_step == 0
        assert req.current_llm_step is steps[0]

    def test_step_progression(self):
        steps = [
            LLMStep(input_tokens=100, output_tokens=50, tool_call=ToolCall("calc")),
            LLMStep(input_tokens=50, output_tokens=50),
        ]
        req = MultiStepRequest(
            request_id="prog-1",
            arrival_time_ns=0,
            workload_type="agentic",
            steps=steps,
        )
        assert req.current_step == 0
        assert not req.is_complete

        req.current_step = 1
        assert req.current_llm_step is steps[1]
        assert not req.is_complete

        req.current_step = 2
        assert req.is_complete
        assert req.current_llm_step is None

    def test_timing_accumulators(self):
        req = MultiStepRequest(
            request_id="time-1",
            arrival_time_ns=0,
            workload_type="agentic",
            steps=[
                LLMStep(input_tokens=100, output_tokens=50, tool_call=ToolCall("calc")),
                LLMStep(input_tokens=50, output_tokens=50),
            ],
        )
        req.step_prefill_times_ns = [100_000, 200_000]
        req.step_decode_times_ns = [500_000, 300_000]
        req.step_tool_times_ns = [1_000_000]

        assert req.total_prefill_time_ns == 300_000
        assert req.total_decode_time_ns == 800_000
        assert req.total_tool_time_ns == 1_000_000


class TestRequestMultiStepIntegration:
    """Test Request integration with multi-step fields."""

    def test_request_with_steps(self):
        from inference_simulator.request.request import Request, RequestState

        step = LLMStep(input_tokens=1024, output_tokens=512)
        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=1024,
            max_output_tokens=512,
            steps=[step],
        )
        assert req.is_multi_step
        assert req.current_llm_step is step

    def test_request_without_steps(self):
        from inference_simulator.request.request import Request

        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=1024,
            max_output_tokens=512,
        )
        assert not req.is_multi_step

    def test_request_state_tool_executing(self):
        from inference_simulator.request.request import Request, RequestState

        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=1024,
            max_output_tokens=512,
        )
        req.state = RequestState.TOOL_EXECUTING
        assert req.state == RequestState.TOOL_EXECUTING

    def test_request_advance_step(self):
        from inference_simulator.request.request import Request

        steps = [
            LLMStep(input_tokens=100, output_tokens=50, tool_call=ToolCall("calc")),
            LLMStep(input_tokens=50, output_tokens=50),
        ]
        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=100,
            max_output_tokens=50,
            steps=steps,
        )
        assert req.current_step == 0
        has_more = req.advance_step()
        assert has_more
        assert req.current_step == 1
        has_more = req.advance_step()
        assert not has_more
        assert req.current_step == 2
