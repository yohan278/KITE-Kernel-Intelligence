import pytest
from evals.telemetry.trace_collector import TurnTrace, QueryTrace
from evals.telemetry.trace_to_profile import TraceToProfile


def _make_trace(query_id, num_turns, input_toks=100, output_toks=50, tools=None):
    turns = []
    for i in range(num_turns):
        turns.append(TurnTrace(
            turn_index=i,
            input_tokens=input_toks + i * 10,
            output_tokens=output_toks + i * 5,
            tools_called=tools or [],
            tool_result_tokens=20 if tools else 0,
            wall_clock_s=0.5,
        ))
    return QueryTrace(
        query_id=query_id,
        workload_type="chat",
        turns=turns,
        total_wall_clock_s=num_turns * 0.5,
        completed=True,
    )


class TestTraceToProfile:
    def test_empty_traces(self):
        converter = TraceToProfile()
        profile = converter.convert([])
        assert profile.n_samples == 0

    def test_single_turn_traces(self):
        traces = [_make_trace(f"q{i}", 1, 100, 50) for i in range(20)]
        converter = TraceToProfile()
        profile = converter.convert(traces, workload_type="chat")
        assert profile.n_samples == 20
        assert profile.workload_type == "chat"
        assert profile.turns_or_steps_dist is not None
        assert profile.input_tokens_dist is not None
        assert profile.answer_tokens_dist is not None
        assert profile.tool_call_probability == 0.0

    def test_multi_turn_traces(self):
        traces = [_make_trace(f"q{i}", 3, 100, 50) for i in range(20)]
        converter = TraceToProfile()
        profile = converter.convert(traces, workload_type="chat")
        assert profile.turns_or_steps_dist is not None
        assert profile.turns_or_steps_dist.mean == 3.0
        assert 0 in profile.input_tokens_by_position

    def test_tool_traces(self):
        traces = [_make_trace(f"q{i}", 2, 100, 50, tools=["search"]) for i in range(20)]
        converter = TraceToProfile()
        profile = converter.convert(traces, workload_type="agentic")
        assert profile.tool_call_probability == 1.0
        assert "search" in profile.tool_type_distribution
        assert profile.tool_call_tokens_dist is not None

    def test_profile_save_load(self, tmp_path):
        traces = [_make_trace(f"q{i}", 2, 100, 50) for i in range(20)]
        converter = TraceToProfile()
        profile = converter.convert(traces, workload_type="chat", source_dataset="test")

        path = tmp_path / "test_profile.json"
        profile.save(path)

        from inference_simulator.types.workload_profile import WorkloadProfile
        loaded = WorkloadProfile.load(path)
        assert loaded.workload_type == "chat"
        assert loaded.n_samples == 20
