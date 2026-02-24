"""Tests for evals.telemetry.trace_collector module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.telemetry.trace_collector import TurnTrace, QueryTrace, TraceCollector


# ---------------------------------------------------------------------------
# TurnTrace tests
# ---------------------------------------------------------------------------

class TestTurnTrace:
    def test_creation_defaults(self):
        t = TurnTrace(turn_index=0)
        assert t.turn_index == 0
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.tool_result_tokens == 0
        assert t.tools_called == []
        assert t.tool_latencies_s == {}
        assert t.wall_clock_s == 0.0
        assert t.error is None

    def test_creation_with_values(self):
        t = TurnTrace(
            turn_index=2,
            input_tokens=100,
            output_tokens=50,
            tool_result_tokens=20,
            tools_called=["search", "read"],
            tool_latencies_s={"search": 1.5, "read": 0.3},
            wall_clock_s=3.2,
            error="timeout",
        )
        assert t.turn_index == 2
        assert t.input_tokens == 100
        assert t.output_tokens == 50
        assert t.tool_result_tokens == 20
        assert t.tools_called == ["search", "read"]
        assert t.tool_latencies_s == {"search": 1.5, "read": 0.3}
        assert t.wall_clock_s == 3.2
        assert t.error == "timeout"

    def test_to_dict(self):
        t = TurnTrace(
            turn_index=1,
            input_tokens=10,
            output_tokens=20,
            tools_called=["tool_a"],
            tool_latencies_s={"tool_a": 0.5},
            wall_clock_s=1.0,
        )
        d = t.to_dict()
        assert d["turn_index"] == 1
        assert d["input_tokens"] == 10
        assert d["output_tokens"] == 20
        assert d["tool_result_tokens"] == 0
        assert d["tools_called"] == ["tool_a"]
        assert d["tool_latencies_s"] == {"tool_a": 0.5}
        assert d["wall_clock_s"] == 1.0
        assert d["error"] is None

    def test_from_dict(self):
        d = {
            "turn_index": 3,
            "input_tokens": 200,
            "output_tokens": 100,
            "tool_result_tokens": 50,
            "tools_called": ["bash"],
            "tool_latencies_s": {"bash": 2.0},
            "wall_clock_s": 5.0,
            "error": "some error",
        }
        t = TurnTrace.from_dict(d)
        assert t.turn_index == 3
        assert t.input_tokens == 200
        assert t.output_tokens == 100
        assert t.tool_result_tokens == 50
        assert t.tools_called == ["bash"]
        assert t.tool_latencies_s == {"bash": 2.0}
        assert t.wall_clock_s == 5.0
        assert t.error == "some error"

    def test_roundtrip(self):
        original = TurnTrace(
            turn_index=5,
            input_tokens=500,
            output_tokens=250,
            tool_result_tokens=75,
            tools_called=["search", "write"],
            tool_latencies_s={"search": 1.1, "write": 0.9},
            wall_clock_s=4.5,
            error=None,
        )
        restored = TurnTrace.from_dict(original.to_dict())
        assert restored.turn_index == original.turn_index
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.tool_result_tokens == original.tool_result_tokens
        assert restored.tools_called == original.tools_called
        assert restored.tool_latencies_s == original.tool_latencies_s
        assert restored.wall_clock_s == original.wall_clock_s
        assert restored.error == original.error

    def test_from_dict_missing_optional_fields(self):
        d = {"turn_index": 0}
        t = TurnTrace.from_dict(d)
        assert t.turn_index == 0
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.tools_called == []
        assert t.error is None


# ---------------------------------------------------------------------------
# QueryTrace tests
# ---------------------------------------------------------------------------

class TestQueryTrace:
    def _make_query_trace(self) -> QueryTrace:
        turns = [
            TurnTrace(
                turn_index=0,
                input_tokens=100,
                output_tokens=50,
                tools_called=["search"],
            ),
            TurnTrace(
                turn_index=1,
                input_tokens=200,
                output_tokens=80,
                tools_called=["read", "write"],
            ),
            TurnTrace(
                turn_index=2,
                input_tokens=150,
                output_tokens=60,
                tools_called=[],
            ),
        ]
        return QueryTrace(
            query_id="q-001",
            workload_type="chat",
            query_text="Hello world",
            response_text="Hi there",
            turns=turns,
            total_wall_clock_s=10.5,
            completed=True,
        )

    def test_creation(self):
        q = QueryTrace(query_id="q1", workload_type="reasoning")
        assert q.query_id == "q1"
        assert q.workload_type == "reasoning"
        assert q.query_text == ""
        assert q.response_text == ""
        assert q.turns == []
        assert q.total_wall_clock_s == 0.0
        assert q.completed is False

    def test_num_turns(self):
        q = self._make_query_trace()
        assert q.num_turns == 3

    def test_total_input_tokens(self):
        q = self._make_query_trace()
        assert q.total_input_tokens == 450  # 100 + 200 + 150

    def test_total_output_tokens(self):
        q = self._make_query_trace()
        assert q.total_output_tokens == 190  # 50 + 80 + 60

    def test_tool_call_count(self):
        q = self._make_query_trace()
        assert q.tool_call_count == 3  # 1 + 2 + 0

    def test_properties_empty_turns(self):
        q = QueryTrace(query_id="q2", workload_type="agentic")
        assert q.num_turns == 0
        assert q.total_input_tokens == 0
        assert q.total_output_tokens == 0
        assert q.tool_call_count == 0

    def test_to_dict(self):
        q = self._make_query_trace()
        d = q.to_dict()
        assert d["query_id"] == "q-001"
        assert d["workload_type"] == "chat"
        assert d["query_text"] == "Hello world"
        assert d["response_text"] == "Hi there"
        assert len(d["turns"]) == 3
        assert d["total_wall_clock_s"] == 10.5
        assert d["completed"] is True

    def test_from_dict(self):
        d = {
            "query_id": "q-002",
            "workload_type": "agentic",
            "query_text": "Fix the bug",
            "response_text": "Done",
            "turns": [
                {"turn_index": 0, "input_tokens": 50, "output_tokens": 30},
            ],
            "total_wall_clock_s": 5.0,
            "completed": True,
        }
        q = QueryTrace.from_dict(d)
        assert q.query_id == "q-002"
        assert q.workload_type == "agentic"
        assert q.num_turns == 1
        assert q.turns[0].input_tokens == 50
        assert q.completed is True

    def test_roundtrip(self):
        original = self._make_query_trace()
        restored = QueryTrace.from_dict(original.to_dict())
        assert restored.query_id == original.query_id
        assert restored.workload_type == original.workload_type
        assert restored.query_text == original.query_text
        assert restored.response_text == original.response_text
        assert restored.num_turns == original.num_turns
        assert restored.total_input_tokens == original.total_input_tokens
        assert restored.total_output_tokens == original.total_output_tokens
        assert restored.tool_call_count == original.tool_call_count
        assert restored.total_wall_clock_s == original.total_wall_clock_s
        assert restored.completed == original.completed

    def test_save_jsonl(self, tmp_path: Path):
        q = self._make_query_trace()
        outfile = tmp_path / "traces.jsonl"
        q.save_jsonl(outfile)

        assert outfile.exists()
        lines = outfile.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["query_id"] == "q-001"

    def test_save_jsonl_appends(self, tmp_path: Path):
        outfile = tmp_path / "traces.jsonl"
        q1 = QueryTrace(query_id="q1", workload_type="chat", completed=True)
        q2 = QueryTrace(query_id="q2", workload_type="reasoning", completed=False)
        q1.save_jsonl(outfile)
        q2.save_jsonl(outfile)

        lines = outfile.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["query_id"] == "q1"
        assert json.loads(lines[1])["query_id"] == "q2"

    def test_load_jsonl(self, tmp_path: Path):
        outfile = tmp_path / "traces.jsonl"
        q1 = self._make_query_trace()
        q2 = QueryTrace(
            query_id="q-002",
            workload_type="reasoning",
            completed=True,
            total_wall_clock_s=3.0,
        )
        q1.save_jsonl(outfile)
        q2.save_jsonl(outfile)

        loaded = QueryTrace.load_jsonl(outfile)
        assert len(loaded) == 2
        assert loaded[0].query_id == "q-001"
        assert loaded[0].num_turns == 3
        assert loaded[1].query_id == "q-002"
        assert loaded[1].num_turns == 0

    def test_save_load_roundtrip(self, tmp_path: Path):
        outfile = tmp_path / "sub" / "dir" / "traces.jsonl"
        original = self._make_query_trace()
        original.save_jsonl(outfile)
        loaded = QueryTrace.load_jsonl(outfile)
        assert len(loaded) == 1
        restored = loaded[0]
        assert restored.query_id == original.query_id
        assert restored.total_input_tokens == original.total_input_tokens
        assert restored.total_output_tokens == original.total_output_tokens
        assert restored.tool_call_count == original.tool_call_count

    def test_save_jsonl_creates_parent_dirs(self, tmp_path: Path):
        outfile = tmp_path / "a" / "b" / "c" / "traces.jsonl"
        q = QueryTrace(query_id="q1", workload_type="chat")
        q.save_jsonl(outfile)
        assert outfile.exists()


# ---------------------------------------------------------------------------
# TraceCollector tests
# ---------------------------------------------------------------------------

def _mock_vllm_response(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    content: str = "Hello from vLLM",
) -> dict:
    """Create a mock vLLM chat completions response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestTraceCollectorDirectVLLM:
    def test_successful_query(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_vllm_response(
            prompt_tokens=120,
            completion_tokens=60,
            content="Test response",
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_direct_vllm(
                query_id="q-test",
                workload_type="chat",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert trace.query_id == "q-test"
        assert trace.workload_type == "chat"
        assert trace.completed is True
        assert trace.num_turns == 1
        assert trace.turns[0].input_tokens == 120
        assert trace.turns[0].output_tokens == 60
        assert trace.response_text == "Test response"
        assert trace.query_text == "Hi"
        assert trace.turns[0].error is None

        # Verify the correct URL was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:8000/v1/chat/completions"

    def test_query_with_custom_params(self):
        collector = TraceCollector(
            vllm_url="http://myhost:9000/",
            model_name="big-model",
        )
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_vllm_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_direct_vllm(
                query_id="q2",
                workload_type="reasoning",
                messages=[{"role": "user", "content": "Think hard"}],
                max_tokens=2048,
                temperature=0.1,
                top_p=0.95,
            )

        assert trace.completed is True
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "big-model"
        assert payload["max_tokens"] == 2048
        assert payload["temperature"] == 0.1
        assert payload["top_p"] == 0.95
        # URL should have trailing slash stripped
        assert call_args[0][0] == "http://myhost:9000/v1/chat/completions"

    def test_error_handling(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("Connection refused")

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_direct_vllm(
                query_id="q-fail",
                workload_type="chat",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert trace.query_id == "q-fail"
        assert trace.completed is False
        assert trace.num_turns == 1
        assert trace.turns[0].error is not None
        assert "Connection refused" in trace.turns[0].error
        assert trace.query_text == "Hello"
        assert trace.total_wall_clock_s > 0


class TestTraceCollectorMultiTurnVLLM:
    def test_multi_turn_conversation(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )

        responses = [
            _mock_vllm_response(prompt_tokens=50, completion_tokens=30, content="Reply 1"),
            _mock_vllm_response(prompt_tokens=100, completion_tokens=40, content="Reply 2"),
        ]
        call_count = 0

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def make_response(*args, **kwargs):
            nonlocal call_count
            resp = MagicMock()
            resp.json.return_value = responses[call_count]
            resp.raise_for_status = MagicMock()
            call_count += 1
            return resp

        mock_client.post.side_effect = make_response

        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_multi_turn_vllm(
                query_id="q-multi",
                workload_type="chat",
                conversation=conversation,
            )

        assert trace.query_id == "q-multi"
        assert trace.completed is True
        assert trace.num_turns == 2
        assert trace.turns[0].turn_index == 0
        assert trace.turns[0].input_tokens == 50
        assert trace.turns[0].output_tokens == 30
        assert trace.turns[1].turn_index == 1
        assert trace.turns[1].input_tokens == 100
        assert trace.turns[1].output_tokens == 40
        assert trace.response_text == "Reply 2"
        assert trace.query_text == "Hello"
        assert trace.total_wall_clock_s > 0

    def test_multi_turn_error_mid_conversation(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )

        call_count = 0
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def make_response(*args, **kwargs):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                resp = MagicMock()
                resp.json.return_value = _mock_vllm_response(
                    prompt_tokens=50, completion_tokens=30, content="OK"
                )
                resp.raise_for_status = MagicMock()
                return resp
            else:
                raise Exception("Server error")

        mock_client.post.side_effect = make_response

        conversation = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
        ]

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_multi_turn_vllm(
                query_id="q-err",
                workload_type="chat",
                conversation=conversation,
            )

        assert trace.completed is False
        assert trace.num_turns == 2
        assert trace.turns[0].error is None
        assert trace.turns[1].error is not None
        assert "Server error" in trace.turns[1].error

    def test_single_user_turn(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = _mock_vllm_response(
            prompt_tokens=80, completion_tokens=40, content="Single reply"
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        conversation = [
            {"role": "user", "content": "Just one turn"},
        ]

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_multi_turn_vllm(
                query_id="q-single",
                workload_type="chat",
                conversation=conversation,
            )

        assert trace.completed is True
        assert trace.num_turns == 1
        assert trace.turns[0].input_tokens == 80


class TestTraceCollectorStubs:
    def test_run_query_react_delegates_to_direct(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = _mock_vllm_response(content="React result")
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("evals.telemetry.trace_collector.httpx.Client", return_value=mock_client):
            trace = collector.run_query_react(
                query_id="q-react",
                workload_type="agentic",
                query="Find the answer",
                tools=[{"name": "search"}],
            )

        assert trace.query_id == "q-react"
        assert trace.workload_type == "agentic"
        assert trace.completed is True

    def test_run_query_openhands_returns_stub(self):
        collector = TraceCollector(
            vllm_url="http://localhost:8000",
            model_name="test-model",
        )
        trace = collector.run_query_openhands(
            query_id="q-oh",
            workload_type="coding",
            task_description="Fix the bug in main.py",
        )
        assert trace.query_id == "q-oh"
        assert trace.workload_type == "coding"
        assert trace.query_text == "Fix the bug in main.py"
        assert trace.completed is False


class TestTraceCollectorInit:
    def test_default_init(self):
        c = TraceCollector()
        assert c.vllm_url == "http://localhost:8000"
        assert c.model_name == ""

    def test_custom_init(self):
        c = TraceCollector(vllm_url="http://gpu-server:9000", model_name="llama-70b")
        assert c.vllm_url == "http://gpu-server:9000"
        assert c.model_name == "llama-70b"

    def test_trailing_slash_stripped(self):
        c = TraceCollector(vllm_url="http://localhost:8000/")
        assert c.vllm_url == "http://localhost:8000"
