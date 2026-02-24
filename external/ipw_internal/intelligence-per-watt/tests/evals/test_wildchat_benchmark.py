"""Tests for evals.benchmarks.wildchat module."""
from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from evals.benchmarks.wildchat.dataset import WildChatSample, load_wildchat_samples
from evals.benchmarks.wildchat.scorer import score_completion
from evals.benchmarks.wildchat.main import WildChatBenchmark


# ---------------------------------------------------------------------------
# WildChatSample tests
# ---------------------------------------------------------------------------


class TestWildChatSample:
    def test_num_turns_single_user(self):
        sample = WildChatSample(
            original_index=0,
            conversation=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        )
        assert sample.num_turns == 1

    def test_num_turns_multi(self):
        sample = WildChatSample(
            original_index=0,
            conversation=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good"},
                {"role": "user", "content": "Great"},
            ],
        )
        assert sample.num_turns == 3

    def test_num_turns_empty(self):
        sample = WildChatSample(original_index=0, conversation=[])
        assert sample.num_turns == 0

    def test_first_user_message(self):
        sample = WildChatSample(
            original_index=0,
            conversation=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        )
        assert sample.first_user_message == "What is 2+2?"

    def test_first_user_message_empty_conversation(self):
        sample = WildChatSample(original_index=0, conversation=[])
        assert sample.first_user_message == ""

    def test_first_user_message_no_user_turns(self):
        sample = WildChatSample(
            original_index=0,
            conversation=[
                {"role": "system", "content": "system msg"},
                {"role": "assistant", "content": "assistant msg"},
            ],
        )
        assert sample.first_user_message == ""

    def test_metadata_defaults(self):
        sample = WildChatSample(original_index=5, conversation=[])
        assert sample.model == ""
        assert sample.language == "English"
        assert sample.metadata == {}


# ---------------------------------------------------------------------------
# load_wildchat_samples tests
# ---------------------------------------------------------------------------


class TestLoadWildchatSamples:
    def _make_mock_conversation_sample(self, turns, model="gpt-4", language="English"):
        """Create a mock ConversationSample."""
        mock_turns = []
        for role, content in turns:
            t = MagicMock()
            t.role = role
            t.content = content
            mock_turns.append(t)

        conv = MagicMock()
        conv.turns = mock_turns
        conv.model = model
        conv.language = language
        conv.metadata = {"source": "wildchat"}
        return conv

    @patch("dataset_generator.datasets.wildchat.WildChatLoader")
    def test_basic_load(self, mock_loader_cls):
        convs = [
            self._make_mock_conversation_sample([
                ("user", "Hello"),
                ("assistant", "Hi"),
            ]),
            self._make_mock_conversation_sample([
                ("user", "Question"),
                ("assistant", "Answer"),
                ("user", "Follow up"),
                ("assistant", "More"),
            ]),
        ]
        mock_loader_cls.return_value.load_conversations.return_value = convs

        samples = list(load_wildchat_samples())
        assert len(samples) == 2
        assert samples[0].original_index == 0
        assert samples[0].num_turns == 1
        assert samples[1].num_turns == 2

    @patch("dataset_generator.datasets.wildchat.WildChatLoader")
    def test_min_turns_filter(self, mock_loader_cls):
        convs = [
            self._make_mock_conversation_sample([
                ("user", "Hello"),
                ("assistant", "Hi"),
            ]),
            self._make_mock_conversation_sample([
                ("user", "Q1"),
                ("assistant", "A1"),
                ("user", "Q2"),
                ("assistant", "A2"),
            ]),
        ]
        mock_loader_cls.return_value.load_conversations.return_value = convs

        samples = list(load_wildchat_samples(min_turns=2))
        assert len(samples) == 1
        assert samples[0].num_turns == 2

    @patch("dataset_generator.datasets.wildchat.WildChatLoader")
    def test_max_turns_filter(self, mock_loader_cls):
        convs = [
            self._make_mock_conversation_sample([
                ("user", "Q1"),
                ("assistant", "A1"),
            ]),
            self._make_mock_conversation_sample([
                ("user", "Q1"),
                ("assistant", "A1"),
                ("user", "Q2"),
                ("assistant", "A2"),
                ("user", "Q3"),
            ]),
        ]
        mock_loader_cls.return_value.load_conversations.return_value = convs

        samples = list(load_wildchat_samples(max_turns=2))
        assert len(samples) == 1
        assert samples[0].num_turns == 1


# ---------------------------------------------------------------------------
# score_completion tests
# ---------------------------------------------------------------------------


class TestScoreCompletion:
    def test_empty_results(self):
        metrics = score_completion({})
        assert metrics["completion_rate"] == 0.0
        assert metrics["total_samples"] == 0.0

    def test_all_completed(self):
        results = {
            "0": {"completed": True, "error": None, "num_turns_completed": 3},
            "1": {"completed": True, "error": None, "num_turns_completed": 2},
        }
        metrics = score_completion(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["completed_count"] == 2.0
        assert metrics["completion_rate"] == 100.0
        assert metrics["error_count"] == 0.0
        assert metrics["avg_turns"] == 2.5

    def test_partial_completion(self):
        results = {
            "0": {"completed": True, "error": None, "num_turns_completed": 3},
            "1": {"completed": False, "error": "timeout", "num_turns_completed": 1},
            "2": {"completed": True, "error": None, "num_turns_completed": 2},
            "3": {"completed": False, "error": "500", "num_turns_completed": 0},
        }
        metrics = score_completion(results)
        assert metrics["total_samples"] == 4.0
        assert metrics["completed_count"] == 2.0
        assert metrics["completion_rate"] == 50.0
        assert metrics["error_count"] == 2.0
        assert metrics["avg_turns"] == 1.5

    def test_none_completed(self):
        results = {
            "0": {"completed": False, "error": "err", "num_turns_completed": 0},
        }
        metrics = score_completion(results)
        assert metrics["completion_rate"] == 0.0
        assert metrics["error_count"] == 1.0


# ---------------------------------------------------------------------------
# WildChatBenchmark tests
# ---------------------------------------------------------------------------


class TestWildChatBenchmark:
    @patch("evals.benchmarks.wildchat.main.load_wildchat_samples")
    def test_generate_responses_with_trace_collector(self, mock_load):
        mock_load.return_value = iter([
            WildChatSample(
                original_index=0,
                conversation=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            ),
            WildChatSample(
                original_index=1,
                conversation=[
                    {"role": "user", "content": "Bye"},
                    {"role": "assistant", "content": "See ya"},
                ],
            ),
        ])

        benchmark = WildChatBenchmark(logger=logging.getLogger("test"))
        mock_orchestrator = MagicMock()

        # Mock TraceCollector to return a QueryTrace-like object
        mock_trace = MagicMock()
        mock_trace.num_turns = 1
        mock_trace.completed = True
        mock_trace.total_input_tokens = 100
        mock_trace.total_output_tokens = 50
        mock_trace.total_wall_clock_s = 1.5
        mock_trace.to_dict.return_value = {"query_id": "0"}

        mock_collector_cls = MagicMock(return_value=MagicMock(
            run_query_multi_turn_vllm=MagicMock(return_value=mock_trace),
        ))

        # Inject a fake trace_collector module into sys.modules
        fake_mod = types.ModuleType("evals.telemetry.trace_collector")
        fake_mod.TraceCollector = mock_collector_cls
        with patch.dict(sys.modules, {"evals.telemetry.trace_collector": fake_mod}):
            results = benchmark.generate_responses(mock_orchestrator)

        assert len(results) == 2
        assert results["0"]["original_index"] == 0
        assert results["0"]["completed"] is True
        assert results["0"]["total_input_tokens"] == 100
        assert results["1"]["original_index"] == 1

    @patch("evals.benchmarks.wildchat.main.load_wildchat_samples")
    def test_generate_responses_handles_error(self, mock_load):
        mock_load.return_value = iter([
            WildChatSample(
                original_index=0,
                conversation=[
                    {"role": "user", "content": "Hello"},
                ],
            ),
        ])

        benchmark = WildChatBenchmark(logger=logging.getLogger("test"))
        mock_orchestrator = MagicMock()

        # Inject a fake trace_collector module that raises on instantiation
        mock_collector_cls = MagicMock(side_effect=Exception("connection error"))
        fake_mod = types.ModuleType("evals.telemetry.trace_collector")
        fake_mod.TraceCollector = mock_collector_cls
        with patch.dict(sys.modules, {"evals.telemetry.trace_collector": fake_mod}):
            results = benchmark.generate_responses(mock_orchestrator)

        assert len(results) == 1
        assert results["0"]["completed"] is False
        assert results["0"]["error"] == "connection error"

    def test_evaluate_responses(self):
        benchmark = WildChatBenchmark(logger=logging.getLogger("test"))
        results = {
            "0": {"completed": True, "error": None, "num_turns_completed": 3},
            "1": {"completed": False, "error": "timeout", "num_turns_completed": 1},
        }
        metrics = benchmark.evaluate_responses(results)
        assert metrics["completion_rate"] == 50.0
        assert metrics["total_samples"] == 2.0

    def test_benchmark_init_defaults(self):
        benchmark = WildChatBenchmark()
        assert benchmark.limit is None
        assert benchmark.min_turns == 1
        assert benchmark.max_turns is None
        assert benchmark.vllm_url == "http://localhost:8000"
        assert benchmark.model_name == ""
        assert benchmark.max_tokens == 8192
        assert benchmark.temperature == 0.7

    def test_benchmark_init_custom(self):
        benchmark = WildChatBenchmark(
            limit=50,
            min_turns=2,
            max_turns=5,
            vllm_url="http://gpu:9000",
            model_name="llama-70b",
            max_tokens=4096,
            temperature=0.5,
        )
        assert benchmark.limit == 50
        assert benchmark.min_turns == 2
        assert benchmark.max_turns == 5
        assert benchmark.vllm_url == "http://gpu:9000"
        assert benchmark.model_name == "llama-70b"
        assert benchmark.max_tokens == 4096
        assert benchmark.temperature == 0.5


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestWildChatRegistry:
    def test_register_benchmark(self):
        from evals.registry import get_benchmark

        factory = get_benchmark("wildchat")
        benchmark = factory(options={})
        assert isinstance(benchmark, WildChatBenchmark)

    def test_register_benchmark_with_options(self):
        from evals.registry import get_benchmark

        factory = get_benchmark("wildchat")
        benchmark = factory(options={"limit": 10, "min_turns": 2})
        assert isinstance(benchmark, WildChatBenchmark)
        assert benchmark.limit == 10
        assert benchmark.min_turns == 2
