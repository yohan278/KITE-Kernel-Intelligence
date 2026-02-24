"""Tests for evals.benchmarks.openthoughts module."""
from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from evals.benchmarks.openthoughts.dataset import (
    OpenThoughtsSample,
    load_openthoughts_samples,
    DEFAULT_INPUT_PROMPT,
)
from evals.benchmarks.openthoughts.scorer import (
    normalize_answer,
    extract_final_answer,
    score_exact_match,
    score_contains,
    score_openthoughts,
)
from evals.benchmarks.openthoughts.main import OpenThoughtsBenchmark


# ---------------------------------------------------------------------------
# OpenThoughtsSample tests
# ---------------------------------------------------------------------------


class TestOpenThoughtsSample:
    def test_get_prompt_default(self):
        sample = OpenThoughtsSample(
            original_index=0,
            problem="What is 2+2?",
            answer="4",
        )
        prompt = sample.get_prompt()
        assert "What is 2+2?" in prompt
        assert "Problem:" in prompt

    def test_get_prompt_custom(self):
        sample = OpenThoughtsSample(
            original_index=0,
            problem="Solve x^2=4",
            answer="x=2 or x=-2",
        )
        custom = "Please solve: {problem}"
        prompt = sample.get_prompt(custom)
        assert prompt == "Please solve: Solve x^2=4"

    def test_defaults(self):
        sample = OpenThoughtsSample(
            original_index=5,
            problem="test",
            answer="answer",
        )
        assert sample.domain == ""
        assert sample.reasoning == ""
        assert sample.metadata == {}


# ---------------------------------------------------------------------------
# load_openthoughts_samples tests
# ---------------------------------------------------------------------------


class TestLoadOpenthoughtsSamples:
    def _make_mock_sample(self, query, answer, domain="", reasoning=""):
        s = MagicMock()
        s.query = query
        s.expected_answer = answer
        s.metadata = {"domain": domain}
        if reasoning:
            s.metadata["deepseek_reasoning"] = reasoning
        return s

    @patch("dataset_generator.datasets.openthoughts.OpenThoughtsLoader")
    def test_basic_load(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = [
            self._make_mock_sample("What is 2+2?", "4", domain="math"),
            self._make_mock_sample("Capital of France?", "Paris", domain="geography"),
        ]

        samples = list(load_openthoughts_samples())
        assert len(samples) == 2
        assert samples[0].problem == "What is 2+2?"
        assert samples[0].answer == "4"
        assert samples[0].domain == "math"
        assert samples[1].problem == "Capital of France?"

    @patch("dataset_generator.datasets.openthoughts.OpenThoughtsLoader")
    def test_domain_filter(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = [
            self._make_mock_sample("Q1", "A1", domain="math"),
            self._make_mock_sample("Q2", "A2", domain="science"),
            self._make_mock_sample("Q3", "A3", domain="math"),
        ]

        samples = list(load_openthoughts_samples(domains=["math"]))
        assert len(samples) == 2
        assert all(s.domain == "math" for s in samples)

    @patch("dataset_generator.datasets.openthoughts.OpenThoughtsLoader")
    def test_reasoning_extraction(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = [
            self._make_mock_sample("Q1", "A1", reasoning="Step 1, Step 2"),
        ]

        samples = list(load_openthoughts_samples())
        assert len(samples) == 1
        assert samples[0].reasoning == "Step 1, Step 2"

    @patch("dataset_generator.datasets.openthoughts.OpenThoughtsLoader")
    def test_none_answer_becomes_empty(self, mock_loader_cls):
        s = MagicMock()
        s.query = "Q1"
        s.expected_answer = None
        s.metadata = {"domain": ""}
        mock_loader_cls.return_value.load.return_value = [s]

        samples = list(load_openthoughts_samples())
        assert len(samples) == 1
        assert samples[0].answer == ""


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_strips_whitespace(self):
        assert normalize_answer("  hello  ") == "hello"

    def test_lowercases(self):
        assert normalize_answer("HELLO World") == "hello world"

    def test_collapses_spaces(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_removes_special_chars(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_preserves_dots(self):
        assert normalize_answer("3.14") == "3.14"


class TestExtractFinalAnswer:
    def test_answer_is_pattern(self):
        response = "After calculation, the answer is 42."
        assert extract_final_answer(response) == "42"

    def test_final_answer_is_pattern(self):
        response = "Working through this... The final answer is Paris."
        assert extract_final_answer(response) == "Paris"

    def test_boxed_pattern(self):
        response = r"Therefore \boxed{42}"
        assert extract_final_answer(response) == "42"

    def test_bold_pattern_at_end(self):
        response = "The result is\n**42**"
        assert extract_final_answer(response) == "42"

    def test_therefore_pattern(self):
        response = "We see that x=5, therefore the answer is 5."
        # The "answer is" pattern matches first, extracting "5"
        assert extract_final_answer(response) == "5"

    def test_thus_pattern(self):
        response = "Combining both, thus 7."
        assert extract_final_answer(response) == "7"

    def test_fallback_last_line(self):
        response = "Step 1: compute\nStep 2: verify\n42"
        assert extract_final_answer(response) == "42"

    def test_empty_response(self):
        assert extract_final_answer("") == ""

    def test_single_line(self):
        assert extract_final_answer("Just a number") == "Just a number"


class TestScoreExactMatch:
    def test_exact_match(self):
        assert score_exact_match("42", "42") is True

    def test_case_insensitive(self):
        assert score_exact_match("Paris", "paris") is True

    def test_whitespace_normalized(self):
        assert score_exact_match("  42  ", "42") is True

    def test_no_match(self):
        assert score_exact_match("41", "42") is False


class TestScoreContains:
    def test_contains(self):
        assert score_contains("The answer is 42 degrees", "42") is True

    def test_exact_is_contains(self):
        assert score_contains("42", "42") is True

    def test_no_contains(self):
        assert score_contains("The answer is 41", "42") is False


class TestScoreOpenthoughts:
    def test_empty_results(self):
        metrics = score_openthoughts({})
        assert metrics["total_samples"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_all_correct(self):
        results = {
            "0": {"model_answer": "The answer is 42.", "ground_truth": "42", "error": None},
            "1": {"model_answer": "The answer is Paris.", "ground_truth": "Paris", "error": None},
        }
        metrics = score_openthoughts(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["exact_match_accuracy"] == 100.0
        assert metrics["contains_accuracy"] == 100.0
        assert metrics["error_count"] == 0.0

    def test_partial_correct(self):
        results = {
            "0": {"model_answer": "The answer is 42.", "ground_truth": "42", "error": None},
            "1": {"model_answer": "I think it's 41.", "ground_truth": "42", "error": None},
        }
        metrics = score_openthoughts(results)
        assert metrics["exact_match_accuracy"] == 50.0

    def test_with_errors(self):
        results = {
            "0": {"model_answer": "The answer is 42.", "ground_truth": "42", "error": None},
            "1": {"model_answer": "", "ground_truth": "Paris", "error": "timeout"},
        }
        metrics = score_openthoughts(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["error_count"] == 1.0
        # Only 1 of 2 total is exact match -> 50%
        assert metrics["exact_match_accuracy"] == 50.0

    def test_contains_vs_exact(self):
        # Use a response where the extracted answer contains the gold but isn't exact
        results = {
            "0": {
                "model_answer": "The answer is 42 degrees celsius",
                "ground_truth": "42",
                "error": None,
            },
        }
        metrics = score_openthoughts(results)
        # extract_final_answer gets "42 degrees celsius" from "answer is" pattern
        # contains check finds "42" inside "42 degrees celsius"
        assert metrics["contains_accuracy"] == 100.0
        # exact match fails since "42 degrees celsius" != "42"
        assert metrics["exact_match_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# OpenThoughtsBenchmark tests
# ---------------------------------------------------------------------------


class TestOpenThoughtsBenchmark:
    @patch("evals.benchmarks.openthoughts.main.load_openthoughts_samples")
    def test_generate_responses_with_trace_collector(self, mock_load):
        mock_load.return_value = iter([
            OpenThoughtsSample(
                original_index=0,
                problem="What is 2+2?",
                answer="4",
                domain="math",
            ),
            OpenThoughtsSample(
                original_index=1,
                problem="Capital of France?",
                answer="Paris",
                domain="geography",
            ),
        ])

        benchmark = OpenThoughtsBenchmark(logger=logging.getLogger("test"))
        mock_orchestrator = MagicMock()

        # Mock TraceCollector to return a QueryTrace-like object
        mock_trace = MagicMock()
        mock_trace.response_text = "The answer is 42."
        mock_trace.total_input_tokens = 200
        mock_trace.total_output_tokens = 100
        mock_trace.total_wall_clock_s = 2.5
        mock_trace.to_dict.return_value = {"query_id": "0"}

        mock_collector_cls = MagicMock(return_value=MagicMock(
            run_query_direct_vllm=MagicMock(return_value=mock_trace),
        ))

        fake_mod = types.ModuleType("evals.telemetry.trace_collector")
        fake_mod.TraceCollector = mock_collector_cls
        with patch.dict(sys.modules, {"evals.telemetry.trace_collector": fake_mod}):
            results = benchmark.generate_responses(mock_orchestrator)

        assert len(results) == 2
        assert results["0"]["problem"] == "What is 2+2?"
        assert results["0"]["ground_truth"] == "4"
        assert results["0"]["error"] is None
        assert results["0"]["model_answer"] == "The answer is 42."
        assert results["1"]["domain"] == "geography"

    @patch("evals.benchmarks.openthoughts.main.load_openthoughts_samples")
    def test_generate_responses_handles_error(self, mock_load):
        mock_load.return_value = iter([
            OpenThoughtsSample(
                original_index=0,
                problem="Hard problem",
                answer="42",
                domain="math",
            ),
        ])

        benchmark = OpenThoughtsBenchmark(logger=logging.getLogger("test"))
        mock_orchestrator = MagicMock()

        # Inject a fake trace_collector module that raises on instantiation
        mock_collector_cls = MagicMock(side_effect=Exception("model error"))
        fake_mod = types.ModuleType("evals.telemetry.trace_collector")
        fake_mod.TraceCollector = mock_collector_cls
        with patch.dict(sys.modules, {"evals.telemetry.trace_collector": fake_mod}):
            results = benchmark.generate_responses(mock_orchestrator)

        assert len(results) == 1
        assert results["0"]["error"] == "model error"
        assert results["0"]["model_answer"] == ""

    def test_evaluate_responses(self):
        benchmark = OpenThoughtsBenchmark(logger=logging.getLogger("test"))
        results = {
            "0": {"model_answer": "The answer is 42.", "ground_truth": "42", "error": None},
            "1": {"model_answer": "I think 99.", "ground_truth": "100", "error": None},
        }
        metrics = benchmark.evaluate_responses(results)
        assert metrics["total_samples"] == 2.0
        assert "exact_match_accuracy" in metrics
        assert "contains_accuracy" in metrics

    def test_benchmark_init_defaults(self):
        benchmark = OpenThoughtsBenchmark()
        assert benchmark.limit is None
        assert benchmark.domains is None
        assert benchmark.input_prompt is None
        assert benchmark.vllm_url == "http://localhost:8000"
        assert benchmark.model_name == ""
        assert benchmark.max_tokens == 32768
        assert benchmark.temperature == 0.6

    def test_benchmark_init_custom(self):
        benchmark = OpenThoughtsBenchmark(
            limit=100,
            domains=["math", "science"],
            input_prompt="Solve: {problem}",
            vllm_url="http://gpu:9000",
            model_name="deepseek-r1",
            max_tokens=16384,
            temperature=0.3,
        )
        assert benchmark.limit == 100
        assert benchmark.domains == ["math", "science"]
        assert benchmark.input_prompt == "Solve: {problem}"
        assert benchmark.vllm_url == "http://gpu:9000"
        assert benchmark.model_name == "deepseek-r1"
        assert benchmark.max_tokens == 16384
        assert benchmark.temperature == 0.3


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestOpenThoughtsRegistry:
    def test_register_benchmark(self):
        from evals.registry import get_benchmark

        factory = get_benchmark("openthoughts")
        benchmark = factory(options={})
        assert isinstance(benchmark, OpenThoughtsBenchmark)

    def test_register_benchmark_with_options(self):
        from evals.registry import get_benchmark

        factory = get_benchmark("openthoughts")
        benchmark = factory(options={"limit": 50, "domains": ["math"]})
        assert isinstance(benchmark, OpenThoughtsBenchmark)
        assert benchmark.limit == 50
        assert benchmark.domains == ["math"]
