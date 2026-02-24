"""Tests for the HotpotQA eval benchmark."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evals.benchmarks.hotpotqa.dataset import HotpotQASample, load_hotpotqa_samples
from evals.benchmarks.hotpotqa.scorer import (
    _normalize_answer,
    exact_match_score,
    extract_answer,
    f1_score,
    score_hotpotqa,
)
from evals.benchmarks.hotpotqa.main import HotpotQABenchmark


# ---------------------------------------------------------------------------
# dataset.py
# ---------------------------------------------------------------------------


class TestHotpotQASample:
    def test_create_sample(self):
        sample = HotpotQASample(
            original_index=0,
            question="Who was the president?",
            answer="Lincoln",
        )
        assert sample.original_index == 0
        assert sample.question == "Who was the president?"
        assert sample.answer == "Lincoln"
        assert sample.context == ""
        assert sample.question_type == ""
        assert sample.level == ""
        assert sample.metadata == {}
        assert sample.supporting_facts == {}

    def test_create_sample_full(self):
        sample = HotpotQASample(
            original_index=5,
            question="What city?",
            answer="Paris",
            context="France is in Europe.",
            supporting_facts={"title": ["France"]},
            question_type="bridge",
            level="hard",
            metadata={"source": "hotpotqa"},
        )
        assert sample.context == "France is in Europe."
        assert sample.question_type == "bridge"
        assert sample.level == "hard"
        assert sample.supporting_facts == {"title": ["France"]}


class TestLoadHotpotQASamples:
    def _make_dataset_sample(self, query, answer, qtype="bridge", level="hard", context="ctx"):
        from dataset_generator.datasets.base import DatasetSample

        return DatasetSample(
            query=query,
            expected_answer=answer,
            workload_type="rag",
            metadata={
                "type": qtype,
                "level": level,
                "context": context,
                "supporting_facts": {},
            },
        )

    @patch("dataset_generator.datasets.hotpotqa.HotpotQALoader")
    def test_load_samples(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            self._make_dataset_sample("Q1?", "A1"),
            self._make_dataset_sample("Q2?", "A2", qtype="comparison"),
        ]
        mock_loader_cls.return_value = mock_loader

        samples = list(load_hotpotqa_samples(limit=10))
        assert len(samples) == 2
        assert samples[0].question == "Q1?"
        assert samples[0].answer == "A1"
        assert samples[0].question_type == "bridge"
        assert samples[1].question_type == "comparison"

    @patch("dataset_generator.datasets.hotpotqa.HotpotQALoader")
    def test_load_samples_filter_question_type(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            self._make_dataset_sample("Q1?", "A1", qtype="bridge"),
            self._make_dataset_sample("Q2?", "A2", qtype="comparison"),
        ]
        mock_loader_cls.return_value = mock_loader

        samples = list(load_hotpotqa_samples(question_types=["bridge"]))
        assert len(samples) == 1
        assert samples[0].question_type == "bridge"


# ---------------------------------------------------------------------------
# scorer.py
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_basic(self):
        assert _normalize_answer("The Answer!") == "answer"

    def test_articles_and_punctuation(self):
        assert _normalize_answer("A cat, a dog.") == "cat dog"

    def test_whitespace(self):
        assert _normalize_answer("  hello   world  ") == "hello world"


class TestExactMatchScore:
    def test_exact_match(self):
        assert exact_match_score("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match_score("paris", "Paris") == 1.0

    def test_mismatch(self):
        assert exact_match_score("London", "Paris") == 0.0

    def test_articles_ignored(self):
        assert exact_match_score("the Paris", "Paris") == 1.0


class TestF1Score:
    def test_perfect_match(self):
        assert f1_score("Abraham Lincoln", "Abraham Lincoln") == 1.0

    def test_partial_match(self):
        score = f1_score("Abraham Lincoln president", "Abraham Lincoln")
        # precision = 2/3, recall = 2/2 = 1
        # f1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
        assert abs(score - 0.8) < 0.01

    def test_no_match(self):
        assert f1_score("London", "Paris") == 0.0

    def test_empty_prediction(self):
        assert f1_score("", "Paris") == 0.0

    def test_empty_ground_truth(self):
        assert f1_score("Paris", "") == 0.0


class TestExtractAnswer:
    def test_answer_is_pattern(self):
        assert extract_answer("The answer is Paris.") == "Paris"

    def test_therefore_pattern(self):
        assert extract_answer("Therefore, the answer is 42.") == "42"

    def test_fallback_last_sentence(self):
        assert extract_answer("Some reasoning. The capital is Paris") == "The capital is Paris"

    def test_empty_string(self):
        assert extract_answer("") == ""


class TestScoreHotpotqa:
    def test_empty_results(self):
        metrics = score_hotpotqa({})
        assert metrics["total_samples"] == 0.0
        assert metrics["em"] == 0.0
        assert metrics["f1"] == 0.0

    def test_perfect_scores(self):
        results = {
            "0": {
                "model_answer": "The answer is Paris.",
                "ground_truth": "Paris",
                "error": None,
            },
        }
        metrics = score_hotpotqa(results)
        assert metrics["total_samples"] == 1.0
        assert metrics["em"] == 100.0
        assert metrics["f1"] == 100.0
        assert metrics["error_count"] == 0.0

    def test_with_errors(self):
        results = {
            "0": {
                "model_answer": "",
                "ground_truth": "Paris",
                "error": "timeout",
            },
        }
        metrics = score_hotpotqa(results)
        assert metrics["total_samples"] == 1.0
        assert metrics["em"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["error_count"] == 1.0

    def test_mixed_results(self):
        results = {
            "0": {
                "model_answer": "The answer is Paris.",
                "ground_truth": "Paris",
                "error": None,
            },
            "1": {
                "model_answer": "The answer is London.",
                "ground_truth": "Paris",
                "error": None,
            },
        }
        metrics = score_hotpotqa(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["em"] == 50.0


# ---------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------


class TestHotpotQABenchmark:
    def test_instantiation(self):
        bench = HotpotQABenchmark(limit=5)
        assert bench.limit == 5
        assert bench.include_context is True
        assert bench.max_tokens == 4096

    def test_build_prompt_with_context(self):
        bench = HotpotQABenchmark()
        sample = HotpotQASample(
            original_index=0,
            question="Who?",
            answer="X",
            context="Some context.",
        )
        prompt = bench._build_prompt(sample)
        assert "Context:" in prompt
        assert "Some context." in prompt
        assert "Question: Who?" in prompt

    def test_build_prompt_without_context(self):
        bench = HotpotQABenchmark(include_context=False)
        sample = HotpotQASample(
            original_index=0,
            question="Who?",
            answer="X",
            context="Some context.",
        )
        prompt = bench._build_prompt(sample)
        assert "Context:" not in prompt
        assert "Question: Who?" in prompt

    def test_evaluate_responses(self):
        bench = HotpotQABenchmark()
        results = {
            "0": {
                "model_answer": "The answer is Paris.",
                "ground_truth": "Paris",
                "error": None,
            },
            "1": {
                "model_answer": "The answer is London.",
                "ground_truth": "Paris",
                "error": None,
            },
        }
        metrics = bench.evaluate_responses(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["em"] == 50.0
        assert "f1" in metrics
