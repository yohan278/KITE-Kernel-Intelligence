"""Tests for the AgentData eval benchmark."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evals.benchmarks.agentdata.dataset import AgentDataSample, load_agentdata_samples
from evals.benchmarks.agentdata.scorer import (
    score_agentdata,
    score_task_completion,
)
from evals.benchmarks.agentdata.main import AgentDataBenchmark


# ---------------------------------------------------------------------------
# dataset.py
# ---------------------------------------------------------------------------


class TestAgentDataSample:
    def test_create_sample(self):
        sample = AgentDataSample(
            original_index=0,
            task="Do something",
        )
        assert sample.original_index == 0
        assert sample.task == "Do something"
        assert sample.expected_steps == 0
        assert sample.expected_answer is None
        assert sample.domain == ""
        assert sample.metadata == {}

    def test_create_sample_full(self):
        sample = AgentDataSample(
            original_index=3,
            task="Navigate to the kitchen",
            expected_steps=5,
            expected_answer="Done",
            domain="agenttuning_alfworld",
            metadata={"config": "alfworld"},
        )
        assert sample.expected_steps == 5
        assert sample.expected_answer == "Done"
        assert sample.domain == "agenttuning_alfworld"


class TestLoadAgentdataSamples:
    def _make_trajectory(self, task, steps_count, domain="alfworld", final_answer="done"):
        from dataset_generator.datasets.agentdata import AgentStep, TrajectorySample

        steps = [AgentStep(action=f"act{i}", observation=f"obs{i}") for i in range(steps_count)]
        return TrajectorySample(
            task=task,
            steps=steps,
            final_answer=final_answer,
            metadata={"domain": domain, "source": "agentdata"},
        )

    @patch("dataset_generator.datasets.agentdata.AgentDataLoader")
    def test_load_samples(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader.load_trajectories.return_value = [
            self._make_trajectory("Task A", 3, domain="alfworld"),
            self._make_trajectory("Task B", 5, domain="webshop"),
        ]
        mock_loader_cls.return_value = mock_loader

        samples = list(load_agentdata_samples(limit=10))
        assert len(samples) == 2
        assert samples[0].task == "Task A"
        assert samples[0].expected_steps == 3
        assert samples[0].domain == "alfworld"
        assert samples[1].expected_steps == 5

    @patch("dataset_generator.datasets.agentdata.AgentDataLoader")
    def test_load_samples_filter_domain(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader.load_trajectories.return_value = [
            self._make_trajectory("Task A", 3, domain="alfworld"),
            self._make_trajectory("Task B", 5, domain="webshop"),
        ]
        mock_loader_cls.return_value = mock_loader

        samples = list(load_agentdata_samples(domains=["webshop"]))
        assert len(samples) == 1
        assert samples[0].domain == "webshop"


# ---------------------------------------------------------------------------
# scorer.py
# ---------------------------------------------------------------------------


class TestScoreTaskCompletion:
    def test_empty_response(self):
        assert score_task_completion("Do task", "") == "FAILED"

    def test_whitespace_only(self):
        assert score_task_completion("Do task", "   ") == "FAILED"

    def test_short_response(self):
        assert score_task_completion("Do task", "ok") == "FAILED"

    def test_error_indicator(self):
        assert score_task_completion("Do task", "I cannot complete this request") == "PARTIAL"

    def test_failed_indicator(self):
        assert score_task_completion("Do task", "The operation failed due to issues") == "PARTIAL"

    def test_successful_response(self):
        assert score_task_completion("Do task", "I have completed the task successfully and here are the results") == "COMPLETED"


class TestScoreAgentdata:
    def test_empty_results(self):
        metrics = score_agentdata({})
        assert metrics["total_samples"] == 0.0
        assert metrics["completion_rate"] == 0.0

    def test_all_completed(self):
        results = {
            "0": {"grade": "COMPLETED", "error": None},
            "1": {"grade": "COMPLETED", "error": None},
        }
        metrics = score_agentdata(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["completed_count"] == 2.0
        assert metrics["completion_rate"] == 100.0
        assert metrics["success_rate"] == 100.0

    def test_mixed_grades(self):
        results = {
            "0": {"grade": "COMPLETED", "error": None},
            "1": {"grade": "PARTIAL", "error": None},
            "2": {"grade": "FAILED", "error": None},
        }
        metrics = score_agentdata(results)
        assert metrics["total_samples"] == 3.0
        assert metrics["completed_count"] == 1.0
        assert metrics["partial_count"] == 1.0
        assert metrics["failed_count"] == 1.0
        assert metrics["completion_rate"] == round((1 / 3) * 100, 2)
        assert metrics["success_rate"] == round((2 / 3) * 100, 2)

    def test_with_errors(self):
        results = {
            "0": {"grade": "COMPLETED", "error": None},
            "1": {"grade": "FAILED", "error": "timeout"},
        }
        metrics = score_agentdata(results)
        assert metrics["total_samples"] == 2.0
        assert metrics["error_count"] == 1.0
        assert metrics["failed_count"] == 1.0  # error entry counted as failed


# ---------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------


class TestAgentDataBenchmark:
    def test_instantiation(self):
        bench = AgentDataBenchmark(limit=10)
        assert bench.limit == 10
        assert bench.max_tokens == 8192
        assert bench.temperature == 0.3

    def test_build_prompt(self):
        bench = AgentDataBenchmark()
        sample = AgentDataSample(
            original_index=0,
            task="Navigate to the kitchen",
        )
        prompt = bench._build_prompt(sample)
        assert "Task: Navigate to the kitchen" in prompt
        assert "Provide your response:" in prompt

    def test_evaluate_responses(self):
        bench = AgentDataBenchmark()
        results = {
            "0": {"grade": "COMPLETED", "error": None},
            "1": {"grade": "PARTIAL", "error": None},
            "2": {"grade": "FAILED", "error": None},
        }
        metrics = bench.evaluate_responses(results)
        assert metrics["total_samples"] == 3.0
        assert metrics["completed_count"] == 1.0
        assert "completion_rate" in metrics
        assert "success_rate" in metrics
