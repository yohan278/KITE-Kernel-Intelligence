"""Tests for dataset characterizers and the characterization registry."""

import pytest
from unittest.mock import patch, MagicMock

from dataset_generator.characterization.base import BaseCharacterizer
from dataset_generator.characterization.tokenizer import FastTokenCounter
from dataset_generator.characterization.registry import (
    CHARACTERIZER_REGISTRY,
    characterize_workload,
    register_characterizer,
    _ensure_registered,
)
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


def _mock_fit(data, candidates=("lognormal", "gamma", "normal")):
    """Stub for FittedDistribution.fit that avoids scipy dependency."""
    import numpy as np
    arr = np.array(data, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return FittedDistribution(dist_name="empirical", n_samples=0)
    return FittedDistribution(
        dist_name="empirical",
        n_samples=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        empirical_samples=arr.tolist(),
    )


class TestFastTokenCounter:
    def test_empty_string(self):
        tc = FastTokenCounter()
        assert tc.count("") == 0

    def test_nonempty_string(self):
        tc = FastTokenCounter()
        result = tc.count("Hello world, this is a test sentence.")
        assert result > 0

    def test_fallback_word_count(self):
        tc = FastTokenCounter()
        tc._tried_import = True
        tc._encoder = None
        # Fallback: max(1, int(word_count * 1.3))
        result = tc.count("one two three")
        assert result == max(1, int(3 * 1.3))

    def test_single_word_fallback(self):
        tc = FastTokenCounter()
        tc._tried_import = True
        tc._encoder = None
        result = tc.count("hello")
        assert result == max(1, int(1 * 1.3))


class TestBaseCharacterizer:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseCharacterizer()


def _make_wildchat_rows(n=20):
    """Generate mock WildChat rows for testing."""
    rows = []
    for i in range(n):
        rows.append({
            "language": "English",
            "conversation": [
                {"role": "user", "content": f"Hello, this is user message {i} with some text."},
                {"role": "assistant", "content": f"Response {i} from the assistant with details."},
                {"role": "user", "content": f"Follow-up question {i} about the topic."},
                {"role": "assistant", "content": f"Another response {i} with more information."},
            ],
            "model": "gpt-4" if i % 2 == 0 else "gpt-3.5",
        })
    return rows


def _make_openthoughts_rows(n=20):
    """Generate mock OpenThoughts rows for testing."""
    rows = []
    for i in range(n):
        rows.append({
            "problem": f"What is {i} + {i}? Explain your reasoning step by step.",
            "solution": f"The answer is {2*i}.",
            "domain": "math" if i % 2 == 0 else "physics",
            "conversations": [
                {"from": "human", "value": f"What is {i} + {i}?"},
                {"from": "assistant", "value": f"<think>Let me think about {i} + {i} carefully.</think>\n\nThe answer is {2*i}."},
            ],
        })
    return rows


def _make_hotpotqa_rows(n=20):
    """Generate mock HotpotQA rows for testing."""
    rows = []
    for i in range(n):
        rows.append({
            "question": f"Who directed movie {i}? This requires multi-hop reasoning.",
            "answer": f"Director {i}",
            "context": {
                "title": [f"Movie {i}", f"Director {i}"],
                "sentences": [
                    [f"Movie {i} was released in 2020.", f" It was directed by Director {i}."],
                    [f"Director {i} is a filmmaker.", f" Known for many works."],
                ],
            },
            "supporting_facts": {"title": [f"Movie {i}"], "sent_id": [0]},
            "type": "bridge" if i % 2 == 0 else "comparison",
            "level": "medium",
        })
    return rows


def _make_agentdata_rows(n=20):
    """Generate mock agent-data-collection rows in std format for testing."""
    rows = []
    for i in range(n):
        rows.append({
            "content": [
                {"source": "user", "content": f"Search for information about topic {i} on the web and summarize.", "class_": "text_observation"},
                {"source": None, "content": f"search(topic_{i})", "class_": "message_action"},
                {"source": "user", "content": f"Found {i} results about topic {i}.", "class_": "text_observation"},
                {"source": None, "content": f"click(result_{i})", "class_": "message_action"},
                {"source": "user", "content": f"Page about topic {i} loaded.", "class_": "text_observation"},
                {"source": None, "content": f"extract(content_{i})", "class_": "message_action"},
                {"source": "user", "content": f"Extracted content about topic {i}.", "class_": "text_observation"},
            ],
            "details": {"domain": "web_browsing" if i % 2 == 0 else "knowledge"},
            "id": i,
        })
    return rows


def _make_swebench_rows(n=20):
    """Generate mock SWE-bench rows for testing."""
    rows = []
    for i in range(n):
        rows.append({
            "problem_statement": f"There is a bug in the {i}th function. The loop has an off-by-one error that causes incorrect results.",
            "patch": f"--- a/file{i}.py\n+++ b/file{i}.py\n-for j in range({i}+1):\n+for j in range({i}):",
            "instance_id": f"test__{i}",
            "repo": f"org/repo{i % 5}",
        })
    return rows


class TestChatCharacterizer:
    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_characterize(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_wildchat_rows(20))

        from dataset_generator.characterization.chat_characterizer import ChatCharacterizer
        profile = ChatCharacterizer().characterize()

        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == "chat"
        assert profile.source_dataset == "wildchat"
        assert profile.n_samples == 20
        assert profile.turns_or_steps_dist is not None
        assert profile.input_tokens_dist is not None
        assert profile.answer_tokens_dist is not None
        assert len(profile.domain_mix) > 0

    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_characterize_with_limit(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_wildchat_rows(20))

        from dataset_generator.characterization.chat_characterizer import ChatCharacterizer
        profile = ChatCharacterizer().characterize(limit=5)
        assert profile.n_samples == 5


class TestReasoningCharacterizer:
    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.openthoughts._require_datasets")
    def test_characterize(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_openthoughts_rows(20))

        from dataset_generator.characterization.reasoning_characterizer import ReasoningCharacterizer
        profile = ReasoningCharacterizer().characterize()

        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == "reasoning"
        assert profile.source_dataset == "openthoughts"
        assert profile.n_samples == 20
        assert profile.input_tokens_dist is not None
        assert profile.answer_tokens_dist is not None
        assert profile.thinking_tokens_dist is not None
        assert len(profile.domain_mix) > 0


class TestRAGCharacterizer:
    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.hotpotqa._require_datasets")
    def test_characterize(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_hotpotqa_rows(20))

        from dataset_generator.characterization.rag_characterizer import RAGCharacterizer
        profile = RAGCharacterizer().characterize()

        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == "rag"
        assert profile.source_dataset == "hotpotqa"
        assert profile.n_samples == 20
        assert profile.input_tokens_dist is not None
        assert profile.answer_tokens_dist is not None
        assert profile.tool_call_probability > 0
        assert profile.tool_call_tokens_dist is not None
        assert len(profile.domain_mix) > 0


class TestAgenticCharacterizer:
    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.agentdata._require_datasets")
    def test_characterize(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_agentdata_rows(20))

        from dataset_generator.characterization.agentic_characterizer import AgenticCharacterizer
        profile = AgenticCharacterizer().characterize()

        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == "agentic"
        assert profile.source_dataset == "agentdata"
        assert profile.n_samples == 20
        assert profile.turns_or_steps_dist is not None
        assert profile.input_tokens_dist is not None
        assert profile.answer_tokens_dist is not None
        assert profile.tool_call_probability == 1.0
        assert profile.tool_call_tokens_dist is not None
        assert len(profile.domain_mix) > 0


class TestCodingCharacterizer:
    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.swebench._require_datasets")
    def test_characterize(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_swebench_rows(20))

        from dataset_generator.characterization.coding_characterizer import CodingCharacterizer
        profile = CodingCharacterizer().characterize()

        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == "coding"
        assert profile.source_dataset == "swebench"
        assert profile.n_samples == 20
        assert profile.input_tokens_dist is not None
        assert profile.answer_tokens_dist is not None
        assert len(profile.domain_mix) > 0


class TestCharacterizerRegistry:
    def test_registry_populated(self):
        _ensure_registered()
        assert "wildchat" in CHARACTERIZER_REGISTRY
        assert "openthoughts" in CHARACTERIZER_REGISTRY
        assert "hotpotqa" in CHARACTERIZER_REGISTRY
        assert "agentdata" in CHARACTERIZER_REGISTRY
        assert "swebench" in CHARACTERIZER_REGISTRY

    def test_registry_values_are_characterizer_classes(self):
        _ensure_registered()
        for name, cls in CHARACTERIZER_REGISTRY.items():
            assert issubclass(cls, BaseCharacterizer), f"{name} is not a BaseCharacterizer"

    def test_characterize_workload_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown characterizer"):
            characterize_workload("nonexistent")

    @patch("inference_simulator.types.fitted_distribution.FittedDistribution.fit", side_effect=_mock_fit)
    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_characterize_workload_dispatches(self, mock_require, mock_fit):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib
        mock_ds_lib.load_dataset.return_value = iter(_make_wildchat_rows(15))

        profile = characterize_workload("wildchat", limit=15)
        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == "chat"
