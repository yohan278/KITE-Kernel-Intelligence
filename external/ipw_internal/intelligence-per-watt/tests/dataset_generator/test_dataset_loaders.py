"""Tests for dataset loaders and registry."""

import pytest
from unittest.mock import patch, MagicMock

from dataset_generator.datasets.base import BaseDatasetLoader, DatasetSample
from dataset_generator.datasets.wildchat import WildChatLoader, ConversationSample, ConversationTurn
from dataset_generator.datasets.openthoughts import OpenThoughtsLoader
from dataset_generator.datasets.hotpotqa import HotpotQALoader
from dataset_generator.datasets.agentdata import AgentDataLoader, AgentStep, TrajectorySample
from dataset_generator.datasets.swebench import SWEBenchLoader
from dataset_generator.datasets.registry import (
    DATASET_REGISTRY,
    list_datasets,
    load_dataset,
)


class TestDatasetSample:
    def test_defaults(self):
        s = DatasetSample(query="hello")
        assert s.query == "hello"
        assert s.expected_answer is None
        assert s.workload_type == ""
        assert s.metadata == {}

    def test_full_fields(self):
        s = DatasetSample(
            query="q",
            expected_answer="a",
            workload_type="chat",
            metadata={"k": "v"},
        )
        assert s.expected_answer == "a"
        assert s.workload_type == "chat"
        assert s.metadata == {"k": "v"}


class TestBaseDatasetLoader:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseDatasetLoader()


class TestWildChatLoader:
    def test_workload_type(self):
        loader = WildChatLoader()
        assert loader.workload_type() == "chat"

    def test_dataset_name(self):
        loader = WildChatLoader()
        assert loader.dataset_name() == "wildchat"

    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_load_filters_english(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "language": "English",
                "conversation": [{"role": "user", "content": "Hello!"}],
                "model": "gpt-4",
            },
            {
                "language": "Chinese",
                "conversation": [{"role": "user", "content": "你好"}],
                "model": "gpt-4",
            },
            {
                "language": "English",
                "conversation": [{"role": "user", "content": "How are you?"}],
                "model": "gpt-3.5",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = WildChatLoader().load(limit=10)

        assert len(samples) == 2
        assert samples[0].query == "Hello!"
        assert samples[1].query == "How are you?"
        assert all(s.workload_type == "chat" for s in samples)
        assert samples[0].metadata["model"] == "gpt-4"

    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_load_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "language": "English",
                "conversation": [{"role": "user", "content": f"msg{i}"}],
                "model": "m",
            }
            for i in range(10)
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = WildChatLoader().load(limit=3)
        assert len(samples) == 3

    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_load_skips_empty_conversations(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {"language": "English", "conversation": [], "model": "m"},
            {
                "language": "English",
                "conversation": [{"role": "assistant", "content": "no user msg"}],
                "model": "m",
            },
            {
                "language": "English",
                "conversation": [{"role": "user", "content": "valid"}],
                "model": "m",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = WildChatLoader().load()
        assert len(samples) == 1
        assert samples[0].query == "valid"

    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_load_conversations(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "language": "English",
                "conversation": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm good!"},
                ],
                "model": "gpt-4",
            },
            {
                "language": "English",
                "conversation": [
                    {"role": "user", "content": "One turn"},
                ],
                "model": "gpt-3.5",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        convos = WildChatLoader().load_conversations()

        assert len(convos) == 2
        assert isinstance(convos[0], ConversationSample)
        assert len(convos[0].turns) == 4
        assert convos[0].turns[0].role == "user"
        assert convos[0].turns[0].content == "Hello!"
        assert convos[0].model == "gpt-4"
        assert len(convos[1].turns) == 1

    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_load_conversations_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "language": "English",
                "conversation": [{"role": "user", "content": f"msg{i}"}],
                "model": "m",
            }
            for i in range(10)
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        convos = WildChatLoader().load_conversations(limit=3)
        assert len(convos) == 3

    @patch("dataset_generator.datasets.wildchat._require_datasets")
    def test_load_conversations_filters_no_user(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "language": "English",
                "conversation": [{"role": "assistant", "content": "no user"}],
                "model": "m",
            },
            {
                "language": "English",
                "conversation": [{"role": "user", "content": "valid"}],
                "model": "m",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        convos = WildChatLoader().load_conversations()
        assert len(convos) == 1


class TestOpenThoughtsLoader:
    def test_workload_type(self):
        assert OpenThoughtsLoader().workload_type() == "reasoning"

    def test_dataset_name(self):
        assert OpenThoughtsLoader().dataset_name() == "openthoughts"

    @patch("dataset_generator.datasets.openthoughts._require_datasets")
    def test_load(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {"problem": "What is 2+2?", "solution": "4", "domain": "math"},
            {"problem": "Explain gravity.", "solution": "Force of attraction.", "domain": "physics"},
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = OpenThoughtsLoader().load()

        assert len(samples) == 2
        assert samples[0].query == "What is 2+2?"
        assert samples[0].expected_answer == "4"
        assert samples[0].workload_type == "reasoning"
        assert samples[0].metadata["domain"] == "math"

    @patch("dataset_generator.datasets.openthoughts._require_datasets")
    def test_load_skips_empty_problem(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {"problem": "", "solution": "empty"},
            {"problem": "Real problem", "solution": "answer"},
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = OpenThoughtsLoader().load()
        assert len(samples) == 1
        assert samples[0].query == "Real problem"

    @patch("dataset_generator.datasets.openthoughts._require_datasets")
    def test_load_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [{"problem": f"p{i}", "solution": f"s{i}", "domain": "d"} for i in range(20)]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = OpenThoughtsLoader().load(limit=5)
        assert len(samples) == 5

    @patch("dataset_generator.datasets.openthoughts._require_datasets")
    def test_load_extracts_deepseek_reasoning(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "problem": "What is 2+2?",
                "solution": "4",
                "domain": "math",
                "conversations": [
                    {"from": "human", "value": "What is 2+2?"},
                    {"from": "assistant", "value": "<think>Let me add 2 and 2.</think>\n\nThe answer is 4."},
                ],
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = OpenThoughtsLoader().load()
        assert len(samples) == 1
        assert samples[0].metadata["deepseek_reasoning"] == "Let me add 2 and 2."

    @patch("dataset_generator.datasets.openthoughts._require_datasets")
    def test_load_no_reasoning_when_no_think_tags(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "problem": "Simple question",
                "solution": "answer",
                "domain": "math",
                "conversations": [
                    {"from": "human", "value": "Simple question"},
                    {"from": "assistant", "value": "The answer is answer."},
                ],
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = OpenThoughtsLoader().load()
        assert "deepseek_reasoning" not in samples[0].metadata


class TestHotpotQALoader:
    def test_workload_type(self):
        assert HotpotQALoader().workload_type() == "rag"

    def test_dataset_name(self):
        assert HotpotQALoader().dataset_name() == "hotpotqa"

    @patch("dataset_generator.datasets.hotpotqa._require_datasets")
    def test_load(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "question": "Who directed Jaws?",
                "answer": "Steven Spielberg",
                "supporting_facts": {"title": ["Jaws"], "sent_id": [0]},
                "type": "bridge",
                "level": "medium",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = HotpotQALoader().load()

        assert len(samples) == 1
        assert samples[0].query == "Who directed Jaws?"
        assert samples[0].expected_answer == "Steven Spielberg"
        assert samples[0].workload_type == "rag"
        assert samples[0].metadata["supporting_facts"] == {"title": ["Jaws"], "sent_id": [0]}
        assert samples[0].metadata["type"] == "bridge"

    @patch("dataset_generator.datasets.hotpotqa._require_datasets")
    def test_load_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = HotpotQALoader().load(limit=3)
        assert len(samples) == 3

    @patch("dataset_generator.datasets.hotpotqa._require_datasets")
    def test_load_extracts_context(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "question": "Who directed Jaws?",
                "answer": "Steven Spielberg",
                "context": {
                    "title": ["Jaws", "Spielberg"],
                    "sentences": [
                        ["Jaws is a 1975 thriller film.", " It was directed by Spielberg."],
                        ["Steven Spielberg is a director.", " He made many films."],
                    ],
                },
                "type": "bridge",
                "level": "medium",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = HotpotQALoader().load()
        assert len(samples) == 1
        context = samples[0].metadata["context"]
        assert "Jaws" in context
        assert "Spielberg" in context
        assert len(context) > 0


class TestAgentDataLoader:
    def test_workload_type(self):
        assert AgentDataLoader().workload_type() == "agentic"

    def test_dataset_name(self):
        assert AgentDataLoader().dataset_name() == "agentdata"

    @patch("dataset_generator.datasets.agentdata._require_datasets")
    def test_load(self, mock_require):
        """Test loading with std format (content list of messages)."""
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        # Std format: content is a list of messages with source/content/class_
        rows = [
            {
                "content": [
                    {"source": "user", "content": "Search for Python docs", "class_": "text_observation"},
                    {"source": None, "content": "search(python)", "class_": "message_action"},
                    {"source": "user", "content": "Found docs", "class_": "text_observation"},
                    {"source": None, "content": "click(link)", "class_": "message_action"},
                    {"source": "user", "content": "Page loaded", "class_": "text_observation"},
                ],
                "details": {},
                "id": 1,
            },
            {
                "content": [
                    {"source": "user", "content": "Calculate compound interest", "class_": "text_observation"},
                    {"source": None, "content": "use_calculator(1000*1.05^10)", "class_": "message_action"},
                    {"source": "user", "content": "1628.89", "class_": "text_observation"},
                ],
                "details": {},
                "id": 2,
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = AgentDataLoader().load()

        assert len(samples) == 2
        assert all(isinstance(s, DatasetSample) for s in samples)
        assert all(s.workload_type == "agentic" for s in samples)
        assert samples[0].query == "Search for Python docs"
        assert samples[0].metadata["num_steps"] == 2
        assert samples[0].metadata["source"] == "agentdata"
        assert samples[1].metadata["num_steps"] == 1

    @patch("dataset_generator.datasets.agentdata._require_datasets")
    def test_load_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "content": [
                    {"source": "user", "content": f"task{i}", "class_": "text_observation"},
                    {"source": None, "content": f"action{i}", "class_": "message_action"},
                    {"source": "user", "content": f"obs{i}", "class_": "text_observation"},
                ],
                "details": {},
                "id": i,
            }
            for i in range(10)
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = AgentDataLoader().load(limit=3)
        assert len(samples) <= 3

    @patch("dataset_generator.datasets.agentdata._require_datasets")
    def test_load_skips_empty_task(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "content": [
                    {"source": "user", "content": "", "class_": "text_observation"},
                ],
                "details": {},
                "id": 1,
            },
            {
                "content": [
                    {"source": "user", "content": "valid task", "class_": "text_observation"},
                    {"source": None, "content": "action", "class_": "message_action"},
                    {"source": "user", "content": "result", "class_": "text_observation"},
                ],
                "details": {},
                "id": 2,
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = AgentDataLoader().load()
        assert len(samples) == 1
        assert samples[0].query == "valid task"

    @patch("dataset_generator.datasets.agentdata._require_datasets")
    def test_load_trajectories(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "content": [
                    {"source": "user", "content": "Browse website", "class_": "text_observation"},
                    {"source": None, "content": "navigate(url)", "class_": "message_action"},
                    {"source": "user", "content": "Page loaded", "class_": "text_observation"},
                    {"source": None, "content": "extract(data)", "class_": "message_action"},
                    {"source": "user", "content": "Data found", "class_": "text_observation"},
                ],
                "details": {},
                "id": 1,
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        trajs = AgentDataLoader().load_trajectories()

        assert len(trajs) == 1
        assert isinstance(trajs[0], TrajectorySample)
        assert trajs[0].task == "Browse website"
        assert len(trajs[0].steps) == 2
        assert isinstance(trajs[0].steps[0], AgentStep)
        assert trajs[0].steps[0].action == "navigate(url)"
        assert trajs[0].steps[0].observation == "Page loaded"

    @patch("dataset_generator.datasets.agentdata._require_datasets")
    def test_load_trajectories_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "content": [
                    {"source": "user", "content": f"task{i}", "class_": "text_observation"},
                    {"source": None, "content": f"action{i}", "class_": "message_action"},
                    {"source": "user", "content": f"obs{i}", "class_": "text_observation"},
                ],
                "details": {},
                "id": i,
            }
            for i in range(10)
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        trajs = AgentDataLoader().load_trajectories(limit=3)
        assert len(trajs) <= 3


class TestSWEBenchLoader:
    def test_workload_type(self):
        assert SWEBenchLoader().workload_type() == "coding"

    def test_dataset_name(self):
        assert SWEBenchLoader().dataset_name() == "swebench"

    @patch("dataset_generator.datasets.swebench._require_datasets")
    def test_load(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {
                "problem_statement": "Fix the off-by-one error in the loop.",
                "patch": "--- a/file.py\n+++ b/file.py\n-for i in range(n+1):\n+for i in range(n):",
                "instance_id": "test__123",
                "repo": "test/repo",
            },
            {
                "problem_statement": "Add input validation.",
                "patch": "--- a/api.py\n+++ b/api.py\n+if not email: raise ValueError",
                "instance_id": "test__456",
                "repo": "test/repo2",
            },
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = SWEBenchLoader().load()

        assert len(samples) == 2
        assert all(isinstance(s, DatasetSample) for s in samples)
        assert all(s.workload_type == "coding" for s in samples)
        assert samples[0].query == "Fix the off-by-one error in the loop."
        assert "off-by-one" not in (samples[0].expected_answer or "").split("off-by-one")[0] or samples[0].expected_answer is not None
        assert samples[0].metadata["source"] == "swebench"
        assert samples[0].metadata["instance_id"] == "test__123"
        assert samples[0].metadata["repo"] == "test/repo"

    @patch("dataset_generator.datasets.swebench._require_datasets")
    def test_load_respects_limit(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {"problem_statement": f"problem{i}", "patch": f"patch{i}", "instance_id": f"id{i}", "repo": "r"}
            for i in range(10)
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = SWEBenchLoader().load(limit=3)
        assert len(samples) == 3

    @patch("dataset_generator.datasets.swebench._require_datasets")
    def test_load_skips_empty_problem(self, mock_require):
        mock_ds_lib = MagicMock()
        mock_require.return_value = mock_ds_lib

        rows = [
            {"problem_statement": "", "patch": "p"},
            {"problem_statement": "Valid problem", "patch": "p"},
        ]
        mock_ds_lib.load_dataset.return_value = iter(rows)

        samples = SWEBenchLoader().load()
        assert len(samples) == 1
        assert samples[0].query == "Valid problem"


class TestRegistry:
    def test_all_loaders_registered(self):
        assert "wildchat" in DATASET_REGISTRY
        assert "openthoughts" in DATASET_REGISTRY
        assert "hotpotqa" in DATASET_REGISTRY
        assert "agentdata" in DATASET_REGISTRY
        assert "swebench" in DATASET_REGISTRY

    def test_list_datasets(self):
        names = list_datasets()
        assert names == ["agentdata", "hotpotqa", "openthoughts", "swebench", "wildchat"]

    def test_load_dataset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset 'nonexistent'"):
            load_dataset("nonexistent")

    def test_registry_values_are_loader_classes(self):
        for name, cls in DATASET_REGISTRY.items():
            assert issubclass(cls, BaseDatasetLoader), f"{name} is not a BaseDatasetLoader"
