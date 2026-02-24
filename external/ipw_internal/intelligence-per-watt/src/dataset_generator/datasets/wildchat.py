"""WildChat dataset loader — real-world chat conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dataset_generator.datasets.base import BaseDatasetLoader, DatasetSample


def _require_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        )


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    role: str
    content: str


@dataclass
class ConversationSample:
    """A full multi-turn conversation from WildChat."""

    turns: List[ConversationTurn]
    model: str = ""
    language: str = "English"
    metadata: Dict[str, Any] = field(default_factory=dict)


class WildChatLoader(BaseDatasetLoader):
    """Load chat queries from allenai/WildChat-4.8M."""

    def load(self, limit: Optional[int] = None) -> List[DatasetSample]:
        ds_lib = _require_datasets()
        ds = ds_lib.load_dataset(
            "allenai/WildChat-4.8M",
            split="train",
            streaming=True,
        )

        samples: List[DatasetSample] = []
        for row in ds:
            if limit is not None and len(samples) >= limit:
                break

            # Filter English conversations
            if row.get("language") != "English":
                continue

            # Extract first user message as query
            conversation = row.get("conversation", [])
            query = None
            for turn in conversation:
                if turn.get("role") == "user":
                    query = turn.get("content", "")
                    break

            if not query:
                continue

            samples.append(DatasetSample(
                query=query,
                workload_type=self.workload_type(),
                metadata={"source": "wildchat", "model": row.get("model", "")},
            ))

        return samples

    def load_conversations(
        self, limit: Optional[int] = None
    ) -> List[ConversationSample]:
        """Load full multi-turn conversations.

        Returns ConversationSample instances with all turns preserved,
        useful for characterizing turn counts and per-turn token lengths.
        """
        ds_lib = _require_datasets()
        ds = ds_lib.load_dataset(
            "allenai/WildChat-4.8M",
            split="train",
            streaming=True,
        )

        conversations: List[ConversationSample] = []
        for row in ds:
            if limit is not None and len(conversations) >= limit:
                break

            if row.get("language") != "English":
                continue

            raw_turns = row.get("conversation", [])
            if not raw_turns:
                continue

            turns = [
                ConversationTurn(
                    role=t.get("role", ""),
                    content=t.get("content", ""),
                )
                for t in raw_turns
            ]

            # Skip conversations with no user turn
            if not any(t.role == "user" for t in turns):
                continue

            conversations.append(ConversationSample(
                turns=turns,
                model=row.get("model", ""),
                language=row.get("language", "English"),
                metadata={"source": "wildchat"},
            ))

        return conversations

    def workload_type(self) -> str:
        return "chat"

    def dataset_name(self) -> str:
        return "wildchat"
