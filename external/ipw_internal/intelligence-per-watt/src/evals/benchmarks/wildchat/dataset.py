"""WildChat dataset loader for eval benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class WildChatSample:
    """A single WildChat benchmark sample (multi-turn conversation)."""

    original_index: int
    conversation: List[Dict[str, str]]  # [{"role": "user", "content": ...}, ...]
    model: str = ""
    language: str = "English"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return sum(1 for msg in self.conversation if msg["role"] == "user")

    @property
    def first_user_message(self) -> str:
        for msg in self.conversation:
            if msg["role"] == "user":
                return msg["content"]
        return ""


def load_wildchat_samples(
    *,
    limit: Optional[int] = None,
    min_turns: int = 1,
    max_turns: Optional[int] = None,
) -> Iterator[WildChatSample]:
    """Load WildChat samples wrapping the existing WildChatLoader."""
    from dataset_generator.datasets.wildchat import WildChatLoader

    loader = WildChatLoader()
    conversations = loader.load_conversations(limit=limit)

    for idx, conv in enumerate(conversations):
        turns = [{"role": t.role, "content": t.content} for t in conv.turns]
        user_turn_count = sum(1 for t in turns if t["role"] == "user")

        if user_turn_count < min_turns:
            continue
        if max_turns and user_turn_count > max_turns:
            continue

        yield WildChatSample(
            original_index=idx,
            conversation=turns,
            model=conv.model,
            language=conv.language,
            metadata=conv.metadata,
        )
