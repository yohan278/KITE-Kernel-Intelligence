"""HotpotQA dataset loader for eval benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class HotpotQASample:
    """A single HotpotQA benchmark sample."""

    original_index: int
    question: str
    answer: str
    context: str = ""
    supporting_facts: Dict[str, Any] = field(default_factory=dict)
    question_type: str = ""
    level: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_hotpotqa_samples(
    *,
    limit: Optional[int] = None,
    question_types: Optional[List[str]] = None,
) -> Iterator[HotpotQASample]:
    """Load HotpotQA samples wrapping the existing HotpotQALoader."""
    from dataset_generator.datasets.hotpotqa import HotpotQALoader

    loader = HotpotQALoader()
    samples = loader.load(limit=limit)

    for idx, sample in enumerate(samples):
        qtype = sample.metadata.get("type", "")
        if question_types and qtype not in question_types:
            continue

        yield HotpotQASample(
            original_index=idx,
            question=sample.query,
            answer=sample.expected_answer or "",
            context=sample.metadata.get("context", ""),
            supporting_facts=sample.metadata.get("supporting_facts", {}),
            question_type=qtype,
            level=sample.metadata.get("level", ""),
            metadata=sample.metadata,
        )
