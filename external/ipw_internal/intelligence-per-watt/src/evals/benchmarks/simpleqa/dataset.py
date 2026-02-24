# benchmarks/simpleqa/dataset.py
"""
SimpleQA Verified dataset loader.

Loads the google/simpleqa-verified dataset from HuggingFace for evaluating
short-form factual QA testing parametric knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset

HF_DATASET_PATH = "basicv8vc/SimpleQA"


@dataclass
class SimpleQASample:
    """A single SimpleQA Verified benchmark sample."""

    original_index: int
    problem: str         # The question
    answer: str          # Gold answer
    topic: str           # Politics, Art, History, Sports, etc.
    answer_type: str     # Person, Place, Date, Number, Other
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_prompt(self, input_prompt: Optional[str] = None) -> str:
        """Get the formatted prompt for this sample."""
        prompt = input_prompt or DEFAULT_INPUT_PROMPT
        return prompt.format(question=self.problem)


def load_simpleqa_samples(
    *,
    split: str = "test",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    topics: Optional[List[str]] = None,
    answer_types: Optional[List[str]] = None,
) -> Iterator[SimpleQASample]:
    """
    Load SimpleQA Verified samples from HuggingFace.

    Args:
        split: Dataset split/config to load (default: "simpleqa")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        topics: Optional list of topics to filter (e.g., ["Politics", "Art"])
        answer_types: Optional list of answer types to filter (e.g., ["Person", "Date"])

    Yields:
        SimpleQASample objects ready for evaluation
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for idx, row in enumerate(ds):
        # Parse metadata - it may be a JSON string or dict
        raw_metadata = row.get("metadata", {})
        if isinstance(raw_metadata, str):
            import ast
            try:
                parsed_metadata = ast.literal_eval(raw_metadata)
            except (ValueError, SyntaxError):
                parsed_metadata = {}
        else:
            parsed_metadata = raw_metadata or {}

        # Extract topic
        topic = parsed_metadata.get("topic", row.get("topic", "General"))

        # Filter by topic if specified
        if topics and topic not in topics:
            continue

        # Extract answer type
        answer_type = parsed_metadata.get("answer_type", row.get("answer_type", "Other"))

        # Filter by answer_type if specified
        if answer_types and answer_type not in answer_types:
            continue

        # Extract problem/question
        problem = row.get("problem", row.get("question", ""))

        # Extract answer
        answer = row.get("answer", row.get("gold_answer", ""))

        # Collect remaining fields as metadata
        known_fields = {
            "problem", "question", "answer", "gold_answer",
            "topic", "answer_type", "metadata"
        }
        extra_metadata = {k: v for k, v in row.items() if k not in known_fields}

        # Merge with parsed metadata
        metadata = parsed_metadata.copy()
        metadata.update(extra_metadata)

        yield SimpleQASample(
            original_index=idx,
            problem=problem,
            answer=str(answer) if answer is not None else "",
            topic=topic,
            answer_type=answer_type,
            metadata=metadata,
        )


def get_simpleqa_topics() -> List[str]:
    """Get list of topics in the SimpleQA dataset."""
    return [
        "Science",
        "Technology",
        "History",
        "Geography",
        "Politics",
        "Art",
        "Music",
        "Literature",
        "Sports",
        "Entertainment",
        "Other",
    ]


def get_simpleqa_answer_types() -> List[str]:
    """Get list of answer types in the SimpleQA dataset."""
    return [
        "Person",
        "Place",
        "Date",
        "Number",
        "Organization",
        "Other",
    ]


DEFAULT_INPUT_PROMPT = """Please answer the question below with a short, factual response.

- Return only your answer, which should be a word, phrase, name, number, or date.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a name, use the full name without abbreviations.
- Do not include any explanations or additional context.

Question: {question}"""
