# benchmarks/frames/dataset.py
"""
FRAMES (Factual Retrieval And Multi-hop Evaluation Suite) dataset loader.

Loads the google/frames-benchmark dataset from HuggingFace for evaluating
multi-hop factual retrieval requiring 2-15 Wikipedia articles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset

HF_DATASET_PATH = "google/frames-benchmark"


@dataclass
class FRAMESSample:
    """A single FRAMES benchmark sample."""

    index: int
    prompt: str           # Multi-hop question
    answer: str           # Gold answer
    reasoning_types: str  # numerical, tabular, temporal, etc.
    wiki_links: List[str] # Relevant Wikipedia URLs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_prompt(self, input_prompt: Optional[str] = None) -> str:
        """Get the formatted prompt for this sample."""
        prompt = input_prompt or DEFAULT_INPUT_PROMPT

        # Include wiki links as context hints if available
        wiki_context = ""
        if self.wiki_links:
            wiki_context = (
                f"\n\nRelevant Wikipedia articles that may help answer this question:\n"
                + "\n".join(f"- {link}" for link in self.wiki_links)
            )

        return prompt.format(question=self.prompt, wiki_context=wiki_context)


def load_frames_samples(
    *,
    split: str = "test",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    reasoning_types: Optional[List[str]] = None,
) -> Iterator[FRAMESSample]:
    """
    Load FRAMES benchmark samples from HuggingFace.

    Args:
        split: Dataset split to load (default: "test")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        reasoning_types: Optional list of reasoning types to filter
                        (e.g., ["numerical", "tabular", "temporal"])

    Yields:
        FRAMESSample objects ready for evaluation
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for idx, row in enumerate(ds):
        # Extract reasoning types
        reasoning = row.get("reasoning_types", row.get("reasoning_type", ""))
        if isinstance(reasoning, list):
            reasoning = ", ".join(reasoning)

        # Filter by reasoning type if specified
        if reasoning_types:
            row_types = [t.strip().lower() for t in reasoning.split(",")]
            if not any(rt.lower() in row_types for rt in reasoning_types):
                continue

        # Extract wiki links
        wiki_links_raw = row.get("wiki_links", row.get("wikipedia_links", []))
        if isinstance(wiki_links_raw, str):
            wiki_links = [link.strip() for link in wiki_links_raw.split(",") if link.strip()]
        elif isinstance(wiki_links_raw, list):
            wiki_links = wiki_links_raw
        else:
            wiki_links = []

        # Extract prompt/question
        prompt = row.get("Prompt", row.get("prompt", row.get("question", "")))

        # Extract answer
        answer = row.get("Answer", row.get("answer", row.get("gold_answer", "")))

        # Collect remaining fields as metadata
        known_fields = {
            "Prompt", "prompt", "question", "Answer", "answer", "gold_answer",
            "reasoning_types", "reasoning_type", "wiki_links", "wikipedia_links"
        }
        metadata = {k: v for k, v in row.items() if k not in known_fields}

        yield FRAMESSample(
            index=idx,
            prompt=prompt,
            answer=str(answer) if answer is not None else "",
            reasoning_types=reasoning,
            wiki_links=wiki_links,
            metadata=metadata,
        )


def get_frames_reasoning_types() -> List[str]:
    """Get list of reasoning types in the FRAMES dataset."""
    return [
        "numerical",
        "tabular",
        "temporal",
        "multi-hop",
        "comparison",
        "aggregation",
    ]


DEFAULT_INPUT_PROMPT = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.
- This question may require multi-hop reasoning across multiple Wikipedia articles.
{wiki_context}

Here is the question:

{question}"""
