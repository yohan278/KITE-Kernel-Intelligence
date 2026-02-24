"""OpenThoughts dataset loader for eval benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


DEFAULT_INPUT_PROMPT = """Solve the following problem. Show your reasoning step by step, then provide your final answer.

Problem: {problem}

Think carefully and show your work."""


@dataclass
class OpenThoughtsSample:
    """A single OpenThoughts benchmark sample."""

    original_index: int
    problem: str
    answer: str
    domain: str = ""
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_prompt(self, input_prompt: Optional[str] = None) -> str:
        prompt = input_prompt or DEFAULT_INPUT_PROMPT
        return prompt.format(problem=self.problem)


def load_openthoughts_samples(
    *,
    limit: Optional[int] = None,
    domains: Optional[List[str]] = None,
) -> Iterator[OpenThoughtsSample]:
    """Load OpenThoughts samples wrapping the existing OpenThoughtsLoader."""
    from dataset_generator.datasets.openthoughts import OpenThoughtsLoader

    loader = OpenThoughtsLoader()
    samples = loader.load(limit=limit)

    for idx, sample in enumerate(samples):
        domain = sample.metadata.get("domain", "")
        if domains and domain not in domains:
            continue

        yield OpenThoughtsSample(
            original_index=idx,
            problem=sample.query,
            answer=sample.expected_answer or "",
            domain=domain,
            reasoning=sample.metadata.get("deepseek_reasoning", ""),
            metadata=sample.metadata,
        )
