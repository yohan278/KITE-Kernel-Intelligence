# benchmarks/simpleqa/__init__.py
"""
SimpleQA Verified Benchmark.

Evaluates AI models on short-form factual QA testing parametric knowledge
across diverse topics (Politics, Art, History, Sports, etc.).

Uses the openai/simple-evals dataset from HuggingFace.

Usage:
    from evals.benchmarks.simpleqa import load_simpleqa_samples

    samples = list(load_simpleqa_samples(limit=100))
"""

# Dataset - always available
from evals.benchmarks.simpleqa.dataset import (
    SimpleQASample,
    load_simpleqa_samples,
    get_simpleqa_topics,
    get_simpleqa_answer_types,
    DEFAULT_INPUT_PROMPT,
)

__all__ = [
    # Dataset
    "SimpleQASample",
    "load_simpleqa_samples",
    "get_simpleqa_topics",
    "get_simpleqa_answer_types",
    "DEFAULT_INPUT_PROMPT",
]
