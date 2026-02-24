# benchmarks/deepresearch/__init__.py
"""
DeepResearchBench Benchmark.

Evaluates AI models on PhD-level research report generation with citations.
100 tasks across 22 domains (50 EN + 50 ZH).

Uses RACE + FACT evaluation via Gemini API.

Reference: https://github.com/Ayanami0730/deep_research_bench

Usage:
    from evals.benchmarks.deepresearch import load_deepresearch_samples

    samples = list(load_deepresearch_samples(limit=10))
"""

# Dataset - always available
from evals.benchmarks.deepresearch.dataset import (
    DeepResearchSample,
    load_deepresearch_samples,
    get_deepresearch_domains,
    DEFAULT_INPUT_PROMPT,
)

__all__ = [
    # Dataset
    "DeepResearchSample",
    "load_deepresearch_samples",
    "get_deepresearch_domains",
    "DEFAULT_INPUT_PROMPT",
]
