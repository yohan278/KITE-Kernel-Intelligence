# benchmarks/frames/__init__.py
"""
FRAMES (Factual Retrieval And Multi-hop Evaluation Suite) Benchmark.

Evaluates AI models on multi-hop factual retrieval questions that require
information synthesis from 2-15 Wikipedia articles.

Uses the google/frames-benchmark dataset from HuggingFace.

Usage:
    from evals.benchmarks.frames import FRAMESBenchmark

    benchmark = FRAMESBenchmark(limit=10)
    results = benchmark.run_benchmark(orchestrator)

Or via registry:
    from evals.registry import get_benchmark

    benchmark = get_benchmark("frames")(options={"limit": 10})
"""

from evals.benchmarks.frames.main import FRAMESBenchmark, FRAMESResult
from evals.benchmarks.frames.dataset import (
    FRAMESSample,
    load_frames_samples,
    get_frames_reasoning_types,
    DEFAULT_INPUT_PROMPT,
)
from evals.benchmarks.frames.scorer import (
    question_scorer,
    grade_answer,
    grade_answer_async,
    normalize_str,
    normalize_number_str,
    question_scorer_exact_match,
)

__all__ = [
    # Main benchmark
    "FRAMESBenchmark",
    "FRAMESResult",
    # Dataset
    "FRAMESSample",
    "load_frames_samples",
    "get_frames_reasoning_types",
    "DEFAULT_INPUT_PROMPT",
    # Scorer (LLM-based)
    "question_scorer",
    "grade_answer",
    "grade_answer_async",
    # Scorer (fallback)
    "question_scorer_exact_match",
    "normalize_str",
    "normalize_number_str",
]
