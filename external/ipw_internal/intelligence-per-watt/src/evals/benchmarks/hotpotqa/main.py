"""HotpotQA benchmark: multi-hop QA with retrieval."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.hotpotqa.dataset import HotpotQASample, load_hotpotqa_samples
from evals.benchmarks.hotpotqa.scorer import score_hotpotqa


class HotpotQABenchmark(DatasetBenchmark):
    """Multi-hop QA benchmark using HotpotQA with retrieval.

    Provides context paragraphs to the model and asks multi-hop questions.
    For the eval benchmark, we include context directly in the prompt
    (simulating perfect retrieval) or use FAISS-based retrieval.
    Scores using standard EM + F1 metrics.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        question_types: Optional[List[str]] = None,
        include_context: bool = True,
        vllm_url: str = "http://localhost:8000",
        model_name: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)
        self.limit = limit
        self.question_types = question_types
        self.include_context = include_context
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._samples: List[HotpotQASample] = []

    def _load_samples(self) -> List[HotpotQASample]:
        if not self._samples:
            self._samples = list(
                load_hotpotqa_samples(
                    limit=self.limit,
                    question_types=self.question_types,
                )
            )
            self.logger.info(f"Loaded {len(self._samples)} HotpotQA samples")
        return self._samples

    def _build_prompt(self, sample: HotpotQASample) -> str:
        """Build prompt with optional context."""
        if self.include_context and sample.context:
            return (
                "Answer the following question using the provided context. "
                "Give a short, direct answer.\n\n"
                f"Context:\n{sample.context}\n\n"
                f"Question: {sample.question}\n\n"
                "Answer:"
            )
        return (
            "Answer the following question. Give a short, direct answer.\n\n"
            f"Question: {sample.question}\n\n"
            "Answer:"
        )

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Generate answers for all HotpotQA samples."""
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] {sample.question_type}/{sample.level}: "
                f"{sample.question[:60]}..."
            )
            start_time = time.time()

            try:
                try:
                    from evals.telemetry.trace_collector import TraceCollector

                    collector = TraceCollector(
                        vllm_url=self.vllm_url, model_name=self.model_name
                    )
                    prompt = self._build_prompt(sample)
                    trace = collector.run_query_direct_vllm(
                        query_id=str(sample.original_index),
                        workload_type="rag",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "question": sample.question,
                        "ground_truth": sample.answer,
                        "model_answer": trace.response_text,
                        "question_type": sample.question_type,
                        "level": sample.level,
                        "input_tokens": trace.total_input_tokens,
                        "output_tokens": trace.total_output_tokens,
                        "wall_clock_s": trace.total_wall_clock_s,
                        "error": None,
                        "trace": trace.to_dict(),
                    }
                except ImportError:
                    prompt = self._build_prompt(sample)
                    response = orchestrator.run(prompt)
                    elapsed = time.time() - start_time
                    response_text = (
                        str(getattr(response, "content", response))
                        if response
                        else ""
                    )
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "question": sample.question,
                        "ground_truth": sample.answer,
                        "model_answer": response_text,
                        "question_type": sample.question_type,
                        "level": sample.level,
                        "wall_clock_s": elapsed,
                        "error": None,
                    }
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Error: {e}")
                results[str(sample.original_index)] = {
                    "original_index": sample.original_index,
                    "question": sample.question,
                    "ground_truth": sample.answer,
                    "model_answer": "",
                    "question_type": sample.question_type,
                    "level": sample.level,
                    "wall_clock_s": elapsed,
                    "error": str(e),
                }

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Score using EM + F1."""
        metrics = score_hotpotqa(results)
        self.logger.info(f"HotpotQA: EM={metrics['em']}%, F1={metrics['f1']}%")
        return metrics


@register_benchmark("hotpotqa")
def _create_hotpotqa_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> HotpotQABenchmark:
    options = options or {}
    return HotpotQABenchmark(**options)
