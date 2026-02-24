"""OpenThoughts benchmark: reasoning with chain-of-thought."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.openthoughts.dataset import (
    OpenThoughtsSample,
    load_openthoughts_samples,
)
from evals.benchmarks.openthoughts.scorer import score_openthoughts


class OpenThoughtsBenchmark(DatasetBenchmark):
    """Reasoning benchmark using OpenThoughts problems.

    Sends problems to vLLM with high max_tokens for chain-of-thought.
    Extracts final answer from output and compares with gold answer.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        domains: Optional[List[str]] = None,
        input_prompt: Optional[str] = None,
        vllm_url: str = "http://localhost:8000",
        model_name: str = "",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)
        self.limit = limit
        self.domains = domains
        self.input_prompt = input_prompt
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._samples: List[OpenThoughtsSample] = []

    def _load_samples(self) -> List[OpenThoughtsSample]:
        if not self._samples:
            self._samples = list(
                load_openthoughts_samples(
                    limit=self.limit,
                    domains=self.domains,
                )
            )
            self.logger.info(f"Loaded {len(self._samples)} OpenThoughts samples")
        return self._samples

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Generate reasoning responses for all samples."""
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing sample {sample.original_index} "
                f"(domain: {sample.domain})"
            )
            start_time = time.time()

            try:
                try:
                    from evals.telemetry.trace_collector import TraceCollector

                    collector = TraceCollector(
                        vllm_url=self.vllm_url, model_name=self.model_name
                    )
                    prompt = sample.get_prompt(self.input_prompt)
                    trace = collector.run_query_direct_vllm(
                        query_id=str(sample.original_index),
                        workload_type="reasoning",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "problem": sample.problem,
                        "ground_truth": sample.answer,
                        "model_answer": trace.response_text,
                        "domain": sample.domain,
                        "input_tokens": trace.total_input_tokens,
                        "output_tokens": trace.total_output_tokens,
                        "wall_clock_s": trace.total_wall_clock_s,
                        "error": None,
                        "trace": trace.to_dict(),
                    }
                except ImportError:
                    prompt = sample.get_prompt(self.input_prompt)
                    response = orchestrator.run(prompt)
                    elapsed = time.time() - start_time
                    response_text = (
                        str(getattr(response, "content", response))
                        if response
                        else ""
                    )
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "problem": sample.problem,
                        "ground_truth": sample.answer,
                        "model_answer": response_text,
                        "domain": sample.domain,
                        "wall_clock_s": elapsed,
                        "error": None,
                    }
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Error: {e}")
                results[str(sample.original_index)] = {
                    "original_index": sample.original_index,
                    "problem": sample.problem,
                    "ground_truth": sample.answer,
                    "model_answer": "",
                    "domain": sample.domain,
                    "wall_clock_s": elapsed,
                    "error": str(e),
                }

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Score responses using exact match."""
        metrics = score_openthoughts(results)
        self.logger.info(
            f"OpenThoughts: {metrics['exact_match_accuracy']}% exact match"
        )
        return metrics


@register_benchmark("openthoughts")
def _create_openthoughts_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> OpenThoughtsBenchmark:
    """Create an OpenThoughts benchmark instance."""
    options = options or {}
    return OpenThoughtsBenchmark(**options)
