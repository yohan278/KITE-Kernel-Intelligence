"""AgentData benchmark: agentic task completion."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.agentdata.dataset import AgentDataSample, load_agentdata_samples
from evals.benchmarks.agentdata.scorer import score_agentdata, score_task_completion


class AgentDataBenchmark(DatasetBenchmark):
    """Agentic task completion benchmark using AgentData trajectories.

    Extracts task descriptions from agent trajectories and runs them
    through an agent or model. Scores using task completion heuristics
    or LLM-as-judge.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        domains: Optional[List[str]] = None,
        vllm_url: str = "http://localhost:8000",
        model_name: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)
        self.limit = limit
        self.domains = domains
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._samples: List[AgentDataSample] = []

    def _load_samples(self) -> List[AgentDataSample]:
        if not self._samples:
            self._samples = list(
                load_agentdata_samples(
                    limit=self.limit,
                    domains=self.domains,
                )
            )
            self.logger.info(f"Loaded {len(self._samples)} AgentData samples")
        return self._samples

    def _build_prompt(self, sample: AgentDataSample) -> str:
        return (
            "You are a helpful AI agent. Complete the following task.\n\n"
            f"Task: {sample.task}\n\n"
            "Provide your response:"
        )

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Generate responses for all AgentData samples."""
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing {sample.domain}: "
                f"{sample.task[:60]}..."
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
                        workload_type="agentic",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    grade = score_task_completion(sample.task, trace.response_text)
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "task": sample.task,
                        "model_answer": trace.response_text,
                        "domain": sample.domain,
                        "grade": grade,
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
                    grade = score_task_completion(sample.task, response_text)
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "task": sample.task,
                        "model_answer": response_text,
                        "domain": sample.domain,
                        "grade": grade,
                        "wall_clock_s": elapsed,
                        "error": None,
                    }
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Error: {e}")
                results[str(sample.original_index)] = {
                    "original_index": sample.original_index,
                    "task": sample.task,
                    "model_answer": "",
                    "domain": sample.domain,
                    "grade": "FAILED",
                    "wall_clock_s": elapsed,
                    "error": str(e),
                }

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Score task completion."""
        metrics = score_agentdata(results)
        self.logger.info(
            f"AgentData: {metrics['completion_rate']}% completion, "
            f"{metrics['success_rate']}% success"
        )
        return metrics


@register_benchmark("agentdata")
def _create_agentdata_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> AgentDataBenchmark:
    options = options or {}
    return AgentDataBenchmark(**options)
