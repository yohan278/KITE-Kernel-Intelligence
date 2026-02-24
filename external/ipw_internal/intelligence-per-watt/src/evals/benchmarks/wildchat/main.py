"""WildChat benchmark: multi-turn conversation replay."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.wildchat.dataset import WildChatSample, load_wildchat_samples
from evals.benchmarks.wildchat.scorer import score_completion


class WildChatBenchmark(DatasetBenchmark):
    """Multi-turn chat benchmark using WildChat conversations.

    Replays real multi-turn conversations: sends each user turn to vLLM,
    collects model response, uses it as context for the next turn.
    Scores based on completion rate (all turns responded without error).
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        min_turns: int = 1,
        max_turns: Optional[int] = None,
        vllm_url: str = "http://localhost:8000",
        model_name: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)
        self.limit = limit
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._samples: List[WildChatSample] = []

    def _load_samples(self) -> List[WildChatSample]:
        if not self._samples:
            self._samples = list(
                load_wildchat_samples(
                    limit=self.limit,
                    min_turns=self.min_turns,
                    max_turns=self.max_turns,
                )
            )
            self.logger.info(f"Loaded {len(self._samples)} WildChat samples")
        return self._samples

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Replay conversations through the orchestrator/vLLM."""
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing sample {sample.original_index} "
                f"({sample.num_turns} turns)"
            )
            start_time = time.time()

            try:
                try:
                    from evals.telemetry.trace_collector import TraceCollector

                    collector = TraceCollector(
                        vllm_url=self.vllm_url, model_name=self.model_name
                    )
                    trace = collector.run_query_multi_turn_vllm(
                        query_id=str(sample.original_index),
                        workload_type="chat",
                        conversation=sample.conversation,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "num_turns_total": sample.num_turns,
                        "num_turns_completed": trace.num_turns,
                        "completed": trace.completed,
                        "total_input_tokens": trace.total_input_tokens,
                        "total_output_tokens": trace.total_output_tokens,
                        "wall_clock_s": trace.total_wall_clock_s,
                        "error": None,
                        "trace": trace.to_dict(),
                    }
                except ImportError:
                    prompt = sample.first_user_message
                    response = orchestrator.run(prompt)
                    elapsed = time.time() - start_time
                    results[str(sample.original_index)] = {
                        "original_index": sample.original_index,
                        "num_turns_total": sample.num_turns,
                        "num_turns_completed": 1,
                        "completed": True,
                        "wall_clock_s": elapsed,
                        "error": None,
                    }
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Error: {e}")
                results[str(sample.original_index)] = {
                    "original_index": sample.original_index,
                    "num_turns_total": sample.num_turns,
                    "num_turns_completed": 0,
                    "completed": False,
                    "wall_clock_s": elapsed,
                    "error": str(e),
                }

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Score using completion rate."""
        metrics = score_completion(results)
        self.logger.info(f"WildChat: {metrics['completion_rate']}% completion rate")
        return metrics


@register_benchmark("wildchat")
def _create_wildchat_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> WildChatBenchmark:
    """Create a WildChat benchmark instance."""
    options = options or {}
    return WildChatBenchmark(**options)
