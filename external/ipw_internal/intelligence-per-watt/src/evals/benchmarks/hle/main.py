"""HLE (Humanity's Last Exam) benchmark.

Evaluates orchestrator performance on challenging multi-step reasoning tasks
with telemetry tracking for energy, cost, and latency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from .dataset import HLESample, load_hle_dataset


class HLEBenchmark(DatasetBenchmark):
    """Humanity's Last Exam benchmark with energy/cost tracking.

    Evaluates orchestrator on HLE tasks and tracks:
    - Accuracy (correctness of answers)
    - Energy consumption (joules)
    - Cost (USD)
    - Latency (seconds)
    - Power usage (watts)
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        category_filter: Optional[str] = None,
        dataset_path: Optional[str] = None,
        grading_method: str = "exact_match",
        logger=None,
        system_instruction: Optional[str] = None,
    ):
        """Initialize HLE benchmark.

        Args:
            split: Dataset split ("train", "validation", "test")
            limit: Maximum samples to evaluate
            category_filter: Filter by category
            dataset_path: Custom dataset path
            grading_method: How to grade answers ("exact_match", "contains", "fuzzy")
            logger: Logger instance
            system_instruction: System instruction for orchestrator
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.split = split
        self.limit = limit
        self.category_filter = category_filter
        self.dataset_path = dataset_path
        self.grading_method = grading_method

        # Load dataset
        self.samples = load_hle_dataset(
            split=split,
            limit=limit,
            category_filter=category_filter,
            dataset_path=dataset_path
        )
        self.logger.info(f"Loaded {len(self.samples)} HLE samples from {split} split")

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Generate responses using orchestrator.

        Args:
            orchestrator: Orchestrator with run() method

        Returns:
            Dictionary with results for each sample
        """
        results = []

        for idx, sample in enumerate(self.samples):
            self.logger.info(f"Processing sample {idx+1}/{len(self.samples)}: {sample.task_id}")

            # Run orchestrator
            start_time = time.time()
            try:
                response = orchestrator.run(sample.question)

                # Extract content (handle different response types)
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, dict):
                    content = response.get('content', str(response))
                else:
                    content = str(response)

                # Ensure content is a string (handle list responses from agents like OpenHands)
                if isinstance(content, list):
                    # Join list elements if they're strings, or extract content from message dicts
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and 'content' in item:
                            parts.append(str(item['content']))
                        else:
                            parts.append(str(item))
                    content = '\n'.join(parts)
                elif not isinstance(content, str):
                    content = str(content)

                # Extract telemetry if available
                telemetry = {}
                if hasattr(orchestrator, 'get_metadata'):
                    metadata = orchestrator.get_metadata()
                    telemetry = {
                        'trajectory': metadata.get('trajectory', []),
                        'total_cost_usd': metadata.get('total_cost_usd', 0.0),
                        'total_energy_joules': metadata.get('total_energy_joules', 0.0),
                        'tool_usage': metadata.get('tool_usage', {}),
                        'num_turns': metadata.get('num_turns', 0),
                        'total_forward_passes': metadata.get('total_forward_passes', 0),
                    }

                # Try to extract metrics from response
                if hasattr(response, 'usage'):
                    telemetry['total_tokens'] = response.usage.total_tokens
                    telemetry['prompt_tokens'] = response.usage.prompt_tokens
                    telemetry['completion_tokens'] = response.usage.completion_tokens

                if hasattr(response, 'time_to_first_token_ms'):
                    telemetry['ttft_ms'] = response.time_to_first_token_ms

                error = None

            except Exception as e:
                self.logger.error(f"Error on sample {sample.task_id}: {e}")
                content = ""
                telemetry = {}
                error = str(e)

            end_time = time.time()
            latency = end_time - start_time

            results.append({
                'task_id': sample.task_id,
                'question': sample.question,
                'gold_answer': sample.answer,
                'response': content,
                'category': sample.category,
                'difficulty': sample.difficulty,
                'latency_seconds': latency,
                'telemetry': telemetry,
                'error': error,
            })

        return {'results': results}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate responses and compute metrics.

        Args:
            results: Dictionary from generate_responses()

        Returns:
            Dictionary of metrics (accuracy, avg_energy, avg_cost, etc.)
        """
        results_list = results['results']

        if not results_list:
            self.logger.warning("No results to evaluate")
            return {
                'accuracy': 0.0,
                'num_samples': 0,
            }

        # Compute correctness
        correct = 0
        total_energy = 0.0
        total_cost = 0.0
        total_latency = 0.0
        total_tokens = 0
        total_turns = 0
        total_forward_passes = 0

        for result in results_list:
            # Grade answer
            if result['error']:
                # Error = incorrect
                is_correct = False
            else:
                is_correct = self._grade_answer(
                    result['response'],
                    result['gold_answer'],
                    method=self.grading_method
                )

            if is_correct:
                correct += 1

            # Aggregate telemetry
            telemetry = result.get('telemetry', {})
            total_cost += telemetry.get('total_cost_usd', 0.0)
            total_energy += telemetry.get('total_energy_joules', 0.0)
            total_latency += result.get('latency_seconds', 0.0)
            total_tokens += telemetry.get('total_tokens', 0)
            total_turns += telemetry.get('num_turns', 0)
            total_forward_passes += telemetry.get('total_forward_passes', 0)

        num_samples = len(results_list)
        accuracy = correct / num_samples if num_samples > 0 else 0.0

        # Compute IPJ and other efficiency metrics
        ipj = accuracy / total_energy if total_energy > 0 else 0.0
        intelligence_per_dollar = accuracy / total_cost if total_cost > 0 else float('inf')
        intelligence_per_second = accuracy / total_latency if total_latency > 0 else float('inf')

        metrics = {
            'accuracy': accuracy,
            'num_samples': num_samples,
            'num_correct': correct,
            # Cost metrics
            'avg_cost_usd': total_cost / num_samples if num_samples > 0 else 0.0,
            'total_cost_usd': total_cost,
            # Energy metrics
            'avg_energy_joules': total_energy / num_samples if num_samples > 0 else 0.0,
            'total_energy_joules': total_energy,
            # Latency metrics
            'avg_latency_seconds': total_latency / num_samples if num_samples > 0 else 0.0,
            'total_latency_seconds': total_latency,
            # Other metrics
            'avg_tokens': total_tokens / num_samples if num_samples > 0 else 0.0,
            'avg_turns': total_turns / num_samples if num_samples > 0 else 0.0,
            'total_forward_passes': total_forward_passes,
            'avg_forward_passes': total_forward_passes / num_samples if num_samples > 0 else 0.0,
            # Efficiency metrics (IPJ focus)
            'ipj': ipj,
            'intelligence_per_dollar': intelligence_per_dollar,
            'intelligence_per_second': intelligence_per_second,
        }

        # Log summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"HLE Benchmark Results ({self.split} split)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{num_samples})")
        self.logger.info(f"")
        self.logger.info(f"Efficiency Metrics (IPJ Focus):")
        self.logger.info(f"  IPJ (Intelligence/Joule): {ipj:.6f}")
        self.logger.info(f"  Intelligence/Dollar: {intelligence_per_dollar:.4f}")
        self.logger.info(f"  Intelligence/Second: {intelligence_per_second:.4f}")
        self.logger.info(f"")
        self.logger.info(f"Resource Usage:")
        self.logger.info(f"  Avg Energy: {metrics['avg_energy_joules']:.2f}J")
        self.logger.info(f"  Avg Cost: ${metrics['avg_cost_usd']:.4f}")
        self.logger.info(f"  Avg Latency: {metrics['avg_latency_seconds']:.2f}s")
        self.logger.info(f"  Avg Forward Passes: {metrics['avg_forward_passes']:.1f}")
        self.logger.info(f"  Avg Turns: {metrics['avg_turns']:.1f}")
        self.logger.info(f"")
        self.logger.info(f"Totals:")
        self.logger.info(f"  Total Energy: {total_energy:.2f}J")
        self.logger.info(f"  Total Cost: ${total_cost:.4f}")
        self.logger.info(f"{'='*60}")

        return metrics

    def _grade_answer(
        self,
        predicted: str,
        gold: str,
        method: str = "exact_match"
    ) -> bool:
        """Grade predicted answer against gold answer.

        Args:
            predicted: Predicted answer
            gold: Gold answer
            method: Grading method

        Returns:
            True if correct, False otherwise
        """
        pred = predicted.strip().lower()
        gold_clean = gold.strip().lower()

        if method == "exact_match":
            return pred == gold_clean

        elif method == "contains":
            return gold_clean in pred

        elif method == "fuzzy":
            # Simple fuzzy match (can be improved with edit distance)
            # Check if key words from gold appear in prediction
            gold_words = set(gold_clean.split())
            pred_words = set(pred.split())
            overlap = len(gold_words & pred_words) / len(gold_words) if gold_words else 0
            return overlap > 0.5

        else:
            raise ValueError(f"Unknown grading method: {method}")


@register_benchmark("hle")
def _create_hle_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> HLEBenchmark:
    """Factory function for HLE benchmark.

    Args:
        options: Benchmark configuration options

    Returns:
        HLEBenchmark instance
    """
    options = options or {}
    return HLEBenchmark(**options)


__all__ = [
    "HLEBenchmark",
]
