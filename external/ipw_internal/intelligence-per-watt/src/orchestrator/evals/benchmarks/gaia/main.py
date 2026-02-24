"""GAIA benchmark runner.

Runs evaluation of models on the GAIA benchmark for General AI Assistants.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dataset import GAIADataset, GAIASample
from ..registry import register_benchmark
from ...utils import extract_final_answer, print_result_box, print_task_header


@dataclass
class GAIAResult:
    """Result for a single GAIA evaluation."""

    task_id: str
    """Task identifier"""

    level: int
    """Difficulty level"""

    correct: bool
    """Whether the answer was correct"""

    predicted_answer: str
    """Model's predicted answer"""

    expected_answer: str
    """Expected correct answer"""

    latency_seconds: float
    """Time taken for this task"""

    error: Optional[str] = None
    """Error message if evaluation failed"""

    model_response: Optional[str] = None
    """Full model response for debugging"""

    # Metadata from orchestrator
    num_turns: int = 0
    """Number of orchestrator turns"""

    tools_used: List[str] = field(default_factory=list)
    """Tools that were called"""

    tools_successful: int = 0
    """Number of successful tool calls"""

    tools_failed: int = 0
    """Number of failed tool calls"""

    conversation: List[Dict[str, str]] = field(default_factory=list)
    """Full conversation history"""

    raw_responses: List[str] = field(default_factory=list)
    """Raw model responses per turn"""

    judge_reasoning: Optional[str] = None
    """LM judge reasoning for the correctness verdict"""


@dataclass
class GAIAMetrics:
    """Aggregate metrics for GAIA evaluation."""

    accuracy: float
    """Overall accuracy"""

    level1_accuracy: float
    """Level 1 accuracy"""

    level2_accuracy: float
    """Level 2 accuracy"""

    level3_accuracy: float
    """Level 3 accuracy"""

    avg_latency: float
    """Average latency per task"""

    total_tasks: int
    """Total number of tasks evaluated"""

    correct_tasks: int
    """Number of correct tasks"""

    level_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    """Per-level metrics"""


def judge_answer(predicted: str, expected: str, question: str) -> tuple[bool, str]:
    """Use GPT-5-mini as an LM judge to determine if the answer is correct.

    Returns:
        (is_correct, reasoning)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    judge_prompt = f"""You are an evaluation judge. Determine if the predicted answer matches the expected answer for the given question.

Question: {question}

Expected answer: {expected}

Predicted answer: {predicted}

Consider:
- The predicted answer may be phrased differently but still be correct.
- Minor formatting differences (e.g., "Paris" vs "paris", "3.14" vs "3.14159") should not count as wrong if the core answer matches.
- For numeric answers, small rounding differences are acceptable.
- If the predicted answer contains the correct answer wrapped in extra explanation or phrasing (e.g., "The answer is Paris" for expected "Paris"), it should count as correct.
- However, if the question asks for a specific list, set, or subset, the predicted answer must contain exactly the right items — no more, no less. A superset or subset of the expected items is INCORRECT. For example, if the expected answer is "b, e" then "a, b, c, d, e" is INCORRECT because it includes extra items.

Respond in this exact format:
VERDICT: <CORRECT or INCORRECT>"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": judge_prompt}],
            max_completion_tokens=512,
        )
        judge_response = response.choices[0].message.content.strip()

        # Parse verdict
        is_correct = False
        if "VERDICT:" in judge_response:
            verdict = judge_response.split("VERDICT:")[-1].strip().upper()
            is_correct = "CORRECT" in verdict and "INCORRECT" not in verdict

        return is_correct, ""

    except Exception as e:
        # Do NOT fallback to string matching — it is unreliable.
        # If the LM judge is unavailable, mark as incorrect with the error.
        return False, ""


@register_benchmark(
    name="gaia",
    description="General AI Assistants benchmark",
    domains=["all", "level1", "level2", "level3"],
    default_domain="all",
    metrics=["accuracy", "level1_accuracy", "level2_accuracy", "level3_accuracy"],
)
class GAIARunner:
    """Runner for GAIA evaluation.

    Example:
        runner = GAIARunner(limit=100, seed=42)
        results = runner.run(model_fn=my_model_inference)
        print(f"Accuracy: {results.accuracy:.2%}")
    """

    def __init__(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        levels: Optional[List[int]] = None,
        domain: str = "all",
        verbose: bool = False,
        output_dir: Optional[str] = None,
        save_interval_minutes: int = 10,
    ):
        # GAIA test split has no ground-truth answers; default to validation
        self.split = "validation" if split == "test" else split
        self.limit = limit
        self.seed = seed
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_interval_minutes = save_interval_minutes

        # Convert domain to levels
        if domain == "level1":
            levels = [1]
        elif domain == "level2":
            levels = [2]
        elif domain == "level3":
            levels = [3]

        self.dataset = GAIADataset(
            split=self.split,
            limit=limit,
            seed=seed,
            levels=levels,
        )

    def _load_existing_results(self) -> tuple[list[GAIAResult], set[str]]:
        """Load previously saved results for auto-resume."""
        if not self.output_dir:
            return [], set()
        results_file = self.output_dir / "gaia_results.jsonl"
        if not results_file.exists():
            return [], set()

        results = []
        completed_ids = set()
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed_ids.add(data["task_id"])
                results.append(GAIAResult(
                    task_id=data["task_id"],
                    level=data["level"],
                    correct=data["correct"],
                    predicted_answer=data["predicted_answer"],
                    expected_answer=data["expected_answer"],
                    latency_seconds=data["latency_seconds"],
                    error=data.get("error"),
                    model_response=data.get("model_response"),
                    num_turns=data.get("num_turns", 0),
                    tools_used=data.get("tools_used", []),
                    tools_successful=data.get("tools_successful", 0),
                    tools_failed=data.get("tools_failed", 0),
                    conversation=data.get("conversation", []),
                    raw_responses=data.get("raw_responses", []),
                    judge_reasoning=data.get("judge_reasoning"),
                ))
        return results, completed_ids

    def run(
        self,
        model_fn: Callable[[str, List[Dict]], str],
        orchestrator: bool = True,
    ) -> GAIAMetrics:
        """Run evaluation on the benchmark.

        Args:
            model_fn: Model inference function (system_prompt, messages) -> response
            orchestrator: Whether orchestrator mode is enabled. Selects the appropriate system prompt.
        """
        # Auto-resume: load any previously saved results
        existing_results, completed_ids = self._load_existing_results()
        results: List[GAIAResult] = list(existing_results)
        if completed_ids:
            print(f"  [Resume] Loaded {len(completed_ids)} existing results from {self.output_dir}")

        last_save_time = time.time()

        # Orchestrator mode builds its own system prompt internally;
        # non-orchestrator uses the model's default.
        system_prompt = ""

        total = len(self.dataset)
        for i, sample in enumerate(self.dataset):
            # Skip already-completed samples (auto-resume)
            if sample.task_id in completed_ids:
                continue

            print_task_header(
                index=i, total=total,
                task_id=sample.task_id,
                question=sample.get_prompt(),
                metadata=f"Level {sample.level}",
                verbose=self.verbose,
            )

            result = self._evaluate_sample(sample, model_fn, system_prompt)
            results.append(result)

            self._print_result(result, sample)

            # Periodic intermediate save
            if self.output_dir and (time.time() - last_save_time) >= self.save_interval_minutes * 60:
                self._save_results(results, self._compute_metrics(results))
                print(f"  [Auto-saved {len(results)} results to {self.output_dir}]")
                last_save_time = time.time()

        metrics = self._compute_metrics(results)

        if self.output_dir:
            self._save_results(results, metrics)

        return metrics

    def _evaluate_sample(
        self,
        sample: GAIASample,
        model_fn: Callable,
        system_prompt: str,
    ) -> GAIAResult:
        """Evaluate a single sample."""
        start_time = time.time()

        try:
            prompt = sample.get_prompt()
            messages = [{"role": "user", "content": prompt}]

            response = model_fn(system_prompt, messages)
            latency = time.time() - start_time

            predicted = extract_final_answer(response)
            is_correct, judge_reasoning = judge_answer(predicted, sample.answer, sample.question)

            # Extract orchestrator metadata if available
            num_turns = 0
            tools_used = []
            tools_successful = 0
            tools_failed = 0
            conversation = []
            raw_responses = []

            if hasattr(model_fn, "last_result") and model_fn.last_result is not None:
                orch_result = model_fn.last_result
                num_turns = orch_result.num_turns
                tools_used = orch_result.tools_used
                conversation = orch_result.conversation
                raw_responses = orch_result.raw_responses
                # Count successful vs failed tool calls
                for entry in orch_result.conversation:
                    if entry.get("role") == "tool":
                        content = entry.get("content", "")
                        if content.startswith("Error"):
                            tools_failed += 1
                        else:
                            tools_successful += 1

            return GAIAResult(
                task_id=sample.task_id,
                level=sample.level,
                correct=is_correct,
                predicted_answer=predicted,
                expected_answer=sample.answer,
                latency_seconds=latency,
                model_response=response,
                num_turns=num_turns,
                tools_used=tools_used,
                tools_successful=tools_successful,
                tools_failed=tools_failed,
                conversation=conversation,
                raw_responses=raw_responses,
                judge_reasoning=judge_reasoning,
            )

        except Exception as e:
            return GAIAResult(
                task_id=sample.task_id,
                level=sample.level,
                correct=False,
                predicted_answer="",
                expected_answer=sample.answer,
                latency_seconds=time.time() - start_time,
                error=str(e),
            )

    def _print_result(self, result: GAIAResult, sample: GAIASample):
        """Print result using shared formatting."""
        status = "✅ CORRECT" if result.correct else "❌ WRONG"
        print_result_box(
            status=status,
            latency_seconds=result.latency_seconds,
            num_turns=result.num_turns,
            tools_used=result.tools_used,
            tools_successful=result.tools_successful,
            tools_failed=result.tools_failed,
            predicted=result.predicted_answer,
            expected=result.expected_answer,
            model_response=result.model_response,
            raw_responses=result.raw_responses,
            verbose=self.verbose,
        )

    def _compute_metrics(self, results: List[GAIAResult]) -> GAIAMetrics:
        """Compute aggregate metrics from results."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)

        level_results = {1: [], 2: [], 3: []}
        for r in results:
            level = int(r.level) if isinstance(r.level, str) else r.level
            if level in level_results:
                level_results[level].append(r)

        def level_accuracy(level: int) -> float:
            level_r = level_results.get(level, [])
            if not level_r:
                return 0.0
            return sum(1 for r in level_r if r.correct) / len(level_r)

        accuracy = correct / total if total > 0 else 0.0
        avg_latency = sum(r.latency_seconds for r in results) / total if total else 0.0

        level_metrics = {}
        for level in [1, 2, 3]:
            level_r = level_results.get(level, [])
            if level_r:
                level_metrics[level] = {
                    "accuracy": level_accuracy(level),
                    "total": len(level_r),
                    "correct": sum(1 for r in level_r if r.correct),
                }

        return GAIAMetrics(
            accuracy=accuracy,
            level1_accuracy=level_accuracy(1),
            level2_accuracy=level_accuracy(2),
            level3_accuracy=level_accuracy(3),
            avg_latency=avg_latency,
            total_tasks=total,
            correct_tasks=correct,
            level_metrics=level_metrics,
        )

    def _save_results(self, results: List[GAIAResult], metrics: GAIAMetrics):
        """Save results with full metadata to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_file = self.output_dir / "gaia_results.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps({
                    "task_id": result.task_id,
                    "level": result.level,
                    "correct": result.correct,
                    "predicted_answer": result.predicted_answer,
                    "expected_answer": result.expected_answer,
                    "latency_seconds": result.latency_seconds,
                    "error": result.error,
                    "model_response": result.model_response,
                    "num_turns": result.num_turns,
                    "tools_used": result.tools_used,
                    "tools_successful": result.tools_successful,
                    "tools_failed": result.tools_failed,
                    "conversation": result.conversation,
                    "raw_responses": result.raw_responses,
                    "judge_reasoning": result.judge_reasoning,
                }) + "\n")

        metrics_file = self.output_dir / "gaia_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "accuracy": metrics.accuracy,
                "level1_accuracy": metrics.level1_accuracy,
                "level2_accuracy": metrics.level2_accuracy,
                "level3_accuracy": metrics.level3_accuracy,
                "avg_latency": metrics.avg_latency,
                "total_tasks": metrics.total_tasks,
                "correct_tasks": metrics.correct_tasks,
                "level_metrics": {str(k): v for k, v in metrics.level_metrics.items()},
            }, f, indent=2)

        print(f"Results saved to {self.output_dir}")
