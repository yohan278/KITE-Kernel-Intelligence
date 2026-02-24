"""Humanity's Last Exam (HLE) benchmark runner.

Runs evaluation of models on the HLE benchmark.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dataset import HLEDataset, HLESample
from ..registry import register_benchmark
from ...utils import extract_final_answer, print_result_box, print_task_header


@dataclass
class HLEResult:
    """Result for a single HLE evaluation."""

    question_id: str
    """Question identifier"""

    subject: str
    """Subject area"""

    correct: bool
    """Whether the answer was correct"""

    predicted_answer: str
    """Model's predicted answer"""

    expected_answer: str
    """Expected correct answer"""

    question_type: str
    """Type of question (multiple_choice or short_answer)"""

    latency_seconds: float
    """Time taken for this question"""

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
class HLEMetrics:
    """Aggregate metrics for HLE evaluation."""

    accuracy: float
    """Overall accuracy"""

    mc_accuracy: float
    """Multiple choice accuracy"""

    short_answer_accuracy: float
    """Short answer accuracy"""

    avg_latency: float
    """Average latency per question"""

    total_questions: int
    """Total number of questions evaluated"""

    correct_answers: int
    """Number of correct answers"""

    subject_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """Per-subject metrics"""


def judge_answer(predicted: str, expected: str, question: str, question_type: str) -> tuple[bool, str]:
    """Use GPT-5-mini as an LM judge to determine if the answer is correct.

    Returns:
        (is_correct, reasoning)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    type_hint = ""
    if question_type == "multiple_choice":
        type_hint = (
            "\nThis is a multiple-choice question. The expected answer is a letter (A, B, C, D, etc.). "
            "The predicted answer is correct if it selects the same option letter."
        )
    else:
        type_hint = (
            "\nThis is a short-answer question. The predicted answer is correct if it "
            "conveys the same meaning as the expected answer, even if phrased differently."
        )

    judge_prompt = f"""You are an evaluation judge. Determine if the predicted answer matches the expected answer for the given question.

Question: {question}

Expected answer: {expected}

Predicted answer: {predicted}
{type_hint}

Consider:
- The predicted answer may be phrased differently but still be correct.
- Minor formatting differences should not count as wrong if the core answer matches.
- For numeric answers, small rounding differences are acceptable.

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
    name="hle",
    description="Humanity's Last Exam - frontier academic benchmark",
    domains=["all", "math", "physics", "chemistry", "biology", "cs", "humanities"],
    default_domain="all",
    metrics=["accuracy", "mc_accuracy", "short_answer_accuracy"],
)
class HLERunner:
    """Runner for Humanity's Last Exam evaluation.

    Example:
        runner = HLERunner(limit=100, seed=42)
        results = runner.run(model_fn=my_model_inference)
        print(f"Accuracy: {results.accuracy:.2%}")
    """

    def __init__(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        verbose: bool = False,
        output_dir: Optional[str] = None,
        save_interval_minutes: int = 10,
    ):
        self.split = split
        self.limit = limit
        self.seed = seed
        self.subjects = subjects
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_interval_minutes = save_interval_minutes

        self.dataset = HLEDataset(
            split=split,
            limit=limit,
            seed=seed,
            subjects=subjects,
        )

    def _load_existing_results(self) -> tuple[list[HLEResult], set[str]]:
        """Load previously saved results for auto-resume."""
        if not self.output_dir:
            return [], set()
        results_file = self.output_dir / "hle_results.jsonl"
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
                completed_ids.add(data["question_id"])
                results.append(HLEResult(
                    question_id=data["question_id"],
                    subject=data.get("subject", ""),
                    correct=data["correct"],
                    predicted_answer=data["predicted_answer"],
                    expected_answer=data["expected_answer"],
                    question_type=data["question_type"],
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
    ) -> HLEMetrics:
        """Run evaluation on the benchmark.

        Args:
            model_fn: Model inference function (system_prompt, messages) -> response
            orchestrator: Whether orchestrator mode is enabled. Selects the appropriate system prompt.
        """
        # Auto-resume: load any previously saved results
        existing_results, completed_ids = self._load_existing_results()
        results: List[HLEResult] = list(existing_results)
        if completed_ids:
            print(f"  [Resume] Loaded {len(completed_ids)} existing results from {self.output_dir}")

        last_save_time = time.time()

        # Orchestrator mode builds its own system prompt internally;
        # non-orchestrator uses the model's default.
        system_prompt = ""

        total = len(self.dataset)
        for i, sample in enumerate(self.dataset):
            # Skip already-completed samples (auto-resume)
            if sample.question_id in completed_ids:
                continue

            print_task_header(
                index=i, total=total,
                task_id=sample.question_id,
                question=sample.get_prompt(),
                metadata=f"{sample.subject}, {sample.question_type}",
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
        sample: HLESample,
        model_fn: Callable,
        system_prompt: str,
    ) -> HLEResult:
        """Evaluate a single sample."""
        start_time = time.time()

        try:
            prompt = sample.get_prompt()
            messages = [{"role": "user", "content": prompt}]

            response = model_fn(system_prompt, messages)
            latency = time.time() - start_time

            predicted = extract_final_answer(response)
            is_correct, judge_reasoning = judge_answer(
                predicted, sample.answer, sample.question, sample.question_type
            )

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
                for entry in orch_result.conversation:
                    if entry.get("role") == "tool":
                        content = entry.get("content", "")
                        if content.startswith("Error"):
                            tools_failed += 1
                        else:
                            tools_successful += 1

            return HLEResult(
                question_id=sample.question_id,
                subject=sample.subject,
                correct=is_correct,
                predicted_answer=predicted,
                expected_answer=sample.answer,
                question_type=sample.question_type,
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
            return HLEResult(
                question_id=sample.question_id,
                subject=sample.subject,
                correct=False,
                predicted_answer="",
                expected_answer=sample.answer,
                question_type=sample.question_type,
                latency_seconds=time.time() - start_time,
                error=str(e),
            )

    def _print_result(self, result: HLEResult, sample: HLESample):
        """Print result using shared formatting."""
        status = "✅ CORRECT" if result.correct else "❌ INCORRECT"
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

    def _compute_metrics(self, results: List[HLEResult]) -> HLEMetrics:
        """Compute aggregate metrics from results."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)

        mc_results = [r for r in results if r.question_type == "multiple_choice"]
        sa_results = [r for r in results if r.question_type == "short_answer"]

        mc_correct = sum(1 for r in mc_results if r.correct)
        sa_correct = sum(1 for r in sa_results if r.correct)

        accuracy = correct / total if total > 0 else 0.0
        mc_accuracy = mc_correct / len(mc_results) if mc_results else 0.0
        sa_accuracy = sa_correct / len(sa_results) if sa_results else 0.0
        avg_latency = sum(r.latency_seconds for r in results) / total if total else 0.0

        # Per-subject metrics
        subject_metrics = {}
        subjects = set(r.subject for r in results)
        for subject in subjects:
            subject_results = [r for r in results if r.subject == subject]
            subject_total = len(subject_results)
            subject_metrics[subject] = {
                "accuracy": sum(1 for r in subject_results if r.correct) / subject_total,
                "total": subject_total,
            }

        return HLEMetrics(
            accuracy=accuracy,
            mc_accuracy=mc_accuracy,
            short_answer_accuracy=sa_accuracy,
            avg_latency=avg_latency,
            total_questions=total,
            correct_answers=correct,
            subject_metrics=subject_metrics,
        )

    def _save_results(self, results: List[HLEResult], metrics: HLEMetrics):
        """Save results with full metadata to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_file = self.output_dir / "hle_results.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps({
                    "question_id": result.question_id,
                    "subject": result.subject,
                    "correct": result.correct,
                    "predicted_answer": result.predicted_answer,
                    "expected_answer": result.expected_answer,
                    "question_type": result.question_type,
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

        metrics_file = self.output_dir / "hle_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "accuracy": metrics.accuracy,
                "mc_accuracy": metrics.mc_accuracy,
                "short_answer_accuracy": metrics.short_answer_accuracy,
                "avg_latency": metrics.avg_latency,
                "total_questions": metrics.total_questions,
                "correct_answers": metrics.correct_answers,
                "subject_metrics": metrics.subject_metrics,
            }, f, indent=2)

        print(f"Results saved to {self.output_dir}")
