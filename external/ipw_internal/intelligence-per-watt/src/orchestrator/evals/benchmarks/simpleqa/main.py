"""SimpleQA Verified benchmark runner.

Runs evaluation on the SimpleQA Verified benchmark for short-form factuality.
Uses a 3-class grading scheme: CORRECT, INCORRECT, NOT_ATTEMPTED.

Grading prompt and metrics follow the official OpenAI implementation exactly:
https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dataset import SimpleQADataset, SimpleQASample
from ..registry import register_benchmark
from ...utils import extract_final_answer, print_result_box, print_task_header


@dataclass
class SimpleQAResult:
    """Result for a single SimpleQA evaluation."""

    original_index: int
    """Original index from SimpleQA"""

    topic: str
    """Topic category"""

    answer_type: str
    """Answer type"""

    grade: str
    """Grade: CORRECT, INCORRECT, or NOT_ATTEMPTED"""

    correct: bool
    """Whether the answer was correct"""

    predicted_answer: str
    """Model's predicted answer"""

    expected_answer: str
    """Expected correct answer"""

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


@dataclass
class SimpleQAMetrics:
    """Aggregate metrics for SimpleQA Verified evaluation."""

    accuracy: float
    """Overall accuracy (fraction correct)"""

    f1: float
    """F1 score = 2 * accuracy_given_attempted * correct_rate / (accuracy_given_attempted + correct_rate)"""

    correct_rate: float
    """Fraction graded CORRECT"""

    incorrect_rate: float
    """Fraction graded INCORRECT"""

    not_attempted_rate: float
    """Fraction graded NOT_ATTEMPTED"""

    avg_latency: float
    """Average latency per question"""

    total_questions: int
    """Total number of questions evaluated"""

    correct_count: int
    """Number of correct answers"""

    incorrect_count: int
    """Number of incorrect answers"""

    not_attempted_count: int
    """Number of not attempted answers"""

    topic_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-topic metrics"""

    answer_type_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-answer-type metrics"""


# Exact grader prompt from OpenAI's official SimpleQA implementation:
# https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - A predicted answer that is more specific than the gold target should be considered CORRECT, as long as it is consistent.
    - For example, if the gold target is "Table Mountain, South Africa" and the predicted answer is "Table Mountain in Cape Town", this is CORRECT because Cape Town is in South Africa.
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


def grade_answer(predicted: str, expected: str, question: str) -> str:
    """Use an LM judge to grade the answer using OpenAI's official SimpleQA grader prompt.

    Follows the exact grading protocol from:
    https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py

    Returns:
        grade: one of CORRECT/INCORRECT/NOT_ATTEMPTED
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        target=expected,
        predicted_answer=predicted,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": grader_prompt}],
            max_completion_tokens=64,
        )
        grading_response = response.choices[0].message.content.strip()

        # Parse letter grade (A/B/C) — matches OpenAI's parsing exactly
        match = re.search(r"(A|B|C)", grading_response)
        grade_letter = match.group(0) if match else "C"  # Default to NOT_ATTEMPTED
        return CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")

    except Exception as e:
        return "NOT_ATTEMPTED"


@register_benchmark(
    name="simpleqa",
    description="SimpleQA Verified - factuality benchmark for parametric knowledge",
    domains=["all", "Politics", "Art", "Music", "Science and technology", "Sports",
             "Geography", "History", "TV shows", "Video games"],
    default_domain="all",
    metrics=["accuracy", "f1", "accuracy_given_attempted", "correct_rate", "incorrect_rate", "not_attempted_rate"],
)
class SimpleQARunner:
    """Runner for SimpleQA Verified evaluation.

    Example:
        runner = SimpleQARunner(limit=100, seed=42)
        results = runner.run(model_fn=my_model_inference)
        print(f"F1: {results.f1:.2%}")
    """

    def __init__(
        self,
        split: str = "eval",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        domain: str = "all",
        verbose: bool = False,
        output_dir: Optional[str] = None,
        save_interval_minutes: int = 10,
    ):
        self.split = split
        self.limit = limit
        self.seed = seed
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_interval_minutes = save_interval_minutes

        # Convert domain to topic filter
        topics = None
        if domain != "all":
            topics = [domain]

        self.dataset = SimpleQADataset(
            split=split,
            limit=limit,
            seed=seed,
            topics=topics,
        )

    def _load_existing_results(self) -> tuple[list[SimpleQAResult], set[int]]:
        """Load previously saved results for auto-resume."""
        if not self.output_dir:
            return [], set()
        results_file = self.output_dir / "simpleqa_results.jsonl"
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
                completed_ids.add(data["original_index"])
                results.append(SimpleQAResult(
                    original_index=data["original_index"],
                    topic=data.get("topic", ""),
                    answer_type=data.get("answer_type", ""),
                    grade=data["grade"],
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
                ))
        return results, completed_ids

    def run(
        self,
        model_fn: Callable[[str, List[Dict]], str],
        orchestrator: bool = True,
    ) -> SimpleQAMetrics:
        """Run evaluation on the benchmark.

        Args:
            model_fn: Model inference function (system_prompt, messages) -> response
            orchestrator: Whether orchestrator mode is enabled. Selects the appropriate system prompt.
        """
        # Auto-resume: load any previously saved results
        existing_results, completed_ids = self._load_existing_results()
        results: List[SimpleQAResult] = list(existing_results)
        if completed_ids:
            print(f"  [Resume] Loaded {len(completed_ids)} existing results from {self.output_dir}")

        last_save_time = time.time()

        # Orchestrator mode builds its own system prompt internally;
        # non-orchestrator uses the model's default.
        system_prompt = ""

        total = len(self.dataset)
        for i, sample in enumerate(self.dataset):
            # Skip already-completed samples (auto-resume)
            if sample.original_index in completed_ids:
                continue

            print_task_header(
                index=i, total=total,
                task_id=f"Q{sample.original_index}",
                question=sample.get_prompt(),
                metadata=sample.topic,
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
        sample: SimpleQASample,
        model_fn: Callable,
        system_prompt: str,
    ) -> SimpleQAResult:
        """Evaluate a single sample."""
        start_time = time.time()

        try:
            prompt = sample.get_prompt()
            messages = [{"role": "user", "content": prompt}]

            response = model_fn(system_prompt, messages)
            latency = time.time() - start_time

            predicted = extract_final_answer(response)
            grade = grade_answer(predicted, sample.answer, sample.problem)

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

            return SimpleQAResult(
                original_index=sample.original_index,
                topic=sample.topic,
                answer_type=sample.answer_type,
                grade=grade,
                correct=grade == "CORRECT",
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
            )

        except Exception as e:
            return SimpleQAResult(
                original_index=sample.original_index,
                topic=sample.topic,
                answer_type=sample.answer_type,
                grade="NOT_ATTEMPTED",
                correct=False,
                predicted_answer="",
                expected_answer=sample.answer,
                latency_seconds=time.time() - start_time,
                error=str(e),
            )

    def _print_result(self, result: SimpleQAResult, sample: SimpleQASample):
        """Print result using shared formatting."""
        grade_icon = {"CORRECT": "✅", "INCORRECT": "❌", "NOT_ATTEMPTED": "⏭️"}.get(result.grade, "?")
        status = f"{grade_icon} {result.grade}"
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

    def _compute_metrics(self, results: List[SimpleQAResult]) -> SimpleQAMetrics:
        """Compute aggregate metrics from results.

        Follows the exact metric computation from OpenAI's official SimpleQA:
        https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py

        - is_given_attempted = correct_rate + incorrect_rate
        - accuracy_given_attempted = correct_rate / is_given_attempted
        - f1 = 2 * accuracy_given_attempted * correct_rate / (accuracy_given_attempted + correct_rate)
        """
        total = len(results)
        if total == 0:
            return SimpleQAMetrics(
                accuracy=0.0, f1=0.0, correct_rate=0.0, incorrect_rate=0.0,
                not_attempted_rate=0.0, avg_latency=0.0, total_questions=0,
                correct_count=0, incorrect_count=0, not_attempted_count=0,
            )

        correct_count = sum(1 for r in results if r.grade == "CORRECT")
        incorrect_count = sum(1 for r in results if r.grade == "INCORRECT")
        not_attempted_count = sum(1 for r in results if r.grade == "NOT_ATTEMPTED")

        correct_rate = correct_count / total
        incorrect_rate = incorrect_count / total
        not_attempted_rate = not_attempted_count / total

        # accuracy = fraction correct (same as correct_rate)
        accuracy = correct_rate

        # Matches OpenAI's exact F1 computation:
        # is_given_attempted = correct_rate + incorrect_rate
        # accuracy_given_attempted = correct_rate / is_given_attempted
        # f1 = harmonic_mean(accuracy_given_attempted, correct_rate)
        is_given_attempted = correct_rate + incorrect_rate
        accuracy_given_attempted = (
            correct_rate / is_given_attempted
            if is_given_attempted > 0
            else 0.0
        )
        f1 = (
            2 * accuracy_given_attempted * correct_rate
            / (accuracy_given_attempted + correct_rate)
            if (accuracy_given_attempted + correct_rate) > 0
            else 0.0
        )

        avg_latency = sum(r.latency_seconds for r in results) / total

        # Per-topic metrics (using same F1 formula)
        topic_metrics = {}
        topics = set(r.topic for r in results if r.topic)
        for topic in topics:
            topic_results = [r for r in results if r.topic == topic]
            t_total = len(topic_results)
            t_correct = sum(1 for r in topic_results if r.grade == "CORRECT")
            t_incorrect = sum(1 for r in topic_results if r.grade == "INCORRECT")
            t_not_attempted = sum(1 for r in topic_results if r.grade == "NOT_ATTEMPTED")
            t_correct_rate = t_correct / t_total if t_total > 0 else 0.0
            t_incorrect_rate = t_incorrect / t_total if t_total > 0 else 0.0
            t_attempted = t_correct_rate + t_incorrect_rate
            t_acc_attempted = t_correct_rate / t_attempted if t_attempted > 0 else 0.0
            t_f1 = (
                2 * t_acc_attempted * t_correct_rate / (t_acc_attempted + t_correct_rate)
                if (t_acc_attempted + t_correct_rate) > 0 else 0.0
            )
            topic_metrics[topic] = {
                "accuracy": t_correct_rate,
                "f1": t_f1,
                "accuracy_given_attempted": t_acc_attempted,
                "total": t_total,
                "correct": t_correct,
                "incorrect": t_incorrect,
                "not_attempted": t_not_attempted,
            }

        # Per-answer-type metrics (using same F1 formula)
        answer_type_metrics = {}
        answer_types = set(r.answer_type for r in results if r.answer_type)
        for atype in answer_types:
            at_results = [r for r in results if r.answer_type == atype]
            at_total = len(at_results)
            at_correct = sum(1 for r in at_results if r.grade == "CORRECT")
            at_incorrect = sum(1 for r in at_results if r.grade == "INCORRECT")
            at_not_attempted = sum(1 for r in at_results if r.grade == "NOT_ATTEMPTED")
            at_correct_rate = at_correct / at_total if at_total > 0 else 0.0
            at_incorrect_rate = at_incorrect / at_total if at_total > 0 else 0.0
            at_attempted = at_correct_rate + at_incorrect_rate
            at_acc_attempted = at_correct_rate / at_attempted if at_attempted > 0 else 0.0
            at_f1 = (
                2 * at_acc_attempted * at_correct_rate / (at_acc_attempted + at_correct_rate)
                if (at_acc_attempted + at_correct_rate) > 0 else 0.0
            )
            answer_type_metrics[atype] = {
                "accuracy": at_correct_rate,
                "f1": at_f1,
                "accuracy_given_attempted": at_acc_attempted,
                "total": at_total,
                "correct": at_correct,
                "incorrect": at_incorrect,
                "not_attempted": at_not_attempted,
            }

        return SimpleQAMetrics(
            accuracy=accuracy,
            f1=f1,
            correct_rate=correct_rate,
            incorrect_rate=incorrect_rate,
            not_attempted_rate=not_attempted_rate,
            avg_latency=avg_latency,
            total_questions=total,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            not_attempted_count=not_attempted_count,
            topic_metrics=topic_metrics,
            answer_type_metrics=answer_type_metrics,
        )

    def _save_results(self, results: List[SimpleQAResult], metrics: SimpleQAMetrics):
        """Save results with full metadata to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_file = self.output_dir / "simpleqa_results.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps({
                    "original_index": result.original_index,
                    "topic": result.topic,
                    "answer_type": result.answer_type,
                    "grade": result.grade,
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
                }) + "\n")

        # Recompute accuracy_given_attempted for saving
        is_given_attempted = metrics.correct_rate + metrics.incorrect_rate
        accuracy_given_attempted = (
            metrics.correct_rate / is_given_attempted
            if is_given_attempted > 0
            else 0.0
        )

        metrics_file = self.output_dir / "simpleqa_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "accuracy_given_attempted": accuracy_given_attempted,
                "correct_rate": metrics.correct_rate,
                "incorrect_rate": metrics.incorrect_rate,
                "not_attempted_rate": metrics.not_attempted_rate,
                "avg_latency": metrics.avg_latency,
                "total_questions": metrics.total_questions,
                "correct_count": metrics.correct_count,
                "incorrect_count": metrics.incorrect_count,
                "not_attempted_count": metrics.not_attempted_count,
                "topic_metrics": metrics.topic_metrics,
                "answer_type_metrics": metrics.answer_type_metrics,
            }, f, indent=2)

        print(f"Results saved to {self.output_dir}")
