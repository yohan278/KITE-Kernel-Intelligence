# Scoring utilities for BrowseComp: LLM judging and metrics calculation.

import logging
import re
from typing import Any, List, Optional, Tuple

from evals.benchmarks.scorer import (
    BaseScorer,
    LLMJudgeScorer,
    ScoreResult,
    ScorerResult,
    register_scorer,
)
from .prompts import format_judge_prompt, parse_model_response
from .types import JudgeResult, ParsedResponse

logger = logging.getLogger(__name__)


def parse_judge_response(text: str) -> JudgeResult:
    """Parse an LLM judge response into a JudgeResult.

    Expected format from the judge prompt:
        extracted_final_answer: <answer>
        reasoning: <explanation>
        correct: yes/no
    """
    extracted = ""
    reasoning = ""
    is_correct = False

    answer_match = re.search(
        r"extracted_final_answer:\s*(.+?)(?=\nreasoning:|\ncorrect:|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        extracted = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?=\ncorrect:|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    correct_match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    if correct_match:
        is_correct = correct_match.group(1).lower() == "yes"

    return JudgeResult(
        is_correct=is_correct,
        extracted_answer=extracted,
        reasoning=reasoning,
        raw_response=text,
    )


class BrowseCompScorerLegacy:
    """Legacy scorer for BrowseComp using LLM judge.

    Preserved for backward compatibility. Use BrowseCompScorer for new code.
    """

    def __init__(self, judge_model: Optional[Any] = None, judge_orchestrator: Optional[Any] = None):
        self.judge_model = judge_model
        self.judge_orchestrator = judge_orchestrator

    def judge_response(self, question: str, response: str, correct_answer: str) -> JudgeResult:
        """Judge whether a response is correct."""
        if self.judge_orchestrator is None and self.judge_model is None:
            raise ValueError("No judge configured")

        judge_input = format_judge_prompt(question, response, correct_answer)

        if self.judge_orchestrator:
            judge_response = self.judge_orchestrator.run(judge_input)
            judge_text = self._extract_text(judge_response)
        else:
            judge_text = self._call_model(judge_input)

        return parse_judge_response(judge_text)

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "content") and response.content is not None:
            return str(response.content)
        return str(response)

    def _call_model(self, prompt: str) -> str:
        if hasattr(self.judge_model, "__call__"):
            return self._extract_text(self.judge_model(prompt))
        raise ValueError("Judge model is not callable")

    def score_sample(self, question: str, model_response: str, correct_answer: str) -> Tuple[bool, float, JudgeResult, ParsedResponse]:
        """Score a single sample. Returns (is_correct, confidence, judge_result, parsed_response)."""
        parsed = parse_model_response(model_response)
        judge_result = self.judge_response(question, model_response, correct_answer)
        return judge_result.is_correct, parsed.confidence, judge_result, parsed


# =============================================================================
# Unified Scorer Class
# =============================================================================


@register_scorer("browsecomp")
class BrowseCompScorer(LLMJudgeScorer):
    """BrowseComp scorer with confidence extraction.

    This scorer uses the BrowseComp-specific judge prompt which evaluates
    whether the model's response correctly answers the question and extracts
    a confidence score from the response.

    Example:
        >>> scorer = BrowseCompScorer(model="gpt-5-mini-2025-08-07")
        >>> result = await scorer.score(
        ...     question="What is the capital of France?",
        ...     response="Explanation: Paris is the capital...\\nExact Answer: Paris\\nConfidence: 95%",
        ...     ground_truth="Paris"
        ... )
        >>> print(result.is_correct)  # True
    """

    def __init__(self, model: str = None, api_key: Optional[str] = None, **kwargs):
        """Initialize the BrowseComp scorer.

        Args:
            model: Model for LLM judge scoring
            api_key: API key for the LLM
            **kwargs: Additional arguments passed to LLMJudgeScorer
        """
        super().__init__(model=model, api_key=api_key, **kwargs)

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> ScorerResult:
        """Score a response using BrowseComp's LLM judge.

        Args:
            question: The original question
            response: The model's response (should contain Explanation, Exact Answer, Confidence)
            ground_truth: The correct answer
            **kwargs: Additional arguments (ignored)

        Returns:
            ScorerResult with grading details and confidence metadata
        """
        if not response or not response.strip():
            return self._create_not_attempted_result("Empty response")

        if not ground_truth or not ground_truth.strip():
            return self._create_error_result("No ground truth provided")

        # Parse the model response to extract confidence
        parsed = parse_model_response(response)

        try:
            # Format the BrowseComp-specific judge prompt
            judge_prompt = format_judge_prompt(question, response, ground_truth)
            llm_response = await self._call_llm(judge_prompt, max_tokens=1024)

            # Parse the judge response
            judge_result = parse_judge_response(llm_response)

            return ScorerResult(
                is_correct=judge_result.is_correct,
                grade=ScoreResult.CORRECT if judge_result.is_correct else ScoreResult.INCORRECT,
                score=1.0 if judge_result.is_correct else 0.0,
                explanation=judge_result.reasoning,
                raw_response=llm_response,
                metadata={
                    "confidence": parsed.confidence,
                    "extracted_answer": judge_result.extracted_answer,
                    "parse_errors": parsed.parse_errors,
                },
            )

        except Exception as e:
            logger.error(f"BrowseComp scoring failed: {e}")
            return self._create_error_result(str(e))


def compute_ece(confidences: List[float], correctness: List[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE). Returns percentage (0-100)."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    
    bins = [[0, 0.0, 0.0] for _ in range(n_bins)]  # [count, correct, conf]
    for conf, correct in zip(confidences, correctness):
        b = min(int((conf / 100.0) * n_bins), n_bins - 1)
        bins[b][0] += 1
        bins[b][1] += float(correct)
        bins[b][2] += conf / 100.0
    
    ece = sum((count / len(confidences)) * abs(correct / count - conf_sum / count)
              for count, correct, conf_sum in bins if count > 0)
    return ece * 100.0


def compute_metrics(results: List[dict]) -> dict:
    """Compute aggregate metrics from results."""
    valid = [r for r in results if not r.get("error")]
    if not results:
        return {"accuracy": 0.0, "correct_count": 0, "valid_count": 0, "error_count": 0, "total_count": 0, "ece_10bin": 0.0, "avg_response_time_seconds": 0.0}
    
    correctness = [r.get("is_correct", False) for r in valid]
    confidences = [r.get("confidence", 100.0) for r in valid]
    times = [r.get("response_time_seconds", 0.0) for r in valid]
    
    return {
        "accuracy": round(sum(correctness) / len(results) * 100, 2),
        "correct_count": sum(correctness),
        "valid_count": len(valid),
        "error_count": len(results) - len(valid),
        "total_count": len(results),
        "ece_10bin": round(compute_ece(confidences, correctness), 2) if valid else 0.0,
        "avg_response_time_seconds": round(sum(times) / len(times), 2) if times else 0.0,
    }
