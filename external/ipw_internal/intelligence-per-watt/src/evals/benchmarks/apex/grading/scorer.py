"""APEX Scorer - Rubric-based grading for APEX benchmark.

This module provides the APEXScorer class that wraps the existing
grading executor to provide a unified scorer interface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evals.benchmarks.scorer import (
    BaseScorer,
    RubricScorer,
    RubricCriterionResult,
    RubricResult,
    ScoreResult,
    ScorerResult,
    register_scorer,
)
from .executor import (
    grade_solution_against_rubric,
    DEFAULT_GRADING_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class APEXCriterionResult(RubricCriterionResult):
    """Extended criterion result for APEX with additional metadata."""

    sources: str = ""
    criterion_type: List[str] = None
    raw_response: str = ""
    tokens_used: int = 0
    execution_time_seconds: float = 0.0

    def __post_init__(self):
        if self.criterion_type is None:
            self.criterion_type = []


@register_scorer("apex")
class APEXScorer(RubricScorer):
    """Rubric-based scorer for APEX benchmark.

    APEX uses detailed rubrics with multiple criteria to evaluate solutions.
    Each criterion is graded independently by an LLM, and the final score
    is the percentage of criteria met.

    A score of >= 50% is considered passing (correct).

    Example:
        >>> scorer = APEXScorer(model="gpt-5-mini-2025-08-07")
        >>> result = await scorer.score(
        ...     question="Implement a sorting algorithm",
        ...     response="def sort(arr): return sorted(arr)",
        ...     ground_truth="",
        ...     rubric={"criterion1": {"description": "Handles empty input"}}
        ... )
        >>> print(result.percentage_score, result.is_correct)
    """

    PASSING_THRESHOLD = 50.0

    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        grading_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the APEX scorer.

        Args:
            model: Model for criterion grading
            api_key: API key for the grading model
            grading_prompt_template: Custom grading prompt (defaults to APEX prompt)
            **kwargs: Additional arguments passed to RubricScorer
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.grading_prompt_template = grading_prompt_template or DEFAULT_GRADING_PROMPT

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        rubric: Optional[Dict[str, Any]] = None,
        response_images: Optional[List[str]] = None,
        **kwargs,
    ) -> RubricResult:
        """Score a solution against an APEX rubric.

        Args:
            question: The task prompt (unused, rubric contains criteria)
            response: The solution to grade
            ground_truth: May contain serialized rubric if rubric param not provided
            rubric: The grading rubric dictionary
            response_images: Optional images for vision-based grading
            **kwargs: Additional arguments

        Returns:
            RubricResult with criterion-level details
        """
        if not response or not response.strip():
            return RubricResult(
                is_correct=False,
                grade=ScoreResult.NOT_ATTEMPTED,
                explanation="Empty response",
            )

        # Get rubric from parameter or try to parse from ground_truth
        if not rubric and ground_truth:
            try:
                import ast
                rubric = ast.literal_eval(ground_truth)
            except (ValueError, SyntaxError):
                try:
                    import json
                    rubric = json.loads(ground_truth)
                except json.JSONDecodeError:
                    pass

        if not rubric:
            return RubricResult(
                is_correct=False,
                grade=ScoreResult.ERROR,
                error="No rubric provided",
            )

        try:
            # Use the existing APEX grading executor
            grading_config = {
                "model_id": self.model,
                "temperature": self.temperature,
                "max_tokens": 1024,
                "api_key": self.api_key,
            }

            result = await grade_solution_against_rubric(
                solution=response,
                rubric=rubric,
                grading_model_config=grading_config,
                grading_prompt_template=self.grading_prompt_template,
                response_images=response_images,
            )

            # Convert criteria results to our format
            criteria_results = []
            for cr in result.get("criteria_results", []):
                criteria_results.append(APEXCriterionResult(
                    criterion_key=cr.get("criterion_key", ""),
                    description=cr.get("description", ""),
                    is_met=cr.get("autorating", False),
                    reason=cr.get("reason", ""),
                    weight=float(cr.get("weight", 1)) if cr.get("weight") else 1.0,
                    sources=cr.get("sources", ""),
                    criterion_type=cr.get("criterion_type", []),
                    raw_response=cr.get("raw_response", ""),
                    tokens_used=cr.get("tokens_used", 0),
                    execution_time_seconds=cr.get("execution_time_seconds", 0.0),
                ))

            percentage = result.get("percentage_score", 0.0)
            is_correct = percentage >= self.PASSING_THRESHOLD
            grade = ScoreResult.CORRECT if is_correct else ScoreResult.INCORRECT

            return RubricResult(
                is_correct=is_correct,
                grade=grade,
                score=percentage / 100,
                explanation=f"Score: {percentage:.1f}% ({result.get('points_earned', 0)}/{result.get('points_possible', 0)})",
                criteria_results=criteria_results,
                points_earned=result.get("points_earned", 0),
                points_possible=result.get("points_possible", 0),
                percentage_score=percentage,
                metadata={
                    "total_grading_tokens": result.get("total_grading_tokens", 0),
                    "total_grading_cost": result.get("total_grading_cost", 0.0),
                    "execution_time_seconds": result.get("execution_time_seconds", 0.0),
                    "grading_error": result.get("grading_error"),
                },
            )

        except Exception as e:
            logger.error(f"APEX scoring failed: {e}")
            return RubricResult(
                is_correct=False,
                grade=ScoreResult.ERROR,
                error=str(e),
            )


__all__ = ["APEXScorer", "APEXCriterionResult"]
