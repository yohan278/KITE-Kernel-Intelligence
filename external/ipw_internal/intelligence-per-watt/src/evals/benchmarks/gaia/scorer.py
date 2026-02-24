from __future__ import annotations

import json
import logging
import re
import string
import warnings
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

from evals.benchmarks.scorer import (
    BaseScorer,
    LLMJudgeScorer,
    ScoreResult,
    ScorerResult,
    register_scorer,
)


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False
        
    if model_answer is None:
        model_answer = "None"

    # if gt is a number
    if is_float(ground_truth):
        print(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        print(f"Evaluating {model_answer} as a comma separated list.")
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.", UserWarning
            )
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        print(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)


def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


# =============================================================================
# Unified Scorer Class
# =============================================================================


@register_scorer("gaia")
class GAIAScorer(BaseScorer):
    """GAIA scorer with string normalization and optional LLM fallback.

    This scorer first attempts exact match comparison using string normalization
    (handling numbers, lists, and strings). If the exact match fails and LLM
    fallback is enabled, it uses an LLM judge for semantic comparison.

    The exact match logic handles:
    - Numbers: Normalizes currency, percentages, commas
    - Lists: Splits by comma/semicolon and compares elements
    - Strings: Normalizes whitespace and optionally punctuation

    Example:
        >>> scorer = GAIAScorer(use_llm_fallback=False)
        >>> result = await scorer.score(
        ...     question="What is the capital of France?",
        ...     response="paris",
        ...     ground_truth="Paris"
        ... )
        >>> print(result.is_correct)  # True (exact match after normalization)
    """

    # Trajectory-aware LLM judge prompt for fallback scoring
    FALLBACK_PROMPT_TEMPLATE = """Your job is to determine if the predicted answer is semantically equivalent to the gold target.

IMPORTANT: The predicted answer may be a raw agent trajectory containing reasoning steps, tool calls, intermediate results, self-corrections, and other artifacts. You must first extract the final answer from this trajectory before comparing.

Question: {question}
Gold target: {ground_truth}
Predicted answer: {response}

Consider the following:
- Numerical answers should match exactly (accounting for different formats like $1,000 vs 1000)
- List answers should contain all elements (order may vary)
- String answers should have the same meaning (case and punctuation don't matter)

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the predicted answer, or 'None' if no answer is present>
reasoning: <brief explanation of why the extracted answer matches or does not match the gold target>
correct: <yes or no>"""

    def __init__(
        self,
        use_llm_fallback: bool = True,
        model: str = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the GAIA scorer.

        Args:
            use_llm_fallback: Whether to use LLM for ambiguous cases (default: True)
            model: Model for LLM fallback scoring
            api_key: API key for the LLM
            **kwargs: Additional arguments passed to BaseScorer
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.use_llm_fallback = use_llm_fallback

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> ScorerResult:
        """Score a response using GAIA's exact match + optional LLM fallback.

        Args:
            question: The original question
            response: The model's predicted answer
            ground_truth: The correct answer
            **kwargs: Additional arguments (ignored)

        Returns:
            ScorerResult with grading details
        """
        if not response or response.strip() == "":
            return self._create_not_attempted_result("Empty response")

        if not ground_truth or ground_truth.strip() == "":
            return self._create_error_result("No ground truth provided")

        # Try exact match first (fast, no API call)
        is_correct = question_scorer(response, ground_truth)

        if is_correct:
            return ScorerResult(
                is_correct=True,
                grade=ScoreResult.CORRECT,
                score=1.0,
                explanation="Exact match after normalization",
                metadata={"match_type": "exact"},
            )

        # If LLM fallback is disabled, return incorrect
        if not self.use_llm_fallback:
            return ScorerResult(
                is_correct=False,
                grade=ScoreResult.INCORRECT,
                score=0.0,
                explanation="No exact match (LLM fallback disabled)",
                metadata={"match_type": "exact_failed"},
            )

        # Use LLM fallback for semantic comparison
        try:
            prompt = self.FALLBACK_PROMPT_TEMPLATE.format(
                question=question or "(No question provided)",
                response=response,
                ground_truth=ground_truth,
            )
            logger.debug(f"GAIA LLM judge prompt:\n{prompt[:2000]}")
            llm_response = await self._call_llm(prompt, max_tokens=1024)
            logger.info(f"GAIA LLM judge response: {llm_response[:500]}")

            # Primary: structured "correct: yes/no" format
            structured_match = re.search(
                r"^correct:\s*(yes|no)", llm_response, re.MULTILINE | re.IGNORECASE
            )
            if structured_match:
                is_correct_llm = structured_match.group(1).lower() == "yes"
            else:
                # Fallback: keyword matching
                is_correct_llm = "CORRECT" in llm_response.upper() and "INCORRECT" not in llm_response.upper()

            # Extract the final answer from structured response if present
            metadata = {"match_type": "llm_fallback"}
            extracted_match = re.search(
                r"^extracted_final_answer:\s*(.+)", llm_response, re.MULTILINE
            )
            if extracted_match:
                metadata["extracted_answer"] = extracted_match.group(1).strip()

            return ScorerResult(
                is_correct=is_correct_llm,
                grade=ScoreResult.CORRECT if is_correct_llm else ScoreResult.INCORRECT,
                score=1.0 if is_correct_llm else 0.0,
                explanation=f"LLM fallback: {llm_response}",
                raw_response=llm_response,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"GAIA LLM fallback FAILED (returning INCORRECT): {e}")
            # If LLM fails, fall back to exact match result
            return ScorerResult(
                is_correct=False,
                grade=ScoreResult.INCORRECT,
                score=0.0,
                explanation=f"Exact match failed, LLM fallback error: {e}",
                error=str(e),
                metadata={"match_type": "llm_fallback_error"},
            )