# benchmarks/simpleqa/scorer.py
"""
SimpleQA scoring utilities.

Uses LLM-as-judge grading following the official OpenAI simple-evals approach:
- CORRECT: Answer contains important information without contradictions
- INCORRECT: Answer contains factual contradictions
- NOT_ATTEMPTED: Missing important information but no contradictions

Reference: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Literal, Optional

from evals.benchmarks.scorer import (
    BaseScorer,
    LLMJudgeScorer,
    ScoreResult,
    ScorerResult,
    register_scorer,
)

logger = logging.getLogger(__name__)

# Trajectory-aware grading template: extracts final answer from raw agent output
# before comparing against the gold target. Modeled on BrowseComp's JUDGE_PROMPT.
GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

IMPORTANT: The predicted answer may be a raw agent trajectory containing reasoning steps, tool calls, intermediate results, self-corrections, and other artifacts. You must first extract the final answer from this trajectory before grading.

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
    - Hedging and guessing are permissible, provided that the full answer is included AND the uncertainty expressed is googable (i.e., those uncertainties can be googled to resolve the uncertainty).


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not fully included in the answer.
    - No statements in the predicted answer contradict the gold target.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: Malia.
Predicted answer 2: Malia, Move, and Sasha.
Predicted answer 3: Barack Obama's children are named Malia and Sasha. I believe he also has a son named Lent.
Predicted answer 4: Barack Obama has two children, Malia and Sasha. It is possible that he has more children that I am not aware of.
Predicted answer 5: Barack Obama's children are named Malia and Sasha. She also has a daughter named Winkle, but she was born too late for me to know about her.
Predicted answer 6: I think Barack Obama's children are named Malia and Sasha. I'm less sure about this, but I think he also has another son.
Predicted answer 7: Malia Obama and Sasha Obama. It is possible he has more but I am not sure.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the predicted answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that") are also considered incorrect.


Grade the following example. The predicted answer may be a raw agent trajectory — extract the final answer first, then grade it.

Question: {question}
Gold target: {target}
Predicted answer: {predicted}

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the predicted answer>
reasoning: <brief explanation of why the extracted answer is or is not correct>
correct: <yes, no, or not_attempted>"""


GradeType = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]


async def _call_grader_llm(
    prompt: str,
    model: str = "gpt-5-mini-2025-08-07",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
) -> str:
    """Call the grader LLM."""
    # Try litellm first
    try:
        import litellm
        litellm.drop_params = True  # Allow gpt-5 models that don't support temperature

        if api_key:
            # Set appropriate env var based on model provider
            if "claude" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif "gpt" in model.lower() or "o1" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            elif "gemini" in model.lower():
                os.environ["GOOGLE_API_KEY"] = api_key

        call_kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        # gpt-5-mini doesn't support temperature != 1
        if model != "gpt-5-mini-2025-08-07":
            call_kwargs["temperature"] = temperature
        response = await litellm.acompletion(**call_kwargs)
        return response.choices[0].message.content.strip()
    except ImportError:
        pass

    # Try call_llm
    try:
        from call_llm import (
            LLMMessage,
            LLMRequest,
            LLMRole,
            create_litellm_client,
        )

        messages = [LLMMessage(role=LLMRole.USER, content=prompt)]
        req_kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        # gpt-5-mini doesn't support temperature != 1
        if model != "gpt-5-mini-2025-08-07":
            req_kwargs["temperature"] = temperature
        request = LLMRequest(**req_kwargs)

        client = create_litellm_client()
        response = await client.call_llm(request)

        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error}")

        return response.content.strip()
    except ImportError:
        raise ImportError(
            "No LLM client library found. Install 'litellm': pip install litellm"
        )


def _parse_grade(response: str) -> GradeType:
    """Parse the grade from the LLM response.

    Supports structured format (correct: yes/no/not_attempted) as primary,
    with backwards-compatible fallback to keyword and letter-grade parsing.
    """
    # Primary: structured "correct: yes/no/not_attempted" format
    structured_match = re.search(
        r"^correct:\s*(yes|not_attempted|no)", response, re.MULTILINE | re.IGNORECASE
    )
    if structured_match:
        value = structured_match.group(1).lower()
        if value == "yes":
            return "CORRECT"
        elif value == "no":
            return "INCORRECT"
        elif value == "not_attempted":
            return "NOT_ATTEMPTED"

    # Fallback: keyword matching
    response_upper = response.upper().strip()

    if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
        return "CORRECT"
    if "INCORRECT" in response_upper:
        return "INCORRECT"
    if "NOT_ATTEMPTED" in response_upper or "NOT ATTEMPTED" in response_upper:
        return "NOT_ATTEMPTED"

    # Try to find letter grades (A=CORRECT, B=INCORRECT, C=NOT_ATTEMPTED)
    letter_match = re.search(r'\b([ABC])\b', response_upper)
    if letter_match:
        letter = letter_match.group(1)
        if letter == "A":
            return "CORRECT"
        elif letter == "B":
            return "INCORRECT"
        elif letter == "C":
            return "NOT_ATTEMPTED"

    # Default to NOT_ATTEMPTED if parsing fails
    logger.warning(f"Could not parse grade from response: {response[:100]}")
    return "NOT_ATTEMPTED"


async def grade_answer_async(
    question: str,
    gold_target: str,
    predicted_answer: str,
    grader_model: str = "gpt-5-mini-2025-08-07",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade a predicted answer against a gold target using LLM-as-judge.

    Args:
        question: The original question
        gold_target: The correct answer
        predicted_answer: The model's predicted answer
        grader_model: Model to use for grading (default: gpt-4o)
        api_key: Optional API key for the grader model

    Returns:
        Dict with grade, is_correct, is_incorrect, is_not_attempted
    """
    if not predicted_answer or not predicted_answer.strip():
        return {
            "grade": "NOT_ATTEMPTED",
            "is_correct": False,
            "is_incorrect": False,
            "is_not_attempted": True,
        }

    prompt = GRADER_TEMPLATE.format(
        question=question,
        target=gold_target,
        predicted=predicted_answer,
    )

    try:
        response = await _call_grader_llm(
            prompt=prompt,
            model=grader_model,
            temperature=0.0,
            max_tokens=1024,
            api_key=api_key,
        )

        grade = _parse_grade(response)

        return {
            "grade": grade,
            "is_correct": grade == "CORRECT",
            "is_incorrect": grade == "INCORRECT",
            "is_not_attempted": grade == "NOT_ATTEMPTED",
            "raw_response": response,
        }

    except Exception as e:
        logger.error(f"Grading failed: {e}")
        return {
            "grade": "NOT_ATTEMPTED",
            "is_correct": False,
            "is_incorrect": False,
            "is_not_attempted": True,
            "error": str(e),
        }


def grade_answer(
    question: str,
    gold_target: str,
    predicted_answer: str,
    grader_model: str = "gpt-5-mini-2025-08-07",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for grade_answer_async.
    """
    return asyncio.run(grade_answer_async(
        question=question,
        gold_target=gold_target,
        predicted_answer=predicted_answer,
        grader_model=grader_model,
        api_key=api_key,
    ))


def question_scorer(
    model_answer: str,
    ground_truth: str,
    question: str = "",
    grader_model: str = "gpt-5-mini-2025-08-07",
    api_key: Optional[str] = None,
) -> bool:
    """
    Score a model answer using LLM-as-judge grading.

    This is a convenience function for simple boolean scoring.
    For full grading details, use grade_answer() instead.

    Args:
        model_answer: The model's predicted answer
        ground_truth: The correct answer
        question: The original question (optional but recommended)
        grader_model: Model to use for grading
        api_key: Optional API key

    Returns:
        True if CORRECT, False otherwise
    """
    result = grade_answer(
        question=question,
        gold_target=ground_truth,
        predicted_answer=model_answer,
        grader_model=grader_model,
        api_key=api_key,
    )
    return result["is_correct"]


# Fallback string normalization for offline/non-LLM evaluation
def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """
    Normalize a string for fallback comparison.
    Note: LLM grading is preferred over string matching.
    """
    import string
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def normalize_number_str(number_str: str) -> float:
    """Normalize a number string (fallback for offline evaluation)."""
    for char in ["$", "%", ",", " ", "€", "£"]:
        number_str = number_str.replace(char, "")
    number_str = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", number_str, flags=re.IGNORECASE)
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def normalize_date(date_str: str) -> str:
    """Normalize date string (fallback for offline evaluation)."""
    normalized = re.sub(r"[,.]", " ", date_str)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def score_with_contains(model_answer: str, ground_truth: str) -> bool:
    """Fallback scoring that checks if ground truth is contained in answer."""
    if not model_answer or not ground_truth:
        return False
    normalized_answer = normalize_str(model_answer)
    normalized_truth = normalize_str(ground_truth)
    return normalized_truth in normalized_answer


# =============================================================================
# Unified Scorer Class
# =============================================================================


@register_scorer("simpleqa")
@register_scorer("hle")
class SimpleQAScorer(LLMJudgeScorer):
    """LLM judge scorer using OpenAI simple-evals template.

    This scorer uses an LLM to semantically compare the predicted answer
    against the ground truth, allowing for variations in wording and formatting.

    The scorer returns one of three grades:
    - CORRECT: Answer contains all important information without contradictions
    - INCORRECT: Answer contains factual contradictions
    - NOT_ATTEMPTED: Missing important information but no contradictions

    Example:
        >>> scorer = SimpleQAScorer(model="gpt-5-mini-2025-08-07")
        >>> result = await scorer.score(
        ...     question="What are the names of Barack Obama's children?",
        ...     response="Malia and Sasha",
        ...     ground_truth="Malia Obama and Sasha Obama"
        ... )
        >>> print(result.is_correct)  # True
    """

    JUDGE_PROMPT_TEMPLATE = GRADER_TEMPLATE

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> ScorerResult:
        """Score a response using LLM-as-judge grading.

        Args:
            question: The original question
            response: The model's predicted answer
            ground_truth: The correct answer
            **kwargs: Additional arguments (ignored)

        Returns:
            ScorerResult with grading details
        """
        if not response or not response.strip():
            return self._create_not_attempted_result("Empty response")

        if not ground_truth or not ground_truth.strip():
            return self._create_error_result("No ground truth provided")

        try:
            # Use the existing grade_answer_async function
            result = await grade_answer_async(
                question=question,
                gold_target=ground_truth,
                predicted_answer=response,
                grader_model=self.model,
                api_key=self.api_key,
            )

            # Map grade string to ScoreResult enum
            grade_str = result.get("grade", "NOT_ATTEMPTED")
            if grade_str == "CORRECT":
                grade = ScoreResult.CORRECT
            elif grade_str == "INCORRECT":
                grade = ScoreResult.INCORRECT
            else:
                grade = ScoreResult.NOT_ATTEMPTED

            return ScorerResult(
                is_correct=result.get("is_correct", False),
                grade=grade,
                score=1.0 if result.get("is_correct", False) else 0.0,
                explanation=result.get("raw_response"),
                raw_response=result.get("raw_response"),
                error=result.get("error"),
            )

        except Exception as e:
            logger.error(f"SimpleQA scoring failed: {e}")
            return self._create_error_result(str(e))
