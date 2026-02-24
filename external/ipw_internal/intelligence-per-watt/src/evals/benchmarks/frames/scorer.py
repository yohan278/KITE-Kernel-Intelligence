# benchmarks/frames/scorer.py
"""
FRAMES scoring utilities.

Uses LLM-as-judge grading for semantic comparison following the approach
from the official FRAMES evaluation script.

The LLM evaluator compares answers semantically, focusing on whether the
essential information from the ground truth is present in the predicted answer.

Reference: https://github.com/codelion/optillm/blob/main/scripts/eval_frames_benchmark.py
"""
import asyncio
import logging
import os
import re
import string
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Trajectory-aware evaluation prompt for LLM-based grading (based on official FRAMES evaluation)
GRADER_TEMPLATE = """You are evaluating an AI system's answer to a multi-hop factual question.

IMPORTANT: The predicted answer may be a raw agent trajectory containing reasoning steps, tool calls, intermediate results, self-corrections, and other artifacts. You must first extract the final answer from this trajectory before evaluating.

Compare the extracted final answer against the Ground Truth Answer and determine if the prediction is correct.

## Evaluation Guidelines

1. **Extract the final answer first**: The predicted answer may contain a full trajectory — identify the final, definitive answer.
2. **Focus on semantic meaning**: Look for equivalent information - exact wording is not required.
3. **Assess factual accuracy**: Determine whether the essential facts from the Ground Truth are present in the extracted answer.
4. **Ignore minor differences**: Capitalization, punctuation, formatting, and word order don't matter.
5. **Partial credit**: If the Ground Truth has multiple parts, all essential parts must be present for a correct rating.
6. **Additional information**: Extra correct information in the prediction is acceptable, but extra incorrect information is not.

## Question
{question}

## Ground Truth Answer
{ground_truth}

## Predicted Answer
{predicted_answer}

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the predicted answer, or 'None' if no answer is present>
reasoning: <brief explanation of why the extracted answer is or is not correct>
correct: <yes or no>"""


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

        if api_key:
            if "claude" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif "gpt" in model.lower() or "o1" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            elif "gemini" in model.lower():
                os.environ["GOOGLE_API_KEY"] = api_key

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
        request = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        client = create_litellm_client()
        response = await client.call_llm(request)

        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error}")

        return response.content.strip()
    except ImportError:
        raise ImportError(
            "No LLM client library found. Install 'litellm': pip install litellm"
        )


def _parse_grade(response: str) -> bool:
    """Parse the grade from the LLM response.

    Supports structured format (correct: yes/no) as primary,
    with backwards-compatible fallback to TRUE/FALSE parsing.
    """
    # Primary: structured "correct: yes/no" format
    structured_match = re.search(
        r"^correct:\s*(yes|no)", response, re.MULTILINE | re.IGNORECASE
    )
    if structured_match:
        return structured_match.group(1).lower() == "yes"

    # Fallback: TRUE/FALSE keyword matching
    response_upper = response.upper().strip()

    if response_upper.startswith("TRUE") or "TRUE" in response_upper:
        return True
    if response_upper.startswith("FALSE") or "FALSE" in response_upper:
        return False

    # Default to False if parsing fails
    logger.warning(f"Could not parse grade from response: {response[:50]}")
    return False


async def grade_answer_async(
    question: str,
    ground_truth: str,
    predicted_answer: str,
    grader_model: str = "gpt-5-mini-2025-08-07",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade a predicted answer against ground truth using LLM-as-judge.

    Args:
        question: The original question
        ground_truth: The correct answer
        predicted_answer: The model's predicted answer
        grader_model: Model to use for grading (default: claude-3-5-sonnet)
        api_key: Optional API key for the grader model

    Returns:
        Dict with is_correct, raw_response, and any errors
    """
    if not predicted_answer or not predicted_answer.strip():
        return {
            "is_correct": False,
            "raw_response": "",
            "error": "Empty answer",
        }

    prompt = GRADER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer,
    )

    try:
        response = await _call_grader_llm(
            prompt=prompt,
            model=grader_model,
            temperature=0.0,
            max_tokens=1024,
            api_key=api_key,
        )

        is_correct = _parse_grade(response)

        return {
            "is_correct": is_correct,
            "raw_response": response,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Grading failed: {e}")
        return {
            "is_correct": False,
            "raw_response": "",
            "error": str(e),
        }


def grade_answer(
    question: str,
    ground_truth: str,
    predicted_answer: str,
    grader_model: str = "gpt-5-mini-2025-08-07",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for grade_answer_async.
    """
    return asyncio.run(grade_answer_async(
        question=question,
        ground_truth=ground_truth,
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
        question: The original question (recommended for context)
        grader_model: Model to use for grading
        api_key: Optional API key

    Returns:
        True if correct, False otherwise
    """
    result = grade_answer(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=model_answer,
        grader_model=grader_model,
        api_key=api_key,
    )
    return result["is_correct"]


# ============================================================================
# Fallback string-based scoring (for offline evaluation without LLM)
# ============================================================================

def normalize_number_str(number_str: str) -> float:
    """Normalize a number string by removing common units and formatting."""
    for char in ["$", "%", ",", " "]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    """Split a string by multiple delimiters."""
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """
    Normalize a string by removing whitespace and optionally punctuation.
    """
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(element: Any) -> bool:
    """Check if element can be converted to float."""
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def question_scorer_exact_match(
    model_answer: str,
    ground_truth: str,
) -> bool:
    """
    Fallback exact-match scorer (use question_scorer for LLM-based grading).

    This is provided for offline evaluation without LLM API access.
    """
    if model_answer is None:
        model_answer = "None"

    if not model_answer.strip():
        return False

    # If ground truth is a number
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # If ground truth is a list
    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # If ground truth is a string
    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)
