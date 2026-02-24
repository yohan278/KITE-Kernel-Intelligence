"""OpenThoughts scoring: exact match + contains fallback."""
from __future__ import annotations

import re
from typing import Any, Dict


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip().lower()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"[^\w\s.]", "", answer)
    return answer


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a reasoning response."""
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"\\boxed\{(.+?)\}",
        r"\*\*(.+?)\*\*\s*$",
        r"(?:therefore|thus|so|hence)[,\s]+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Fallback: last non-empty line
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else response


def score_exact_match(predicted: str, gold: str) -> bool:
    """Score using exact match after normalization."""
    return normalize_answer(predicted) == normalize_answer(gold)


def score_contains(predicted: str, gold: str) -> bool:
    """Score using contains check."""
    return normalize_answer(gold) in normalize_answer(predicted)


def score_openthoughts(results: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate scoring for OpenThoughts benchmark."""
    total = len(results)
    if total == 0:
        return {"total_samples": 0.0, "accuracy": 0.0}

    exact_matches = 0
    contains_matches = 0
    errors = 0

    for r in results.values():
        if r.get("error"):
            errors += 1
            continue

        predicted = extract_final_answer(r.get("model_answer", ""))
        gold = r.get("ground_truth", "")

        if score_exact_match(predicted, gold):
            exact_matches += 1
        if score_contains(predicted, gold):
            contains_matches += 1

    return {
        "total_samples": float(total),
        "exact_match_accuracy": round((exact_matches / total) * 100, 2),
        "contains_accuracy": round((contains_matches / total) * 100, 2),
        "error_count": float(errors),
    }
