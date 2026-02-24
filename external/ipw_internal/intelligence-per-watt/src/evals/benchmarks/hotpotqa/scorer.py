"""HotpotQA scoring: Exact Match + F1 on answer tokens (standard metrics)."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List


def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _get_tokens(s: str) -> List[str]:
    """Get normalized tokens."""
    if not s:
        return []
    return _normalize_answer(s).split()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    prediction_tokens = _get_tokens(prediction)
    ground_truth_tokens = _get_tokens(ground_truth)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def extract_answer(response: str) -> str:
    """Extract the final answer from a response."""
    # Try common patterns
    patterns = [
        r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"(?:therefore|thus|so|hence)[,\s]+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: last sentence
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    return sentences[-1] if sentences else response


def score_hotpotqa(results: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate EM and F1 scores for HotpotQA."""
    total = len(results)
    if total == 0:
        return {"total_samples": 0.0, "em": 0.0, "f1": 0.0}

    em_scores = []
    f1_scores = []
    errors = 0

    for r in results.values():
        if r.get("error"):
            errors += 1
            em_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        predicted = extract_answer(r.get("model_answer", ""))
        gold = r.get("ground_truth", "")

        em_scores.append(exact_match_score(predicted, gold))
        f1_scores.append(f1_score(predicted, gold))

    return {
        "total_samples": float(total),
        "em": round(sum(em_scores) / total * 100, 2),
        "f1": round(sum(f1_scores) / total * 100, 2),
        "error_count": float(errors),
    }
