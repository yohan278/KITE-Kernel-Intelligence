from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .prompts import format_judge_prompt

logger = logging.getLogger(__name__)


@dataclass
class ParsedResponse:
    """Parsed components from a model response."""
    explanation: str
    exact_answer: str
    confidence: float
    raw_response: str
    parse_errors: List[str] = field(default_factory=list)


def parse_model_response(response: str) -> ParsedResponse:
    """Parse model response to extract Explanation, Exact Answer, and Confidence."""
    errors = []
    
    # Extract explanation
    explanation = ""
    match = re.search(r"Explanation:\s*(.+?)(?=(?:Exact Answer:|Confidence:|$))", response, re.IGNORECASE | re.DOTALL)
    if match:
        explanation = match.group(1).strip()
    
    # Extract exact answer
    exact_answer = ""
    match = re.search(r"Exact Answer:\s*(.+?)(?=(?:Confidence:|$))", response, re.IGNORECASE | re.DOTALL)
    if match:
        exact_answer = re.sub(r"\s+", " ", match.group(1).strip())
    else:
        errors.append("Could not extract 'Exact Answer' field")
    
    # Extract confidence (default 100 per convention)
    confidence = 100.0
    match = re.search(r"Confidence:\s*([0-9.]+)\s*%?", response, re.IGNORECASE)
    if match:
        try:
            conf = float(match.group(1))
            if conf <= 1.0 and "." in match.group(1):
                conf *= 100
            confidence = max(0.0, min(100.0, conf))
        except ValueError:
            errors.append(f"Could not parse confidence: {match.group(1)}")
    else:
        errors.append("Could not extract 'Confidence' field, defaulting to 100")
    
    return ParsedResponse(explanation=explanation, exact_answer=exact_answer, confidence=confidence, raw_response=response, parse_errors=errors)


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    is_correct: bool
    extracted_answer: str
    reasoning: str
    extracted_confidence: float
    raw_response: str


def parse_judge_response(response: str) -> JudgeResult:
    """Parse the judge model's response."""
    is_correct = False
    match = re.search(r"correct:\s*(yes|no)", response, re.IGNORECASE)
    if match:
        is_correct = match.group(1).lower() == "yes"
    
    extracted_answer = ""
    match = re.search(r"extracted_final_answer:\s*(.+?)(?=(?:\[correct_answer\]|reasoning:|correct:|$))", response, re.IGNORECASE | re.DOTALL)
    if match:
        extracted_answer = match.group(1).strip()
    
    reasoning = ""
    match = re.search(r"reasoning:\s*(.+?)(?=(?:correct:|confidence:|$))", response, re.IGNORECASE | re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
    
    confidence = 100.0
    match = re.search(r"confidence:\s*([0-9.]+)", response, re.IGNORECASE)
    if match:
        try:
            confidence = float(match.group(1))
        except ValueError:
            pass
    
    return JudgeResult(is_correct=is_correct, extracted_answer=extracted_answer, reasoning=reasoning, extracted_confidence=confidence, raw_response=response)


class FinanceBenchScorer:
    """Scorer for FinanceBench using LLM judge."""
    
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
            judge_text = self.extract_text(judge_response)
        else:
            judge_text = self.call_model(judge_input)
        
        return parse_judge_response(judge_text)
    
    def extract_text(self, response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "content") and response.content is not None:
            return str(response.content)
        return str(response)
    
    def call_model(self, prompt: str) -> str:
        if hasattr(self.judge_model, "__call__"):
            return self.extract_text(self.judge_model(prompt))
        raise ValueError("Judge model is not callable")
    
    def score_sample(self, question: str, model_response: str, correct_answer: str) -> Tuple[bool, float, JudgeResult, ParsedResponse]:
        """Score a single sample. Returns (is_correct, confidence, judge_result, parsed_response)."""
        parsed = parse_model_response(model_response)
        judge_result = self.judge_response(question, model_response, correct_answer)
        return judge_result.is_correct, parsed.confidence, judge_result, parsed


def compute_ece(confidences: List[float], correctness: List[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE). Returns percentage (0-100)."""
    if len(confidences) != len(correctness) or not confidences:
        return 0.0
    
    conf_norm = [c / 100.0 for c in confidences]
    bin_counts = [0] * n_bins
    bin_correct = [0.0] * n_bins
    bin_conf = [0.0] * n_bins
    
    for conf, correct in zip(conf_norm, correctness):
        b = min(int(conf * n_bins), n_bins - 1)
        bin_counts[b] += 1
        bin_correct[b] += 1.0 if correct else 0.0
        bin_conf[b] += conf
    
    n_total = len(confidences)
    ece = sum((bin_counts[b] / n_total) * abs(bin_correct[b] / bin_counts[b] - bin_conf[b] / bin_counts[b])
              for b in range(n_bins) if bin_counts[b] > 0)
    return ece * 100.0


def compute_metrics(results: List[dict]) -> dict:
    """Compute aggregate metrics from results."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {"accuracy": 0.0, "correct_count": 0, "valid_count": 0, "error_count": len(results), "total_count": len(results), "ece_10bin": 0.0, "avg_response_time_seconds": 0.0}
    
    correctness = [r.get("is_correct", False) for r in valid]
    confidences = [r.get("confidence", 100.0) for r in valid]
    times = [r.get("response_time_seconds", 0.0) for r in valid]
    
    # Compute per question type metrics
    by_type = {}
    for r in valid:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if r.get("is_correct", False):
            by_type[qtype]["correct"] += 1
    
    type_accuracy = {k: round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0.0 for k, v in by_type.items()}
    
    return {
        "accuracy": round(sum(correctness) / len(valid) * 100, 2),
        "correct_count": sum(correctness),
        "valid_count": len(valid),
        "error_count": len(results) - len(valid),
        "total_count": len(results),
        "ece_10bin": round(compute_ece(confidences, correctness), 2),
        "avg_response_time_seconds": round(sum(times) / len(times), 2) if times else 0.0,
        "accuracy_by_question_type": type_accuracy,
    }
