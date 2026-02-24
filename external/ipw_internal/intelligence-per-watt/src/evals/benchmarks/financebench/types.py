from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    """A single piece of evidence for a FinanceBench sample."""
    evidence_text: str
    evidence_doc_name: str
    evidence_page_num: int
    evidence_text_full_page: str = ""


@dataclass(frozen=True)
class FinanceBenchSample:
    """A single FinanceBench benchmark sample."""
    uid: str
    question: str
    answer: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str
    justification: str = ""
    evidence: tuple = field(default_factory=tuple)


@dataclass
class FinanceBenchResult:
    """Result from evaluating a single FinanceBench sample."""
    uid: str
    question: str
    ground_truth: str
    model_response: str
    extracted_answer: str
    confidence: float
    is_correct: bool
    company: str = ""
    question_type: str = ""
    response_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
