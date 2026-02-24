# Type definitions for BrowseComp benchmark.

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class BrowseCompSample:
    """A single BrowseComp benchmark sample."""
    uid: str
    question: str
    answer: str


@dataclass
class BrowseCompResult:
    """Result from evaluating a single BrowseComp sample."""
    uid: str
    question: str
    answer: str  # Ground truth answer
    model_response: str
    extracted_answer: str
    confidence: float
    is_correct: bool
    response_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """Parsed components from a model response."""
    explanation: str
    exact_answer: str
    confidence: float
    raw_response: str
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    is_correct: bool
    extracted_answer: str
    reasoning: str
    raw_response: str
