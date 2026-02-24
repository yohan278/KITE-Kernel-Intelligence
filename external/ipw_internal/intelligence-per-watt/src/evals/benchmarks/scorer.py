"""Unified scorer base class for all benchmark evaluations.

This module provides a consistent interface for scoring benchmark responses across
all benchmark types in the evaluation framework.

Class Hierarchy:
    BaseScorer                    # Abstract base for all scorers
    ├── LLMJudgeScorer           # Uses LLM-as-judge for semantic scoring
    ├── RubricScorer             # Criterion-based rubric evaluation
    └── CompositeScorer          # Multi-metric evaluation

Usage:
    from evals.benchmarks.scorer import get_scorer, ScoreResult

    scorer = get_scorer("hle", model="gpt-5-mini-2025-08-07")
    result = await scorer.score(question="...", response="...", ground_truth="...")
    print(result.is_correct, result.grade, result.explanation)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class ScoreResult(Enum):
    """Result categories for scoring."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    NOT_ATTEMPTED = "not_attempted"
    ERROR = "error"


@dataclass
class ScorerResult:
    """Unified result from any scorer.

    Attributes:
        is_correct: Whether the response is considered correct
        grade: The scoring category (CORRECT, INCORRECT, NOT_ATTEMPTED, ERROR)
        score: Numeric score (0.0-1.0 for binary, 0.0-100.0 for percentage)
        explanation: Human-readable explanation of the scoring decision
        raw_response: Raw LLM response (for debugging)
        metadata: Additional benchmark-specific metadata
        error: Error message if scoring failed
    """

    is_correct: bool
    grade: ScoreResult
    score: float = 0.0
    explanation: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        """Set default score based on grade if not provided."""
        if self.score == 0.0 and self.grade == ScoreResult.CORRECT:
            self.score = 1.0


@dataclass
class RubricCriterionResult:
    """Result for a single rubric criterion."""

    criterion_key: str
    description: str
    is_met: bool
    reason: str
    weight: float = 1.0


@dataclass
class RubricResult(ScorerResult):
    """Result from rubric-based scoring with criterion details."""

    criteria_results: List[RubricCriterionResult] = field(default_factory=list)
    points_earned: float = 0.0
    points_possible: float = 0.0
    percentage_score: float = 0.0


@dataclass
class CompositeResult(ScorerResult):
    """Result from composite multi-metric scoring."""

    component_scores: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)


class BaseScorer(ABC):
    """Abstract base class for all benchmark scorers.

    Subclasses must implement:
        - score(): Async method to score a single response

    Provides:
        - score_sync(): Synchronous wrapper
        - _call_llm(): Helper for LLM calls via litellm
        - normalize_str(): String normalization utility
        - normalize_number(): Number string normalization

    Attributes:
        model: Model ID for LLM-based scoring
        api_key: API key for the scoring model
        temperature: Sampling temperature for LLM calls
        max_retries: Maximum retry attempts for LLM calls
    """

    # Class-level defaults
    DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        temperature: float = None,
        max_retries: int = None,
    ):
        """Initialize the scorer.

        Args:
            model: Model ID for LLM-based scoring
            api_key: API key (falls back to env var based on model)
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES

    @abstractmethod
    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> ScorerResult:
        """Score a response against ground truth.

        Args:
            question: The original question/prompt
            response: The model's response to score
            ground_truth: The expected correct answer
            **kwargs: Additional benchmark-specific arguments

        Returns:
            ScorerResult with scoring details
        """
        pass

    def score_sync(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> ScorerResult:
        """Synchronous wrapper for score().

        Args:
            question: The original question/prompt
            response: The model's response to score
            ground_truth: The expected correct answer
            **kwargs: Additional benchmark-specific arguments

        Returns:
            ScorerResult with scoring details
        """
        return asyncio.run(self.score(question, response, ground_truth, **kwargs))

    async def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 500,
    ) -> str:
        """Call the LLM for scoring.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            ImportError: If litellm is not installed
            Exception: If LLM call fails after retries
        """
        try:
            import litellm
            litellm.drop_params = True  # Allow gpt-5 models that don't support temperature

            # Set API key based on model provider
            if self.api_key:
                if "gpt" in self.model.lower() or "o1" in self.model.lower():
                    os.environ["OPENAI_API_KEY"] = self.api_key
                elif "claude" in self.model.lower():
                    os.environ["ANTHROPIC_API_KEY"] = self.api_key
                elif "gemini" in self.model.lower():
                    os.environ["GOOGLE_API_KEY"] = self.api_key

            last_error = None
            for attempt in range(self.max_retries):
                try:
                    call_kwargs = dict(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                    )
                    # gpt-5-mini doesn't support temperature != 1
                    if self.model != "gpt-5-mini-2025-08-07":
                        call_kwargs["temperature"] = self.temperature
                    response = await litellm.acompletion(**call_kwargs)
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"LLM scoring call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

            raise last_error

        except ImportError:
            raise ImportError(
                "litellm package required for LLM-based scoring. "
                "Install with: pip install litellm"
            )

    @staticmethod
    def normalize_str(input_str: str, remove_punct: bool = True) -> str:
        """Normalize a string for comparison.

        Args:
            input_str: String to normalize
            remove_punct: Whether to remove punctuation

        Returns:
            Normalized string (lowercase, no whitespace, optionally no punct)
        """
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        return no_spaces.lower()

    @staticmethod
    def normalize_number(number_str: str) -> float:
        """Normalize a number string for comparison.

        Args:
            number_str: String containing a number

        Returns:
            Parsed float value, or inf if parsing fails
        """
        for char in ["$", "%", ",", " ", "€", "£"]:
            number_str = number_str.replace(char, "")
        # Remove ordinal suffixes
        number_str = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", number_str, flags=re.IGNORECASE)
        try:
            return float(number_str)
        except ValueError:
            return float("inf")

    @staticmethod
    def is_float(value: Any) -> bool:
        """Check if a value can be parsed as a float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _create_error_result(self, error_msg: str) -> ScorerResult:
        """Create an error result.

        Args:
            error_msg: Error message

        Returns:
            ScorerResult with ERROR grade
        """
        return ScorerResult(
            is_correct=False,
            grade=ScoreResult.ERROR,
            score=0.0,
            error=error_msg,
        )

    def _create_not_attempted_result(self, reason: str) -> ScorerResult:
        """Create a not-attempted result.

        Args:
            reason: Reason for not attempting

        Returns:
            ScorerResult with NOT_ATTEMPTED grade
        """
        return ScorerResult(
            is_correct=False,
            grade=ScoreResult.NOT_ATTEMPTED,
            score=0.0,
            explanation=reason,
        )


class LLMJudgeScorer(BaseScorer):
    """Base class for LLM-as-judge scorers.

    Uses a language model to evaluate semantic equivalence between
    a response and ground truth. Subclasses define the judge prompt template.
    """

    # Trajectory-aware judge prompt: extracts final answer from raw agent output
    JUDGE_PROMPT_TEMPLATE: str = """Judge whether the following [response] to [question] is correct based on the [ground_truth] below.

IMPORTANT: The response may be a raw agent trajectory containing reasoning steps, tool calls, intermediate results, self-corrections, and other artifacts. You must first extract the final answer from this trajectory before grading.

[question]: {question}

[response]: {response}

[ground_truth]: {ground_truth}

Your response MUST use exactly this format:
extracted_final_answer: <the final answer extracted from the response, or 'None' if no answer is present>
reasoning: <brief explanation of why the extracted answer is or is not correct compared to the ground truth>
correct: <yes, no, or not_attempted>"""

    def _parse_grade(self, response: str) -> ScoreResult:
        """Parse the grade from an LLM response.

        Supports structured format (correct: yes/no/not_attempted) as primary,
        with backwards-compatible fallback to keyword and letter-grade parsing.

        Args:
            response: LLM response text

        Returns:
            Parsed ScoreResult
        """
        # Primary: structured "correct: yes/no/not_attempted" format
        structured_match = re.search(
            r"^correct:\s*(yes|not_attempted|no)", response, re.MULTILINE | re.IGNORECASE
        )
        if structured_match:
            value = structured_match.group(1).lower()
            if value == "yes":
                return ScoreResult.CORRECT
            elif value == "no":
                return ScoreResult.INCORRECT
            elif value == "not_attempted":
                return ScoreResult.NOT_ATTEMPTED

        # Fallback: keyword matching
        response_upper = response.upper().strip()

        if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
            return ScoreResult.CORRECT
        if "INCORRECT" in response_upper:
            return ScoreResult.INCORRECT
        if "NOT_ATTEMPTED" in response_upper or "NOT ATTEMPTED" in response_upper:
            return ScoreResult.NOT_ATTEMPTED

        # Try letter grades (A=CORRECT, B=INCORRECT, C=NOT_ATTEMPTED)
        letter_match = re.search(r"\b([ABC])\b", response_upper)
        if letter_match:
            letter = letter_match.group(1)
            if letter == "A":
                return ScoreResult.CORRECT
            elif letter == "B":
                return ScoreResult.INCORRECT
            elif letter == "C":
                return ScoreResult.NOT_ATTEMPTED

        logger.warning(f"Could not parse grade from: {response[:100]}")
        return ScoreResult.NOT_ATTEMPTED

    def _format_prompt(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> str:
        """Format the judge prompt.

        Args:
            question: Original question
            response: Model response
            ground_truth: Expected answer
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        return self.JUDGE_PROMPT_TEMPLATE.format(
            question=question or "(No question provided)",
            response=response,
            ground_truth=ground_truth,
            predicted=response,  # Alias for compatibility
            target=ground_truth,  # Alias for compatibility
            **kwargs,
        )

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        **kwargs,
    ) -> ScorerResult:
        """Score using LLM-as-judge.

        Args:
            question: Original question
            response: Model response to score
            ground_truth: Expected answer
            **kwargs: Additional arguments

        Returns:
            ScorerResult with scoring details
        """
        if not response or not response.strip():
            return self._create_not_attempted_result("Empty response")

        if not ground_truth or not ground_truth.strip():
            return self._create_error_result("No ground truth provided")

        try:
            prompt = self._format_prompt(question, response, ground_truth, **kwargs)
            llm_response = await self._call_llm(prompt, max_tokens=1024)
            grade = self._parse_grade(llm_response)

            # Extract the final answer from structured response if present
            metadata = {}
            extracted_match = re.search(
                r"^extracted_final_answer:\s*(.+)", llm_response, re.MULTILINE
            )
            if extracted_match:
                metadata["extracted_answer"] = extracted_match.group(1).strip()

            return ScorerResult(
                is_correct=grade == ScoreResult.CORRECT,
                grade=grade,
                score=1.0 if grade == ScoreResult.CORRECT else 0.0,
                explanation=llm_response,
                raw_response=llm_response,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"LLM judge scoring failed: {e}")
            return self._create_error_result(str(e))


class RubricScorer(BaseScorer):
    """Base class for rubric-based scorers.

    Evaluates responses against multiple criteria defined in a rubric.
    Each criterion is scored independently, and the final score is
    the percentage of criteria met.
    """

    # Override in subclasses
    CRITERION_PROMPT_TEMPLATE: str = ""
    PASSING_THRESHOLD: float = 50.0  # Percentage threshold for is_correct

    async def _score_criterion(
        self,
        response: str,
        criterion_key: str,
        criterion: Dict[str, Any],
    ) -> RubricCriterionResult:
        """Score a single criterion.

        Args:
            response: Model response
            criterion_key: Criterion identifier
            criterion: Criterion definition dict

        Returns:
            RubricCriterionResult with scoring details
        """
        raise NotImplementedError("Subclasses must implement _score_criterion")

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str,
        rubric: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RubricResult:
        """Score against a rubric.

        Args:
            question: Original question
            response: Model response
            ground_truth: May contain serialized rubric
            rubric: Rubric dict (preferred over parsing ground_truth)
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

        # Score each criterion
        try:
            criteria_results = []
            for key, criterion in rubric.items():
                result = await self._score_criterion(response, key, criterion)
                criteria_results.append(result)

            # Calculate aggregate score
            points_earned = sum(1 for r in criteria_results if r.is_met)
            points_possible = len(criteria_results)
            percentage = (points_earned / points_possible * 100) if points_possible > 0 else 0

            is_correct = percentage >= self.PASSING_THRESHOLD
            grade = ScoreResult.CORRECT if is_correct else ScoreResult.INCORRECT

            return RubricResult(
                is_correct=is_correct,
                grade=grade,
                score=percentage / 100,
                explanation=f"Score: {percentage:.1f}% ({points_earned}/{points_possible})",
                criteria_results=criteria_results,
                points_earned=points_earned,
                points_possible=points_possible,
                percentage_score=percentage,
            )

        except Exception as e:
            logger.error(f"Rubric scoring failed: {e}")
            return RubricResult(
                is_correct=False,
                grade=ScoreResult.ERROR,
                error=str(e),
            )


class CompositeScorer(BaseScorer):
    """Base class for composite multi-metric scorers.

    Combines multiple scoring metrics with configurable weights.
    """

    # Override in subclasses
    COMPONENT_WEIGHTS: Dict[str, float] = {}
    PASSING_THRESHOLD: float = 50.0

    @abstractmethod
    async def _score_components(
        self,
        question: str,
        response: str,
        **kwargs,
    ) -> Dict[str, float]:
        """Score individual components.

        Args:
            question: Original question
            response: Model response
            **kwargs: Additional arguments

        Returns:
            Dict mapping component name to score (0-100)
        """
        pass

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str = "",
        **kwargs,
    ) -> CompositeResult:
        """Score using multiple metrics.

        Args:
            question: Original question
            response: Model response
            ground_truth: May be unused for some composite scorers
            **kwargs: Additional arguments

        Returns:
            CompositeResult with component-level details
        """
        if not response or not response.strip():
            return CompositeResult(
                is_correct=False,
                grade=ScoreResult.NOT_ATTEMPTED,
                explanation="Empty response",
            )

        try:
            component_scores = await self._score_components(question, response, **kwargs)

            # Calculate weighted average
            weights = self.COMPONENT_WEIGHTS or {k: 1.0 for k in component_scores}
            total_weight = sum(weights.get(k, 1.0) for k in component_scores)
            weighted_sum = sum(
                component_scores[k] * weights.get(k, 1.0)
                for k in component_scores
            )
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0

            is_correct = overall_score >= self.PASSING_THRESHOLD
            grade = ScoreResult.CORRECT if is_correct else ScoreResult.INCORRECT

            return CompositeResult(
                is_correct=is_correct,
                grade=grade,
                score=overall_score / 100,
                explanation=f"Overall: {overall_score:.1f}%",
                component_scores=component_scores,
                component_weights=weights,
            )

        except Exception as e:
            logger.error(f"Composite scoring failed: {e}")
            return CompositeResult(
                is_correct=False,
                grade=ScoreResult.ERROR,
                error=str(e),
            )


# =============================================================================
# Scorer Registry and Factory
# =============================================================================

# Registry mapping benchmark names to scorer classes
SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {}


def register_scorer(name: str) -> Callable[[Type[BaseScorer]], Type[BaseScorer]]:
    """Decorator to register a scorer class.

    Args:
        name: Benchmark name (case-insensitive)

    Returns:
        Decorator function

    Example:
        @register_scorer("gaia")
        class GAIAScorer(BaseScorer):
            ...
    """
    def decorator(cls: Type[BaseScorer]) -> Type[BaseScorer]:
        SCORER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_scorer(
    benchmark: Union[str, Any],
    model: str = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseScorer:
    """Get the appropriate scorer for a benchmark.

    Args:
        benchmark: Benchmark name (str) or BenchmarkType enum
        model: Model for LLM-based scoring (default: gpt-5-mini-2025-08-07)
        api_key: API key for the scoring model
        **kwargs: Additional scorer-specific arguments

    Returns:
        Configured scorer instance

    Example:
        >>> scorer = get_scorer("hle", model="gpt-5-mini-2025-08-07")
        >>> result = await scorer.score(
        ...     question="What is the capital of France?",
        ...     response="Paris",
        ...     ground_truth="Paris"
        ... )
        >>> print(result.is_correct)  # True
    """
    # Handle enum input
    if hasattr(benchmark, "value"):
        benchmark = benchmark.value

    benchmark_lower = str(benchmark).lower()

    # Lazy-load scorers on first access to avoid circular imports
    if not SCORER_REGISTRY:
        _load_scorers()

    scorer_class = SCORER_REGISTRY.get(benchmark_lower)
    if not scorer_class:
        # Default to SimpleQA scorer (generic LLM judge)
        from evals.benchmarks.simpleqa.scorer import SimpleQAScorer
        scorer_class = SimpleQAScorer
        logger.warning(f"No specific scorer for '{benchmark}', using default SimpleQAScorer")

    return scorer_class(model=model, api_key=api_key, **kwargs)


def _load_scorers():
    """Load all scorer classes into the registry.

    This is called lazily on first use of get_scorer() to avoid
    circular import issues.
    """
    # Import each scorer module to trigger @register_scorer decorators
    try:
        from evals.benchmarks.simpleqa.scorer import SimpleQAScorer  # noqa: F401
    except ImportError:
        pass

    try:
        from evals.benchmarks.gaia.scorer import GAIAScorer  # noqa: F401
    except ImportError:
        pass

    try:
        from evals.benchmarks.browsecomp.scorer import BrowseCompScorer  # noqa: F401
    except ImportError:
        pass

    try:
        from evals.benchmarks.deepresearch.scorer import DeepResearchScorer  # noqa: F401
    except ImportError:
        pass

    try:
        from evals.benchmarks.apex.grading.scorer import APEXScorer  # noqa: F401
    except ImportError:
        pass

    try:
        from evals.benchmarks.frames.scorer import FRAMESScorer  # noqa: F401
    except ImportError:
        pass

    try:
        from evals.benchmarks.financebench.scorer import FinanceBenchScorer  # noqa: F401
    except ImportError:
        pass


def get_available_scorers() -> Dict[str, str]:
    """Get a mapping of benchmark names to scorer class names.

    Returns:
        Dict mapping benchmark name to scorer class name
    """
    if not SCORER_REGISTRY:
        _load_scorers()
    return {name: cls.__name__ for name, cls in SCORER_REGISTRY.items()}


__all__ = [
    # Enums and result types
    "ScoreResult",
    "ScorerResult",
    "RubricCriterionResult",
    "RubricResult",
    "CompositeResult",
    # Base classes
    "BaseScorer",
    "LLMJudgeScorer",
    "RubricScorer",
    "CompositeScorer",
    # Factory
    "get_scorer",
    "register_scorer",
    "get_available_scorers",
    "SCORER_REGISTRY",
]
