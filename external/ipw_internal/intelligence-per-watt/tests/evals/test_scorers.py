"""Unit tests for unified scorer interface."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evals.benchmarks.scorer import (
    BaseScorer,
    LLMJudgeScorer,
    RubricScorer,
    CompositeScorer,
    ScoreResult,
    ScorerResult,
    RubricResult,
    CompositeResult,
    RubricCriterionResult,
    get_scorer,
    register_scorer,
    SCORER_REGISTRY,
)


class TestScoreResult:
    """Tests for ScoreResult enum."""

    def test_score_result_values(self):
        """Test ScoreResult enum has expected values."""
        assert ScoreResult.CORRECT.value == "correct"
        assert ScoreResult.INCORRECT.value == "incorrect"
        assert ScoreResult.NOT_ATTEMPTED.value == "not_attempted"
        assert ScoreResult.ERROR.value == "error"

    def test_score_result_members(self):
        """Test ScoreResult has all expected members."""
        members = list(ScoreResult)
        assert len(members) == 4
        assert ScoreResult.CORRECT in members
        assert ScoreResult.INCORRECT in members
        assert ScoreResult.NOT_ATTEMPTED in members
        assert ScoreResult.ERROR in members


class TestScorerResult:
    """Tests for ScorerResult dataclass."""

    def test_scorer_result_defaults(self):
        """Test ScorerResult default values."""
        result = ScorerResult(is_correct=True, grade=ScoreResult.CORRECT)
        # score should be set to 1.0 for CORRECT grade when score is 0.0
        assert result.score == 1.0
        assert result.explanation is None
        assert result.raw_response is None
        assert result.metadata == {}
        assert result.error is None

    def test_scorer_result_incorrect_defaults(self):
        """Test ScorerResult defaults for incorrect answer."""
        result = ScorerResult(is_correct=False, grade=ScoreResult.INCORRECT)
        assert result.score == 0.0
        assert result.is_correct is False

    def test_scorer_result_with_metadata(self):
        """Test ScorerResult with custom metadata."""
        result = ScorerResult(
            is_correct=True,
            grade=ScoreResult.CORRECT,
            score=0.95,
            explanation="Good answer",
            metadata={"confidence": 0.9},
        )
        assert result.score == 0.95
        assert result.explanation == "Good answer"
        assert result.metadata == {"confidence": 0.9}

    def test_scorer_result_explicit_score(self):
        """Test ScorerResult with explicit score."""
        result = ScorerResult(is_correct=True, grade=ScoreResult.CORRECT, score=0.5)
        # Should keep explicit score even if CORRECT
        assert result.score == 0.5


class TestRubricResult:
    """Tests for RubricResult dataclass."""

    def test_rubric_result_defaults(self):
        """Test RubricResult default values."""
        result = RubricResult(is_correct=True, grade=ScoreResult.CORRECT)
        assert result.criteria_results == []
        assert result.points_earned == 0.0
        assert result.points_possible == 0.0
        assert result.percentage_score == 0.0

    def test_rubric_result_with_criteria(self):
        """Test RubricResult with criterion results."""
        criteria = [
            RubricCriterionResult(
                criterion_key="c1",
                description="Test criterion",
                is_met=True,
                reason="Passed",
            ),
        ]
        result = RubricResult(
            is_correct=True,
            grade=ScoreResult.CORRECT,
            criteria_results=criteria,
            points_earned=1.0,
            points_possible=1.0,
            percentage_score=100.0,
        )
        assert len(result.criteria_results) == 1
        assert result.percentage_score == 100.0


class TestCompositeResult:
    """Tests for CompositeResult dataclass."""

    def test_composite_result_defaults(self):
        """Test CompositeResult default values."""
        result = CompositeResult(is_correct=True, grade=ScoreResult.CORRECT)
        assert result.component_scores == {}
        assert result.component_weights == {}

    def test_composite_result_with_components(self):
        """Test CompositeResult with component scores."""
        result = CompositeResult(
            is_correct=True,
            grade=ScoreResult.CORRECT,
            component_scores={"race": 80.0, "fact": 60.0},
            component_weights={"race": 0.6, "fact": 0.4},
        )
        assert result.component_scores["race"] == 80.0
        assert result.component_weights["fact"] == 0.4


class TestBaseScorer:
    """Tests for BaseScorer abstract class."""

    def test_base_scorer_defaults(self):
        """Test BaseScorer class-level defaults."""
        assert BaseScorer.DEFAULT_MODEL == "gpt-5-mini-2025-08-07"
        assert BaseScorer.DEFAULT_TEMPERATURE == 0.0
        assert BaseScorer.DEFAULT_MAX_RETRIES == 3

    def test_normalize_str(self):
        """Test string normalization."""
        assert BaseScorer.normalize_str("Hello World") == "helloworld"
        assert BaseScorer.normalize_str("Hello, World!") == "helloworld"
        assert BaseScorer.normalize_str("  Spaces  ") == "spaces"

    def test_normalize_str_keep_punct(self):
        """Test string normalization with punctuation preserved."""
        assert BaseScorer.normalize_str("Hello, World!", remove_punct=False) == "hello,world!"

    def test_normalize_number(self):
        """Test number normalization."""
        assert BaseScorer.normalize_number("$1,000") == 1000.0
        assert BaseScorer.normalize_number("50%") == 50.0
        assert BaseScorer.normalize_number("1st") == 1.0
        assert BaseScorer.normalize_number("€100") == 100.0

    def test_normalize_number_invalid(self):
        """Test number normalization with invalid input."""
        assert BaseScorer.normalize_number("not a number") == float("inf")

    def test_is_float(self):
        """Test float detection."""
        assert BaseScorer.is_float("3.14") is True
        assert BaseScorer.is_float("42") is True
        assert BaseScorer.is_float("hello") is False
        assert BaseScorer.is_float(None) is False


class TestLLMJudgeScorer:
    """Tests for LLMJudgeScorer base class."""

    def test_parse_grade_correct(self):
        """Test grade parsing for CORRECT."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("CORRECT") == ScoreResult.CORRECT
        assert scorer._parse_grade("The answer is CORRECT") == ScoreResult.CORRECT
        assert scorer._parse_grade("correct") == ScoreResult.CORRECT

    def test_parse_grade_incorrect(self):
        """Test grade parsing for INCORRECT."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("INCORRECT") == ScoreResult.INCORRECT
        assert scorer._parse_grade("This is INCORRECT because...") == ScoreResult.INCORRECT

    def test_parse_grade_not_attempted(self):
        """Test grade parsing for NOT_ATTEMPTED."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("NOT_ATTEMPTED") == ScoreResult.NOT_ATTEMPTED
        assert scorer._parse_grade("NOT ATTEMPTED") == ScoreResult.NOT_ATTEMPTED

    def test_parse_grade_letter(self):
        """Test grade parsing from letter grades."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("A") == ScoreResult.CORRECT
        assert scorer._parse_grade("B") == ScoreResult.INCORRECT
        assert scorer._parse_grade("C") == ScoreResult.NOT_ATTEMPTED

    def test_parse_grade_ambiguous(self):
        """Test grade parsing falls back to NOT_ATTEMPTED."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("Unknown response") == ScoreResult.NOT_ATTEMPTED


class TestGetScorer:
    """Tests for get_scorer factory function."""

    def test_get_scorer_hle(self):
        """Test getting HLE scorer."""
        scorer = get_scorer("hle")
        assert scorer is not None
        # HLE uses SimpleQAScorer
        assert "SimpleQAScorer" in type(scorer).__name__

    def test_get_scorer_gaia(self):
        """Test getting GAIA scorer."""
        scorer = get_scorer("gaia")
        assert scorer is not None
        assert "GAIAScorer" in type(scorer).__name__

    def test_get_scorer_simpleqa(self):
        """Test getting SimpleQA scorer."""
        scorer = get_scorer("simpleqa")
        assert scorer is not None
        assert "SimpleQAScorer" in type(scorer).__name__

    def test_get_scorer_browsecomp(self):
        """Test getting BrowseComp scorer."""
        scorer = get_scorer("browsecomp")
        assert scorer is not None
        assert "BrowseCompScorer" in type(scorer).__name__

    def test_get_scorer_deepresearch(self):
        """Test getting DeepResearch scorer."""
        scorer = get_scorer("deepresearch")
        assert scorer is not None
        assert "DeepResearchScorer" in type(scorer).__name__

    def test_get_scorer_apex(self):
        """Test getting APEX scorer."""
        scorer = get_scorer("apex")
        assert scorer is not None
        assert "APEXScorer" in type(scorer).__name__

    def test_get_scorer_unknown_fallback(self):
        """Test unknown benchmark falls back to SimpleQAScorer."""
        scorer = get_scorer("unknown_benchmark")
        assert scorer is not None
        # Should fall back to SimpleQAScorer
        assert "SimpleQAScorer" in type(scorer).__name__

    def test_get_scorer_with_model(self):
        """Test get_scorer with custom model."""
        scorer = get_scorer("hle", model="gpt-5-mini-2025-08-07")
        assert scorer.model == "gpt-5-mini-2025-08-07"

    def test_get_scorer_with_api_key(self):
        """Test get_scorer with API key."""
        scorer = get_scorer("hle", api_key="test-key")
        assert scorer.api_key == "test-key"

    def test_get_scorer_case_insensitive(self):
        """Test benchmark name is case-insensitive."""
        scorer_lower = get_scorer("gaia")
        scorer_upper = get_scorer("GAIA")
        assert type(scorer_lower).__name__ == type(scorer_upper).__name__


class TestGAIAScorer:
    """Tests for GAIA scorer exact match logic."""

    def test_exact_match_number(self):
        """Test GAIA exact match for numbers."""
        from evals.benchmarks.gaia.scorer import GAIAScorer

        scorer = GAIAScorer(use_llm_fallback=False)
        result = scorer.score_sync("What is 2+2?", "4", "4")
        assert result.is_correct is True
        assert result.grade == ScoreResult.CORRECT

    def test_exact_match_string(self):
        """Test GAIA exact match for strings."""
        from evals.benchmarks.gaia.scorer import GAIAScorer

        scorer = GAIAScorer(use_llm_fallback=False)
        result = scorer.score_sync("Capital?", "paris", "Paris")
        assert result.is_correct is True

    def test_exact_match_with_punctuation(self):
        """Test GAIA exact match with punctuation differences."""
        from evals.benchmarks.gaia.scorer import GAIAScorer

        scorer = GAIAScorer(use_llm_fallback=False)
        result = scorer.score_sync("Name?", "John Doe.", "John Doe")
        assert result.is_correct is True

    def test_empty_response(self):
        """Test GAIA scorer with empty response."""
        from evals.benchmarks.gaia.scorer import GAIAScorer

        scorer = GAIAScorer(use_llm_fallback=False)
        result = scorer.score_sync("Question?", "", "Answer")
        assert result.is_correct is False
        assert result.grade == ScoreResult.NOT_ATTEMPTED


class TestStructuredResponseParsing:
    """Tests for structured 'correct: yes/no/not_attempted' response parsing."""

    def test_parse_grade_structured_correct(self):
        """Test 'correct: yes' parses as CORRECT."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("correct: yes") == ScoreResult.CORRECT

    def test_parse_grade_structured_incorrect(self):
        """Test 'correct: no' parses as INCORRECT."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("correct: no") == ScoreResult.INCORRECT

    def test_parse_grade_structured_not_attempted(self):
        """Test 'correct: not_attempted' parses as NOT_ATTEMPTED."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("correct: not_attempted") == ScoreResult.NOT_ATTEMPTED

    def test_parse_grade_structured_full_response(self):
        """Test full structured response with all three fields parses correctly."""
        scorer = _create_test_llm_judge_scorer()
        full_response = (
            "extracted_final_answer: Paris\n"
            "reasoning: The extracted answer 'Paris' matches the ground truth 'Paris'.\n"
            "correct: yes"
        )
        assert scorer._parse_grade(full_response) == ScoreResult.CORRECT

        full_response_no = (
            "extracted_final_answer: London\n"
            "reasoning: The extracted answer 'London' does not match 'Paris'.\n"
            "correct: no"
        )
        assert scorer._parse_grade(full_response_no) == ScoreResult.INCORRECT

    def test_parse_grade_backwards_compat(self):
        """Test bare 'CORRECT' / 'INCORRECT' still work (backwards compat)."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("CORRECT") == ScoreResult.CORRECT
        assert scorer._parse_grade("INCORRECT") == ScoreResult.INCORRECT
        assert scorer._parse_grade("NOT_ATTEMPTED") == ScoreResult.NOT_ATTEMPTED

    def test_parse_grade_structured_case_insensitive(self):
        """Test structured parsing is case-insensitive."""
        scorer = _create_test_llm_judge_scorer()
        assert scorer._parse_grade("Correct: Yes") == ScoreResult.CORRECT
        assert scorer._parse_grade("CORRECT: NO") == ScoreResult.INCORRECT
        assert scorer._parse_grade("correct: NOT_ATTEMPTED") == ScoreResult.NOT_ATTEMPTED


class TestSimpleQAStructuredParsing:
    """Tests for SimpleQA scorer's structured response parsing."""

    def test_simpleqa_parse_grade_structured_correct(self):
        """Test SimpleQA _parse_grade with 'correct: yes'."""
        from evals.benchmarks.simpleqa.scorer import _parse_grade
        assert _parse_grade("correct: yes") == "CORRECT"

    def test_simpleqa_parse_grade_structured_incorrect(self):
        """Test SimpleQA _parse_grade with 'correct: no'."""
        from evals.benchmarks.simpleqa.scorer import _parse_grade
        assert _parse_grade("correct: no") == "INCORRECT"

    def test_simpleqa_parse_grade_structured_not_attempted(self):
        """Test SimpleQA _parse_grade with 'correct: not_attempted'."""
        from evals.benchmarks.simpleqa.scorer import _parse_grade
        assert _parse_grade("correct: not_attempted") == "NOT_ATTEMPTED"

    def test_simpleqa_parse_grade_full_structured(self):
        """Test SimpleQA _parse_grade with full structured response."""
        from evals.benchmarks.simpleqa.scorer import _parse_grade
        full_response = (
            "extracted_final_answer: Malia and Sasha Obama\n"
            "reasoning: The extracted answer matches the gold target.\n"
            "correct: yes"
        )
        assert _parse_grade(full_response) == "CORRECT"

    def test_simpleqa_parse_grade_backwards_compat(self):
        """Test SimpleQA _parse_grade backwards compat with bare keywords."""
        from evals.benchmarks.simpleqa.scorer import _parse_grade
        assert _parse_grade("CORRECT") == "CORRECT"
        assert _parse_grade("INCORRECT") == "INCORRECT"
        assert _parse_grade("NOT_ATTEMPTED") == "NOT_ATTEMPTED"


class TestFRAMESStructuredParsing:
    """Tests for FRAMES scorer's structured response parsing."""

    @staticmethod
    def _get_frames_parse_grade():
        """Import _parse_grade directly from frames scorer file, bypassing __init__.py."""
        import importlib.util
        from pathlib import Path
        scorer_path = Path(__file__).resolve().parents[2] / "src" / "evals" / "benchmarks" / "frames" / "scorer.py"
        spec = importlib.util.spec_from_file_location("frames_scorer", scorer_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._parse_grade

    def test_frames_parse_grade_structured_correct(self):
        """Test FRAMES _parse_grade with 'correct: yes'."""
        _parse_grade = self._get_frames_parse_grade()
        assert _parse_grade("correct: yes") is True

    def test_frames_parse_grade_structured_incorrect(self):
        """Test FRAMES _parse_grade with 'correct: no'."""
        _parse_grade = self._get_frames_parse_grade()
        assert _parse_grade("correct: no") is False

    def test_frames_parse_grade_backwards_compat(self):
        """Test FRAMES _parse_grade backwards compat with TRUE/FALSE."""
        _parse_grade = self._get_frames_parse_grade()
        assert _parse_grade("TRUE") is True
        assert _parse_grade("FALSE") is False


class TestSimpleQAScorer:
    """Tests for SimpleQA scorer."""

    def test_scorer_initialization(self):
        """Test SimpleQA scorer initialization."""
        from evals.benchmarks.simpleqa.scorer import SimpleQAScorer

        scorer = SimpleQAScorer(model="gpt-5-mini-2025-08-07")
        assert scorer.model == "gpt-5-mini-2025-08-07"

    def test_scorer_default_model(self):
        """Test SimpleQA scorer uses default model."""
        from evals.benchmarks.simpleqa.scorer import SimpleQAScorer

        scorer = SimpleQAScorer()
        assert scorer.model == "gpt-5-mini-2025-08-07"  # BaseScorer default


# =============================================================================
# Helper functions
# =============================================================================


def _create_test_llm_judge_scorer():
    """Create a test LLMJudgeScorer instance."""

    class TestLLMJudgeScorer(LLMJudgeScorer):
        JUDGE_PROMPT_TEMPLATE = "Test prompt: {question} {response} {ground_truth}"

        async def score(self, question, response, ground_truth, **kwargs):
            return ScorerResult(is_correct=True, grade=ScoreResult.CORRECT)

    return TestLLMJudgeScorer()
