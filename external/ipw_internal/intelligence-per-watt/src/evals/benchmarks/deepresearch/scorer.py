# benchmarks/deepresearch/scorer.py
"""
DeepResearchBench scoring utilities.

Implements RACE and FACT evaluation metrics following the DeepResearchBench paper:
- RACE: Reference-based Adaptive Criteria-driven Evaluation
- FACT: Framework for Factual Abundance and Citation Trustworthiness

Reference: https://github.com/Ayanami0730/deep_research_bench
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from evals.benchmarks.scorer import (
    BaseScorer,
    CompositeScorer,
    CompositeResult,
    ScoreResult,
    ScorerResult,
    register_scorer,
)

logger = logging.getLogger(__name__)

# Default API keys from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@dataclass
class RACEScore:
    """RACE (Reference-based Adaptive Criteria-driven Evaluation) scores."""

    comprehensiveness: float = 0.0  # 0-10 scale
    insight_depth: float = 0.0      # 0-10 scale
    instruction_following: float = 0.0  # 0-10 scale
    readability: float = 0.0        # 0-10 scale
    overall: float = 0.0            # Average of above
    raw_response: str = ""
    error: Optional[str] = None


@dataclass
class FACTScore:
    """FACT (Framework for Factual Abundance and Citation Trustworthiness) scores."""

    total_claims: int = 0
    verified_claims: int = 0
    total_citations: int = 0
    valid_citations: int = 0
    citation_accuracy: float = 0.0    # valid_citations / total_citations
    effective_citations: float = 0.0  # verified_claims / total_claims
    claim_details: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DeepResearchScore:
    """Combined score for a DeepResearchBench sample."""

    race: RACEScore
    fact: FACTScore
    overall_score: float = 0.0  # Weighted combination


# ============================================================================
# Gemini API Client
# ============================================================================

async def _call_gemini_async(
    prompt: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
) -> str:
    """Call the Gemini API asynchronously.

    Args:
        prompt: The prompt to send
        model: Gemini model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: API key (defaults to GEMINI_API_KEY env var)

    Returns:
        Generated text response
    """
    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Get an API key at https://makersuite.google.com/app/apikey"
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            params={"key": api_key},
            json=data,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Gemini API error {response.status}: {error_text}")

            result = await response.json()

            # Extract text from response
            try:
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
            except (KeyError, IndexError):
                pass

            return ""


# ============================================================================
# Jina API Client (for URL fetching)
# ============================================================================

async def _fetch_url_content(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[bool, str]:
    """Fetch URL content using Jina Reader API.

    Args:
        url: URL to fetch
        api_key: Jina API key
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, content_or_error)
    """
    api_key = api_key or JINA_API_KEY
    if not api_key:
        # Fall back to direct fetch without Jina
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Truncate to reasonable length
                        return True, text[:10000]
                    else:
                        return False, f"HTTP {response.status}"
        except Exception as e:
            return False, str(e)

    # Use Jina Reader API
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/plain",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                jina_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    return True, text[:10000]  # Truncate for LLM context
                else:
                    error_text = await response.text()
                    return False, f"Jina API error {response.status}: {error_text[:100]}"
    except asyncio.TimeoutError:
        return False, "Request timeout"
    except Exception as e:
        return False, str(e)


# ============================================================================
# RACE Scoring
# ============================================================================

RACE_EVALUATION_PROMPT = """You are an expert evaluator assessing a research report.

## Research Question
{query}

## Research Domain
{domain}

## Research Report to Evaluate
{article}

## Evaluation Criteria

Please evaluate the report on the following dimensions (score 0-10 for each):

1. **Comprehensiveness** (0-10): Does the report thoroughly cover the topic? Does it address multiple relevant aspects and perspectives?

2. **Insight/Depth** (0-10): Does the report provide deep analysis rather than surface-level information? Are there novel insights or connections?

3. **Instruction Following** (0-10): Does the report address the original research question? Does it follow the expected format with proper citations?

4. **Readability** (0-10): Is the report well-organized and clearly written? Is it easy to follow the logical flow?

## Response Format

Provide your evaluation in the following exact format:
COMPREHENSIVENESS: [score]
INSIGHT_DEPTH: [score]
INSTRUCTION_FOLLOWING: [score]
READABILITY: [score]
REASONING: [brief explanation]"""


async def score_race_async(
    query: str,
    domain: str,
    article: str,
    gemini_api_key: Optional[str] = None,
) -> RACEScore:
    """
    Score a research report using RACE evaluation.

    Args:
        query: Original research question
        domain: Research domain
        article: Generated research report
        gemini_api_key: Optional Gemini API key

    Returns:
        RACEScore with dimension scores and overall
    """
    if not article or not article.strip():
        return RACEScore(error="Empty article")

    prompt = RACE_EVALUATION_PROMPT.format(
        query=query,
        domain=domain,
        article=article[:15000],  # Truncate for context limit
    )

    try:
        response = await _call_gemini_async(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
            api_key=gemini_api_key,
        )

        # Parse scores from response
        scores = {
            "comprehensiveness": 0.0,
            "insight_depth": 0.0,
            "instruction_following": 0.0,
            "readability": 0.0,
        }

        patterns = {
            "comprehensiveness": r"COMPREHENSIVENESS:\s*(\d+(?:\.\d+)?)",
            "insight_depth": r"INSIGHT_DEPTH:\s*(\d+(?:\.\d+)?)",
            "instruction_following": r"INSTRUCTION_FOLLOWING:\s*(\d+(?:\.\d+)?)",
            "readability": r"READABILITY:\s*(\d+(?:\.\d+)?)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[key] = min(10.0, max(0.0, float(match.group(1))))

        overall = sum(scores.values()) / 4.0

        return RACEScore(
            comprehensiveness=scores["comprehensiveness"],
            insight_depth=scores["insight_depth"],
            instruction_following=scores["instruction_following"],
            readability=scores["readability"],
            overall=round(overall, 2),
            raw_response=response,
        )

    except Exception as e:
        logger.error(f"RACE scoring failed: {e}")
        return RACEScore(error=str(e))


# ============================================================================
# FACT Scoring
# ============================================================================

CLAIM_EXTRACTION_PROMPT = """Extract factual claims from the following research report that have citations.

For each claim, identify:
1. The factual statement being made
2. The citation number(s) referenced (e.g., [1], [2])

## Research Report
{article}

## Response Format
List each claim in this format:
CLAIM: [the factual statement]
CITATIONS: [citation numbers, e.g., 1, 2]
---
(repeat for each claim)

Only extract claims that have explicit citations. Limit to the 10 most important claims."""


CITATION_VERIFICATION_PROMPT = """Verify whether the following source content supports the claimed fact.

## Claimed Fact
{claim}

## Source Content
{source_content}

## Task
Determine if the source content provides evidence that supports or contradicts the claim.

Respond with exactly one of:
SUPPORTED - The source clearly supports this claim
PARTIALLY_SUPPORTED - The source provides some relevant evidence but not complete confirmation
NOT_SUPPORTED - The source does not support or contradicts this claim
INSUFFICIENT - Cannot determine from the source content"""


async def _extract_citations_from_article(article: str) -> List[Tuple[str, str]]:
    """Extract citation URLs from the article's references section.

    Returns list of (citation_number, url) tuples.
    """
    citations = []

    # Look for references section
    ref_patterns = [
        r"(?:References|Bibliography|Sources|Citations)\s*:?\s*\n([\s\S]+?)(?:\n\n|\Z)",
        r"\[(\d+)\]\s*(https?://[^\s\n]+)",
    ]

    # Try to find numbered references with URLs
    url_pattern = re.compile(r"\[(\d+)\][^\[]*?(https?://[^\s\n\]]+)", re.MULTILINE)
    for match in url_pattern.finditer(article):
        num = match.group(1)
        url = match.group(2).rstrip(".,;:)")
        citations.append((num, url))

    # Also try markdown-style links
    md_pattern = re.compile(r"\[(\d+)\]:\s*(https?://[^\s\n]+)")
    for match in md_pattern.finditer(article):
        num = match.group(1)
        url = match.group(2).rstrip(".,;:)")
        if (num, url) not in citations:
            citations.append((num, url))

    return citations


async def score_fact_async(
    article: str,
    gemini_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    max_claims_to_verify: int = 5,
) -> FACTScore:
    """
    Score a research report using FACT evaluation.

    Args:
        article: Generated research report
        gemini_api_key: Optional Gemini API key
        jina_api_key: Optional Jina API key for URL fetching
        max_claims_to_verify: Maximum number of claims to verify

    Returns:
        FACTScore with citation accuracy and claim verification metrics
    """
    if not article or not article.strip():
        return FACTScore(error="Empty article")

    try:
        # Step 1: Extract citation URLs from article
        citations = await _extract_citations_from_article(article)
        total_citations = len(citations)

        if total_citations == 0:
            return FACTScore(
                total_claims=0,
                verified_claims=0,
                total_citations=0,
                valid_citations=0,
                citation_accuracy=0.0,
                effective_citations=0.0,
                error="No citations found in article",
            )

        # Step 2: Extract claims with citations using LLM
        extract_prompt = CLAIM_EXTRACTION_PROMPT.format(article=article[:10000])
        claims_response = await _call_gemini_async(
            prompt=extract_prompt,
            temperature=0.0,
            max_tokens=2000,
            api_key=gemini_api_key,
        )

        # Parse claims
        claims = []
        claim_blocks = claims_response.split("---")
        for block in claim_blocks:
            claim_match = re.search(r"CLAIM:\s*(.+?)(?=CITATIONS:|$)", block, re.DOTALL)
            citation_match = re.search(r"CITATIONS:\s*(.+)", block)
            if claim_match:
                claim_text = claim_match.group(1).strip()
                citation_nums = []
                if citation_match:
                    citation_nums = re.findall(r"\d+", citation_match.group(1))
                claims.append({
                    "claim": claim_text,
                    "citation_nums": citation_nums,
                    "verified": False,
                    "source_accessible": False,
                })

        total_claims = len(claims)

        # Step 3: Verify claims by fetching sources and checking
        valid_citations = 0
        verified_claims = 0
        citation_to_content: Dict[str, str] = {}

        # Fetch unique citation URLs
        claims_to_verify = claims[:max_claims_to_verify]
        for claim_data in claims_to_verify:
            claim_verified = False
            for cit_num in claim_data["citation_nums"]:
                # Find URL for this citation
                url = None
                for stored_num, stored_url in citations:
                    if stored_num == cit_num:
                        url = stored_url
                        break

                if not url:
                    continue

                # Fetch content if not cached
                if url not in citation_to_content:
                    success, content = await _fetch_url_content(url, api_key=jina_api_key)
                    if success:
                        citation_to_content[url] = content
                        valid_citations += 1
                    else:
                        citation_to_content[url] = ""

                # Verify claim against source
                source_content = citation_to_content.get(url, "")
                if source_content:
                    claim_data["source_accessible"] = True
                    verify_prompt = CITATION_VERIFICATION_PROMPT.format(
                        claim=claim_data["claim"],
                        source_content=source_content[:5000],
                    )

                    verify_response = await _call_gemini_async(
                        prompt=verify_prompt,
                        temperature=0.0,
                        max_tokens=1024,
                        api_key=gemini_api_key,
                    )

                    if "SUPPORTED" in verify_response.upper():
                        claim_verified = True
                        break

            if claim_verified:
                claim_data["verified"] = True
                verified_claims += 1

        # Calculate metrics
        # Use total unique citations found, not just ones we tried to fetch
        citation_accuracy = (valid_citations / total_citations) if total_citations > 0 else 0.0
        effective_citations = (verified_claims / total_claims) if total_claims > 0 else 0.0

        return FACTScore(
            total_claims=total_claims,
            verified_claims=verified_claims,
            total_citations=total_citations,
            valid_citations=valid_citations,
            citation_accuracy=round(citation_accuracy, 3),
            effective_citations=round(effective_citations, 3),
            claim_details=claims_to_verify,
        )

    except Exception as e:
        logger.error(f"FACT scoring failed: {e}")
        return FACTScore(error=str(e))


# ============================================================================
# Combined Scoring
# ============================================================================

async def score_deepresearch_async(
    query: str,
    domain: str,
    article: str,
    gemini_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    race_weight: float = 0.6,
    fact_weight: float = 0.4,
) -> DeepResearchScore:
    """
    Score a research report using both RACE and FACT evaluation.

    Args:
        query: Original research question
        domain: Research domain
        article: Generated research report
        gemini_api_key: Optional Gemini API key
        jina_api_key: Optional Jina API key
        race_weight: Weight for RACE score in overall (default 0.6)
        fact_weight: Weight for FACT score in overall (default 0.4)

    Returns:
        DeepResearchScore with RACE, FACT, and overall scores
    """
    # Run RACE and FACT scoring concurrently
    race_task = score_race_async(query, domain, article, gemini_api_key)
    fact_task = score_fact_async(article, gemini_api_key, jina_api_key)

    race_score, fact_score = await asyncio.gather(race_task, fact_task)

    # Calculate overall score
    # RACE: 0-10 scale, normalize to 0-1
    race_normalized = race_score.overall / 10.0 if race_score.overall else 0.0

    # FACT: already 0-1 scale (average of citation_accuracy and effective_citations)
    fact_normalized = (fact_score.citation_accuracy + fact_score.effective_citations) / 2.0

    overall = (race_weight * race_normalized + fact_weight * fact_normalized) * 100

    return DeepResearchScore(
        race=race_score,
        fact=fact_score,
        overall_score=round(overall, 2),
    )


def score_deepresearch(
    query: str,
    domain: str,
    article: str,
    gemini_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
) -> DeepResearchScore:
    """Synchronous wrapper for score_deepresearch_async."""
    return asyncio.run(score_deepresearch_async(
        query=query,
        domain=domain,
        article=article,
        gemini_api_key=gemini_api_key,
        jina_api_key=jina_api_key,
    ))


# =============================================================================
# Unified Scorer Class
# =============================================================================


@register_scorer("deepresearch")
class DeepResearchScorer(CompositeScorer):
    """Composite scorer using RACE + FACT evaluation for DeepResearchBench.

    This scorer evaluates research reports using two complementary metrics:
    - RACE (60% weight): Evaluates comprehensiveness, insight depth, instruction
      following, and readability on a 0-10 scale.
    - FACT (40% weight): Evaluates citation accuracy and effective citations
      by verifying claims against source URLs.

    The overall score is considered passing (correct) if >= 50%.

    Example:
        >>> scorer = DeepResearchScorer()
        >>> result = await scorer.score(
        ...     question="What are the effects of climate change on coral reefs?",
        ...     response="<research report with citations>",
        ...     ground_truth="",
        ...     domain="environmental_science"
        ... )
        >>> print(result.is_correct, result.score)
    """

    COMPONENT_WEIGHTS = {
        "race": 0.6,
        "fact": 0.4,
    }
    PASSING_THRESHOLD = 50.0

    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the DeepResearch scorer.

        Args:
            model: Model for scoring (defaults to gemini-2.0-flash for RACE)
            api_key: Gemini API key
            jina_api_key: Jina API key for URL fetching
            **kwargs: Additional arguments passed to CompositeScorer
        """
        # DeepResearch uses Gemini by default
        super().__init__(model=model or "gemini-2.0-flash", api_key=api_key, **kwargs)
        self.jina_api_key = jina_api_key

    async def _score_components(
        self,
        question: str,
        response: str,
        **kwargs,
    ) -> Dict[str, float]:
        """Score RACE and FACT components.

        Args:
            question: The research question
            response: The research report
            **kwargs: Additional arguments (domain)

        Returns:
            Dict with 'race' and 'fact' scores (0-100)
        """
        domain = kwargs.get("domain", "general")

        # Run RACE and FACT scoring concurrently
        race_score, fact_score = await asyncio.gather(
            score_race_async(
                query=question,
                domain=domain,
                article=response,
                gemini_api_key=self.api_key or os.environ.get("GEMINI_API_KEY"),
            ),
            score_fact_async(
                article=response,
                gemini_api_key=self.api_key or os.environ.get("GEMINI_API_KEY"),
                jina_api_key=self.jina_api_key or os.environ.get("JINA_API_KEY"),
            ),
        )

        # Normalize RACE (0-10 scale) to 0-100
        race_normalized = (race_score.overall / 10.0) * 100 if race_score.overall else 0.0

        # FACT: average of citation_accuracy and effective_citations (0-1 scale) to 0-100
        fact_normalized = (
            (fact_score.citation_accuracy + fact_score.effective_citations) / 2.0
        ) * 100

        return {
            "race": race_normalized,
            "fact": fact_normalized,
        }

    async def score(
        self,
        question: str,
        response: str,
        ground_truth: str = "",
        **kwargs,
    ) -> CompositeResult:
        """Score a research report using RACE + FACT evaluation.

        Args:
            question: The research question
            response: The research report to evaluate
            ground_truth: Not used for DeepResearch (empty string)
            **kwargs: Additional arguments (domain required)

        Returns:
            CompositeResult with RACE and FACT component scores
        """
        if not response or not response.strip():
            return CompositeResult(
                is_correct=False,
                grade=ScoreResult.NOT_ATTEMPTED,
                explanation="Empty response",
            )

        try:
            domain = kwargs.get("domain", "general")
            component_scores = await self._score_components(question, response, domain=domain)

            # Calculate weighted average
            weights = self.COMPONENT_WEIGHTS
            total_weight = sum(weights.values())
            weighted_sum = sum(
                component_scores[k] * weights[k]
                for k in component_scores
            )
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0

            is_correct = overall_score >= self.PASSING_THRESHOLD
            grade = ScoreResult.CORRECT if is_correct else ScoreResult.INCORRECT

            return CompositeResult(
                is_correct=is_correct,
                grade=grade,
                score=overall_score / 100,
                explanation=f"Overall: {overall_score:.1f}% (RACE: {component_scores['race']:.1f}%, FACT: {component_scores['fact']:.1f}%)",
                component_scores=component_scores,
                component_weights=weights,
            )

        except Exception as e:
            logger.error(f"DeepResearch scoring failed: {e}")
            return CompositeResult(
                is_correct=False,
                grade=ScoreResult.ERROR,
                error=str(e),
            )
