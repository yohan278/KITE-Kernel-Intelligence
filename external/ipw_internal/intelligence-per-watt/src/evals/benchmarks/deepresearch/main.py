# benchmarks/deepresearch/main.py
"""
DeepResearchBench Benchmark.

Evaluates AI models on PhD-level research report generation with citations.
100 tasks across 22 domains (50 EN + 50 ZH).

Uses RACE + FACT evaluation via Gemini API following the official
DeepResearchBench evaluation approach.

Supports retrieval augmentation:
- retrieval_method="bm25"|"dense"|"hybrid": Use RAG from domain corpus
- Retrieval can provide domain-specific reference materials

Reference: https://github.com/Ayanami0730/deep_research_bench
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.retrieval import (
    BenchmarkRetrievalMixin,
    RetrievalConfig,
    RetrievalMethod,
    load_deepresearch_corpus,
)
from evals.benchmarks.deepresearch.dataset import (
    DeepResearchSample,
    load_deepresearch_samples,
)
from evals.benchmarks.deepresearch.scorer import (
    score_deepresearch_async,
    DeepResearchScore,
    RACEScore,
    FACTScore,
)


@dataclass
class DeepResearchResult:
    """Result from evaluating a single DeepResearch sample."""

    task_id: str
    query: str
    domain: str
    language: str
    article: str                   # Generated research report
    race_score: RACEScore
    fact_score: FACTScore
    overall_score: float
    response_time_seconds: float = 0.0
    grading_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeepResearchBenchmark(BenchmarkRetrievalMixin, DatasetBenchmark):
    """
    DeepResearchBench benchmark for evaluating research report generation.

    This benchmark:
    1. Loads research tasks from the deep_research_bench dataset
    2. Optionally loads domain corpus for RAG augmentation
    3. Generates model responses (research reports) using the provided orchestrator
    4. Grades responses using RACE + FACT evaluation via Gemini

    Retrieval note:
    DeepResearch tasks typically require web search for citations.
    Retrieval here provides domain background context, not citation sources.
    For full functionality, use with an orchestrator that has web search tools.

    Example usage:
        >>> # Basic usage
        >>> benchmark = DeepResearchBenchmark(limit=10)
        >>> results = benchmark.run_benchmark(orchestrator)

        >>> # With domain retrieval
        >>> benchmark = DeepResearchBenchmark(
        ...     retrieval_method="bm25",
        ...     limit=10,
        ... )
        >>> results = benchmark.run_benchmark(orchestrator)
    """

    def __init__(
        self,
        *,
        # Dataset options
        language: Optional[str] = None,
        domains: Optional[List[str]] = None,
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        input_prompt: Optional[str] = None,
        # Retrieval options
        retrieval_method: RetrievalMethod = "none",
        retrieval_top_k: int = 5,
        retrieval_cache_dir: Optional[Path] = None,
        telemetry_collector: Optional[Any] = None,
        # Grading options
        gemini_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        max_concurrent_grading: int = 5,
        race_weight: float = 0.6,
        fact_weight: float = 0.4,
        # Execution options
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DeepResearchBench benchmark.

        Args:
            language: Filter by language ("en" or "zh"), None for all
            domains: Filter to specific domains (e.g., ["Physics", "Chemistry"])
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            limit: Maximum number of samples to evaluate
            input_prompt: Custom input prompt template
            retrieval_method: "none", "bm25", "dense", "hybrid" (default: "none")
            retrieval_top_k: Number of documents to retrieve (default: 5)
            retrieval_cache_dir: Cache directory for retrieval index
            telemetry_collector: Energy monitor collector for telemetry
            gemini_api_key: API key for Gemini (for RACE + FACT evaluation)
            jina_api_key: API key for Jina (for citation URL fetching)
            max_concurrent_grading: Max concurrent grading calls (default: 5)
            race_weight: Weight for RACE score in overall (default: 0.6)
            fact_weight: Weight for FACT score in overall (default: 0.4)
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=None)

        # Dataset config
        self.language = language
        self.domains = domains
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.input_prompt = input_prompt

        # Retrieval config
        self.retrieval_method = retrieval_method
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_cache_dir = retrieval_cache_dir
        self.telemetry_collector = telemetry_collector
        self._retrieval_initialized = False

        # Grading config
        self.gemini_api_key = gemini_api_key
        self.jina_api_key = jina_api_key
        self.max_concurrent_grading = max_concurrent_grading
        self.race_weight = race_weight
        self.fact_weight = fact_weight

        # Results storage
        self._samples: List[DeepResearchSample] = []
        self._results: Dict[str, DeepResearchResult] = {}

    def _load_samples(self) -> List[DeepResearchSample]:
        """Load samples from the dataset."""
        if not self._samples:
            self._samples = list(load_deepresearch_samples(
                language=self.language,
                domains=self.domains,
                shuffle=self.shuffle,
                seed=self.seed,
                limit=self.limit,
            ))
            self.logger.info(f"Loaded {len(self._samples)} DeepResearch samples")
            if self.language:
                self.logger.info(f"Language filter: {self.language}")
            if self.domains:
                self.logger.info(f"Domain filter: {self.domains}")

        return self._samples

    def _init_retrieval_if_needed(self) -> None:
        """Initialize retrieval system if configured and not already done."""
        if self._retrieval_initialized:
            return
        self._retrieval_initialized = True

        if self.retrieval_method == "none":
            return

        # Load samples first (needed for corpus)
        samples = self._load_samples()

        # Create retrieval config
        config = RetrievalConfig(
            method=self.retrieval_method,
            top_k=self.retrieval_top_k,
            cache_dir=self.retrieval_cache_dir,
            telemetry_collector=self.telemetry_collector,
        )

        # Load corpus from samples
        documents = load_deepresearch_corpus(
            samples=samples,
            domains=self.domains,
            fetch_sources=False,  # Don't fetch external sources by default
        )

        if not documents:
            self.logger.warning(
                "No documents loaded for DeepResearch retrieval. "
                "DeepResearch typically requires web search for citations."
            )
            return

        # Initialize retrieval
        self.init_retrieval(config, documents, benchmark_name="deepresearch")

    def _build_prompt_with_retrieval(self, sample: DeepResearchSample) -> str:
        """Build prompt with retrieved context.

        Args:
            sample: DeepResearch sample

        Returns:
            Prompt with retrieved context
        """
        # Get base prompt
        base_prompt = sample.get_prompt(self.input_prompt)

        # If retrieval is enabled, augment with retrieved context
        if self.retrieval_enabled:
            results = self.search_retrieval(sample.query, top_k=self.retrieval_top_k)
            if results:
                context = self._retrieval.format_context(results, include_scores=False)
                # Insert context as background information
                context_section = (
                    "Here is some background information on the topic:\n\n"
                    f"{context}\n\n"
                    "Note: You should still conduct your own research and find "
                    "authoritative sources for citations.\n\n"
                )
                return f"{context_section}{base_prompt}"

        return base_prompt

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from orchestrator response."""
        if response is None:
            return ""

        if hasattr(response, "content"):
            content = getattr(response, "content", None)
            if content is not None:
                return str(content)

        return str(response)

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """
        Generate model responses for all DeepResearch samples.

        Args:
            orchestrator: An orchestrator instance with a run() method.

        Returns:
            Dict mapping task_id to response data including raw response and timing.
        """
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        # Initialize retrieval if needed
        self._init_retrieval_if_needed()

        self.logger.info(f"Generating responses for {len(samples)} DeepResearch samples")
        if self.retrieval_method != "none":
            self.logger.info(
                f"Retrieval: {self.retrieval_method} (top_k={self.retrieval_top_k})"
            )

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing task {sample.task_id} "
                f"(domain: {sample.domain}, lang: {sample.language})"
            )

            start_time = time.time()
            try:
                # Use retrieval-augmented prompt if enabled
                if self.retrieval_method != "none":
                    prompt = self._build_prompt_with_retrieval(sample)
                else:
                    prompt = sample.get_prompt(self.input_prompt)
                response = orchestrator.run(prompt)
                response_text = self._extract_response_text(response)

                elapsed = time.time() - start_time
                self.logger.info(
                    f"[{idx}/{len(samples)}] Generated response in {elapsed:.2f}s "
                    f"({len(response_text)} chars)"
                )

                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "query": sample.query,
                    "domain": sample.domain,
                    "language": sample.language,
                    "article": response_text,
                    "response_time_seconds": elapsed,
                    "error": None,
                    "metadata": sample.metadata,
                }

            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"[{idx}/{len(samples)}] Error generating response: {e}"
                )

                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "query": sample.query,
                    "domain": sample.domain,
                    "language": sample.language,
                    "article": "",
                    "response_time_seconds": elapsed,
                    "error": str(e),
                    "metadata": sample.metadata,
                }

        return results

    async def _grade_single_response(
        self,
        task_id: str,
        data: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Grade a single response using RACE + FACT evaluation."""
        async with semaphore:
            start_time = time.time()

            score = await score_deepresearch_async(
                query=data.get("query", ""),
                domain=data.get("domain", ""),
                article=data.get("article", ""),
                gemini_api_key=self.gemini_api_key,
                jina_api_key=self.jina_api_key,
                race_weight=self.race_weight,
                fact_weight=self.fact_weight,
            )

            grading_time = time.time() - start_time

            return {
                "task_id": task_id,
                "race_score": score.race,
                "fact_score": score.fact,
                "overall_score": score.overall_score,
                "grading_time_seconds": grading_time,
                "error": score.race.error or score.fact.error,
            }

    async def _grade_all_responses_async(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Grade all responses concurrently using RACE + FACT evaluation."""
        semaphore = asyncio.Semaphore(self.max_concurrent_grading)

        tasks = []
        for task_id, data in results.items():
            if data.get("error"):
                continue
            if not data.get("article"):
                continue

            tasks.append(self._grade_single_response(task_id, data, semaphore))

        self.logger.info(
            f"Grading {len(tasks)} responses with Gemini "
            f"(max {self.max_concurrent_grading} concurrent)"
        )

        grading_results = {}
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed:
                if isinstance(result, Exception):
                    self.logger.error(f"Grading error: {result}")
                    continue
                grading_results[result["task_id"]] = result

        return grading_results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using RACE + FACT scoring.

        Args:
            results: Dict from generate_responses() with response data.

        Returns:
            Dict with aggregate metrics.
        """
        self.logger.info("Starting DeepResearch evaluation with Gemini")

        # Run async grading
        grading_results = asyncio.run(self._grade_all_responses_async(results))

        # Build DeepResearchResult objects
        self._results = {}
        for task_id, data in results.items():
            grading = grading_results.get(task_id, {})

            race_score = grading.get("race_score", RACEScore())
            fact_score = grading.get("fact_score", FACTScore())
            overall_score = grading.get("overall_score", 0.0)

            if data.get("error"):
                overall_score = 0.0

            self._results[task_id] = DeepResearchResult(
                task_id=data.get("task_id", task_id),
                query=data.get("query", ""),
                domain=data.get("domain", ""),
                language=data.get("language", ""),
                article=data.get("article", ""),
                race_score=race_score,
                fact_score=fact_score,
                overall_score=overall_score,
                response_time_seconds=data.get("response_time_seconds", 0.0),
                grading_time_seconds=grading.get("grading_time_seconds", 0.0),
                error=data.get("error") or grading.get("error"),
                metadata=data.get("metadata", {}),
            )

        # Calculate aggregate metrics
        all_results = list(self._results.values())
        total = len(all_results)

        if not total:
            self.logger.warning("No results to evaluate")
            return {
                "total_samples": 0.0,
                "overall_score": 0.0,
            }

        error_count = sum(1 for r in all_results if r.error)
        valid_count = total - error_count
        valid_results = [r for r in all_results if not r.error]

        # RACE metrics
        avg_race_overall = (
            sum(r.race_score.overall for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )
        avg_comprehensiveness = (
            sum(r.race_score.comprehensiveness for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )
        avg_insight_depth = (
            sum(r.race_score.insight_depth for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )
        avg_instruction_following = (
            sum(r.race_score.instruction_following for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )
        avg_readability = (
            sum(r.race_score.readability for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )

        # FACT metrics
        avg_citation_accuracy = (
            sum(r.fact_score.citation_accuracy for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )
        avg_effective_citations = (
            sum(r.fact_score.effective_citations for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )
        total_claims = sum(r.fact_score.total_claims for r in valid_results)
        total_verified = sum(r.fact_score.verified_claims for r in valid_results)

        # Overall score
        avg_overall = (
            sum(r.overall_score for r in valid_results) / valid_count
            if valid_count > 0 else 0.0
        )

        metrics = {
            "total_samples": float(total),
            "valid_samples": float(valid_count),
            "error_samples": float(error_count),
            # Overall
            "overall_score": round(avg_overall, 2),
            # RACE metrics
            "race_overall": round(avg_race_overall, 2),
            "race_comprehensiveness": round(avg_comprehensiveness, 2),
            "race_insight_depth": round(avg_insight_depth, 2),
            "race_instruction_following": round(avg_instruction_following, 2),
            "race_readability": round(avg_readability, 2),
            # FACT metrics
            "fact_citation_accuracy": round(avg_citation_accuracy * 100, 2),
            "fact_effective_citations": round(avg_effective_citations * 100, 2),
            "fact_total_claims": float(total_claims),
            "fact_verified_claims": float(total_verified),
        }

        # Calculate per-domain metrics
        domain_results: Dict[str, List[DeepResearchResult]] = {}
        for r in valid_results:
            if r.domain not in domain_results:
                domain_results[r.domain] = []
            domain_results[r.domain].append(r)

        for domain, domain_list in domain_results.items():
            domain_key = domain.lower().replace(" ", "_")
            domain_avg = sum(r.overall_score for r in domain_list) / len(domain_list)
            metrics[f"domain_{domain_key}_score"] = round(domain_avg, 2)
            metrics[f"domain_{domain_key}_count"] = float(len(domain_list))

        # Calculate per-language metrics
        for lang in ["en", "zh"]:
            lang_results = [r for r in valid_results if r.language.lower() == lang]
            if lang_results:
                lang_avg = sum(r.overall_score for r in lang_results) / len(lang_results)
                metrics[f"language_{lang}_score"] = round(lang_avg, 2)
                metrics[f"language_{lang}_count"] = float(len(lang_results))

        # Timing metrics
        total_grading_time = sum(r.grading_time_seconds for r in all_results)
        avg_response_time = sum(r.response_time_seconds for r in all_results) / total

        metrics["total_grading_time_seconds"] = round(total_grading_time, 2)
        metrics["avg_response_time_seconds"] = round(avg_response_time, 2)

        # Add retrieval metrics
        retrieval_metrics = self.get_retrieval_metrics()
        metrics.update(retrieval_metrics)

        self.logger.info(
            f"DeepResearch Evaluation Complete: {avg_overall:.1f}% overall score "
            f"(RACE: {avg_race_overall:.1f}/10, FACT: {avg_citation_accuracy*100:.1f}% citations)"
        )

        return metrics

    def get_results(self) -> Dict[str, DeepResearchResult]:
        """Get detailed results for each sample."""
        return self._results.copy()

    def get_domain_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get score breakdown by domain."""
        breakdown: Dict[str, Dict[str, Any]] = {}

        for r in self._results.values():
            if r.error:
                continue

            domain = r.domain
            if domain not in breakdown:
                breakdown[domain] = {
                    "total": 0,
                    "scores": [],
                    "race_scores": [],
                    "fact_scores": [],
                }

            breakdown[domain]["total"] += 1
            breakdown[domain]["scores"].append(r.overall_score)
            breakdown[domain]["race_scores"].append(r.race_score.overall)
            breakdown[domain]["fact_scores"].append(
                (r.fact_score.citation_accuracy + r.fact_score.effective_citations) / 2
            )

        # Calculate averages
        for domain, data in breakdown.items():
            if data["total"] > 0:
                data["avg_overall"] = round(
                    sum(data["scores"]) / data["total"], 2
                )
                data["avg_race"] = round(
                    sum(data["race_scores"]) / data["total"], 2
                )
                data["avg_fact"] = round(
                    sum(data["fact_scores"]) / data["total"], 3
                )

        return breakdown


@register_benchmark("deepresearch")
def _create_deepresearch_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> DeepResearchBenchmark:
    """
    Create a DeepResearchBench benchmark instance.

    Options:
        Dataset options:
        - language: Filter by language ("en" or "zh")
        - domains: List of domains to filter (e.g., ["Physics", "Chemistry"])
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - input_prompt: Custom prompt template (default: None)

        Retrieval options:
        - retrieval_method: "none", "bm25", "dense", "hybrid" (default: "none")
        - retrieval_top_k: Number of documents to retrieve (default: 5)
        - retrieval_cache_dir: Cache directory for retrieval index

        Grading options:
        - gemini_api_key: API key for Gemini (default: None, uses env vars)
        - jina_api_key: API key for Jina (default: None, uses env vars)
        - max_concurrent_grading: Concurrent grading calls (default: 5)
        - race_weight: Weight for RACE score (default: 0.6)
        - fact_weight: Weight for FACT score (default: 0.4)

    Example:
        >>> benchmark = get_benchmark("deepresearch")(options={
        ...     "limit": 10,
        ...     "language": "en",
        ...     "domains": ["Physics", "Chemistry"],
        ... })

    Example with retrieval:
        >>> benchmark = get_benchmark("deepresearch")(options={
        ...     "retrieval_method": "bm25",
        ...     "retrieval_top_k": 5,
        ...     "limit": 10,
        ... })
    """
    options = options or {}
    return DeepResearchBenchmark(**options)
