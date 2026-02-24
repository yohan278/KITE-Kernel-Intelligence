# benchmarks/frames/main.py
"""
FRAMES (Factual Retrieval And Multi-hop Evaluation Suite) Benchmark.

Evaluates AI models on multi-hop factual retrieval questions that require
information synthesis from 2-15 Wikipedia articles.

Uses LLM-as-judge grading for semantic comparison following the official
FRAMES evaluation approach.

Supports retrieval augmentation:
- fetch_wikipedia=True: Fetch and index actual Wikipedia content
- retrieval_method="bm25"|"dense"|"hybrid": Search indexed content

Uses the google/frames-benchmark dataset from HuggingFace.
Reference: https://arxiv.org/abs/2409.12941
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
    load_frames_corpus,
)
from evals.benchmarks.frames.dataset import FRAMESSample, load_frames_samples
from evals.benchmarks.frames.scorer import grade_answer_async


@dataclass
class FRAMESResult:
    """Result from evaluating a single FRAMES sample."""

    index: int
    question: str
    ground_truth: str
    model_answer: str
    reasoning_types: str
    is_correct: bool
    response_time_seconds: float = 0.0
    grading_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FRAMESBenchmark(BenchmarkRetrievalMixin, DatasetBenchmark):
    """
    FRAMES benchmark for evaluating multi-hop factual retrieval.

    This benchmark:
    1. Loads questions from the google/frames-benchmark dataset
    2. Optionally fetches and indexes Wikipedia content for RAG
    3. Generates model responses using the provided orchestrator
    4. Grades responses using LLM-as-judge (semantic comparison)

    Retrieval modes:
    - No retrieval (default): Model uses parametric knowledge only
    - With wiki_links: Include Wikipedia URLs as hints in prompt
    - With retrieval: Fetch Wikipedia content and use RAG

    Example usage:
        >>> # Basic usage (no retrieval)
        >>> benchmark = FRAMESBenchmark(limit=10)
        >>> results = benchmark.run_benchmark(orchestrator)

        >>> # With Wikipedia retrieval
        >>> benchmark = FRAMESBenchmark(
        ...     retrieval_method="bm25",
        ...     fetch_wikipedia=True,
        ...     limit=10,
        ... )
        >>> results = benchmark.run_benchmark(orchestrator)
    """

    def __init__(
        self,
        *,
        # Dataset options
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        reasoning_types: Optional[List[str]] = None,
        input_prompt: Optional[str] = None,
        # Retrieval options
        retrieval_method: RetrievalMethod = "none",
        retrieval_top_k: int = 5,
        fetch_wikipedia: bool = False,
        max_wikipedia_articles: int = 500,
        retrieval_cache_dir: Optional[Path] = None,
        telemetry_collector: Optional[Any] = None,
        # Grading options
        grader_model: str = "claude-3-5-sonnet-20241022",
        grader_api_key: Optional[str] = None,
        max_concurrent_grading: int = 10,
        # Execution options
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the FRAMES benchmark.

        Args:
            split: Dataset split to use (default: "test")
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            limit: Maximum number of samples to evaluate
            reasoning_types: Filter to specific reasoning types
            input_prompt: Custom input prompt template
            retrieval_method: "none", "bm25", "dense", "hybrid" (default: "none")
            retrieval_top_k: Number of documents to retrieve (default: 5)
            fetch_wikipedia: Whether to fetch actual Wikipedia content (default: False)
            max_wikipedia_articles: Maximum articles to fetch (default: 500)
            retrieval_cache_dir: Cache directory for retrieval index
            telemetry_collector: Energy monitor collector for telemetry
            grader_model: Model for LLM-as-judge grading (default: claude-3-5-sonnet)
            grader_api_key: API key for grader model (optional, uses env vars)
            max_concurrent_grading: Max concurrent grading calls (default: 10)
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=None)

        # Dataset config
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.reasoning_types = reasoning_types
        self.input_prompt = input_prompt

        # Retrieval config
        self.retrieval_method = retrieval_method
        self.retrieval_top_k = retrieval_top_k
        self.fetch_wikipedia = fetch_wikipedia
        self.max_wikipedia_articles = max_wikipedia_articles
        self.retrieval_cache_dir = retrieval_cache_dir
        self.telemetry_collector = telemetry_collector
        self._retrieval_initialized = False

        # Grading config
        self.grader_model = grader_model
        self.grader_api_key = grader_api_key
        self.max_concurrent_grading = max_concurrent_grading

        # Results storage
        self._samples: List[FRAMESSample] = []
        self._results: Dict[int, FRAMESResult] = {}

    def _load_samples(self) -> List[FRAMESSample]:
        """Load samples from the dataset."""
        if not self._samples:
            self._samples = list(load_frames_samples(
                split=self.split,
                shuffle=self.shuffle,
                seed=self.seed,
                limit=self.limit,
                reasoning_types=self.reasoning_types,
            ))
            self.logger.info(f"Loaded {len(self._samples)} FRAMES samples")
            self.logger.info(f"Split: {self.split}")

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

        # Load corpus from Wikipedia links
        documents = load_frames_corpus(
            samples=samples,
            fetch_wikipedia=self.fetch_wikipedia,
            max_articles=self.max_wikipedia_articles,
            cache_dir=self.retrieval_cache_dir,
        )

        if not documents:
            self.logger.warning(
                "No documents loaded for FRAMES retrieval. "
                "Set fetch_wikipedia=True to fetch actual Wikipedia content."
            )
            return

        # Initialize retrieval
        self.init_retrieval(config, documents, benchmark_name="frames")

    def _build_prompt_with_retrieval(self, sample: FRAMESSample) -> str:
        """Build prompt with retrieved context.

        Args:
            sample: FRAMES sample

        Returns:
            Prompt with retrieved context
        """
        # Get base prompt
        base_prompt = sample.get_prompt(self.input_prompt)

        # If retrieval is enabled, augment with retrieved context
        if self.retrieval_enabled:
            results = self.search_retrieval(sample.prompt, top_k=self.retrieval_top_k)
            if results:
                context = self._retrieval.format_context(results, include_scores=False)
                # Insert context before the question
                context_section = (
                    "Here is relevant information from Wikipedia that may help:\n\n"
                    f"{context}\n\n"
                )
                # Find where to insert (after the wiki_context section or before question)
                if "Here is the question:" in base_prompt:
                    parts = base_prompt.split("Here is the question:")
                    return f"{parts[0]}{context_section}Here is the question:{parts[1]}"
                else:
                    return f"{context_section}{base_prompt}"

        return base_prompt

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from orchestrator response."""
        if response is None:
            return ""

        if hasattr(response, 'content'):
            content = getattr(response, 'content', None)
            if content is not None:
                return str(content)

        return str(response)

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """
        Generate model responses for all FRAMES samples.

        Args:
            orchestrator: An orchestrator instance with a run() method.

        Returns:
            Dict mapping sample index to response data including raw response and timing.
        """
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        # Initialize retrieval if needed
        self._init_retrieval_if_needed()

        self.logger.info(f"Generating responses for {len(samples)} FRAMES samples")
        if self.retrieval_method != "none":
            self.logger.info(
                f"Retrieval: {self.retrieval_method} (top_k={self.retrieval_top_k}, "
                f"fetch_wikipedia={self.fetch_wikipedia})"
            )

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing sample {sample.index} "
                f"(reasoning: {sample.reasoning_types})"
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
                    f"[{idx}/{len(samples)}] Generated response in {elapsed:.2f}s"
                )

                results[str(sample.index)] = {
                    "index": sample.index,
                    "question": sample.prompt,
                    "ground_truth": sample.answer,
                    "model_answer": response_text,
                    "reasoning_types": sample.reasoning_types,
                    "wiki_links": sample.wiki_links,
                    "response_time_seconds": elapsed,
                    "error": None,
                }

            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"[{idx}/{len(samples)}] Error generating response: {e}"
                )

                results[str(sample.index)] = {
                    "index": sample.index,
                    "question": sample.prompt,
                    "ground_truth": sample.answer,
                    "model_answer": "",
                    "reasoning_types": sample.reasoning_types,
                    "wiki_links": sample.wiki_links,
                    "response_time_seconds": elapsed,
                    "error": str(e),
                }

        return results

    async def _grade_single_response(
        self,
        key: str,
        data: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Grade a single response using LLM-as-judge."""
        async with semaphore:
            start_time = time.time()

            result = await grade_answer_async(
                question=data.get("question", ""),
                ground_truth=data.get("ground_truth", ""),
                predicted_answer=data.get("model_answer", ""),
                grader_model=self.grader_model,
                api_key=self.grader_api_key,
            )

            grading_time = time.time() - start_time

            return {
                "key": key,
                "is_correct": result["is_correct"],
                "grading_time_seconds": grading_time,
                "error": result.get("error"),
            }

    async def _grade_all_responses_async(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Grade all responses concurrently using LLM-as-judge."""
        semaphore = asyncio.Semaphore(self.max_concurrent_grading)

        tasks = []
        for key, data in results.items():
            if data.get("error"):
                continue
            if not data.get("model_answer"):
                continue

            tasks.append(self._grade_single_response(key, data, semaphore))

        self.logger.info(
            f"Grading {len(tasks)} responses with {self.grader_model} "
            f"(max {self.max_concurrent_grading} concurrent)"
        )

        grading_results = {}
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed:
                if isinstance(result, Exception):
                    self.logger.error(f"Grading error: {result}")
                    continue
                grading_results[result["key"]] = result

        return grading_results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using LLM-as-judge grading.

        Uses semantic comparison to determine if the model's answer captures
        the essential information from the ground truth.

        Args:
            results: Dict from generate_responses() with response data.

        Returns:
            Dict with aggregate metrics (accuracy, reasoning type breakdown, etc.)
        """
        self.logger.info(f"Starting FRAMES evaluation with {self.grader_model}")

        # Run async grading
        grading_results = asyncio.run(self._grade_all_responses_async(results))

        # Build FRAMESResult objects
        self._results = {}
        for key, data in results.items():
            grading = grading_results.get(key, {})

            is_correct = grading.get("is_correct", False)
            if data.get("error"):
                is_correct = False

            self._results[int(key)] = FRAMESResult(
                index=data.get("index", 0),
                question=data.get("question", ""),
                ground_truth=data.get("ground_truth", ""),
                model_answer=data.get("model_answer", ""),
                reasoning_types=data.get("reasoning_types", ""),
                is_correct=is_correct,
                response_time_seconds=data.get("response_time_seconds", 0.0),
                grading_time_seconds=grading.get("grading_time_seconds", 0.0),
                error=data.get("error") or grading.get("error"),
                metadata={"wiki_links": data.get("wiki_links", [])},
            )

        # Calculate aggregate metrics
        all_results = list(self._results.values())
        total = len(all_results)

        if not total:
            self.logger.warning("No results to evaluate")
            return {
                "total_samples": 0.0,
                "accuracy": 0.0,
            }

        correct_count = sum(1 for r in all_results if r.is_correct)
        error_count = sum(1 for r in all_results if r.error)
        valid_count = total - error_count

        accuracy = (correct_count / total) * 100
        accuracy_valid = (correct_count / valid_count) * 100 if valid_count > 0 else 0.0

        # Calculate per-reasoning-type metrics
        type_results: Dict[str, Dict[str, int]] = {}
        for r in all_results:
            if r.error:
                continue
            types = [t.strip() for t in r.reasoning_types.split(",") if t.strip()]
            if not types:
                types = ["unknown"]
            for rtype in types:
                if rtype not in type_results:
                    type_results[rtype] = {"correct": 0, "total": 0}
                type_results[rtype]["total"] += 1
                if r.is_correct:
                    type_results[rtype]["correct"] += 1

        metrics = {
            "total_samples": float(total),
            "valid_samples": float(valid_count),
            "error_samples": float(error_count),
            "correct_count": float(correct_count),
            "incorrect_count": float(valid_count - correct_count),
            "accuracy": round(accuracy, 2),
            "accuracy_valid_only": round(accuracy_valid, 2),
        }

        # Add per-reasoning-type accuracy
        for rtype, counts in type_results.items():
            type_key = rtype.lower().replace(" ", "_").replace("-", "_")
            type_accuracy = (counts["correct"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            metrics[f"reasoning_{type_key}_accuracy"] = round(type_accuracy, 2)
            metrics[f"reasoning_{type_key}_correct"] = float(counts["correct"])
            metrics[f"reasoning_{type_key}_total"] = float(counts["total"])

        # Calculate timing metrics
        total_grading_time = sum(r.grading_time_seconds for r in all_results)
        avg_response_time = sum(r.response_time_seconds for r in all_results) / total

        metrics["total_grading_time_seconds"] = round(total_grading_time, 2)
        metrics["avg_response_time_seconds"] = round(avg_response_time, 2)

        # Add retrieval metrics
        retrieval_metrics = self.get_retrieval_metrics()
        metrics.update(retrieval_metrics)

        self.logger.info(
            f"FRAMES Evaluation Complete: {accuracy:.1f}% accuracy "
            f"({correct_count}/{total} correct)"
        )

        return metrics

    def get_results(self) -> Dict[int, FRAMESResult]:
        """Get detailed results for each sample."""
        return self._results.copy()

    def get_reasoning_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get accuracy breakdown by reasoning type."""
        breakdown: Dict[str, Dict[str, Any]] = {}

        for r in self._results.values():
            if r.error:
                continue

            types = [t.strip() for t in r.reasoning_types.split(",") if t.strip()]
            if not types:
                types = ["unknown"]

            for rtype in types:
                if rtype not in breakdown:
                    breakdown[rtype] = {
                        "total": 0,
                        "correct": 0,
                        "questions": [],
                    }

                breakdown[rtype]["total"] += 1
                if r.is_correct:
                    breakdown[rtype]["correct"] += 1

                breakdown[rtype]["questions"].append({
                    "index": r.index,
                    "is_correct": r.is_correct,
                    "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                })

        # Calculate accuracy
        for rtype, data in breakdown.items():
            if data["total"] > 0:
                data["accuracy"] = round((data["correct"] / data["total"]) * 100, 2)
            else:
                data["accuracy"] = 0.0

        return breakdown


@register_benchmark("frames")
def _create_frames_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> FRAMESBenchmark:
    """
    Create a FRAMES benchmark instance.

    Options:
        Dataset options:
        - split: Dataset split (default: "test")
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - reasoning_types: List of reasoning types to filter (default: None = all)
        - input_prompt: Custom prompt template (default: None)

        Retrieval options:
        - retrieval_method: "none", "bm25", "dense", "hybrid" (default: "none")
        - retrieval_top_k: Number of documents to retrieve (default: 5)
        - fetch_wikipedia: Whether to fetch Wikipedia content (default: False)
        - max_wikipedia_articles: Max articles to fetch (default: 500)
        - retrieval_cache_dir: Cache directory for retrieval index

        Grading options:
        - grader_model: LLM for grading (default: "claude-3-5-sonnet-20241022")
        - grader_api_key: API key for grader (default: None, uses env vars)
        - max_concurrent_grading: Concurrent grading calls (default: 10)

    Example without retrieval:
        >>> benchmark = get_benchmark("frames")(options={
        ...     "limit": 10,
        ...     "reasoning_types": ["numerical", "temporal"],
        ... })

    Example with retrieval:
        >>> benchmark = get_benchmark("frames")(options={
        ...     "retrieval_method": "bm25",
        ...     "fetch_wikipedia": True,
        ...     "retrieval_top_k": 5,
        ...     "limit": 10,
        ... })
    """
    options = options or {}
    return FRAMESBenchmark(**options)
