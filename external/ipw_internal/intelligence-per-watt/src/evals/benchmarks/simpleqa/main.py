# benchmarks/simpleqa/main.py
"""
SimpleQA Verified Benchmark.

Evaluates AI models on short-form factual QA testing parametric knowledge
across diverse topics (Politics, Art, History, Sports, etc.).

Uses LLM-as-judge grading following the official OpenAI simple-evals approach:
- CORRECT: Answer contains important information without contradictions
- INCORRECT: Answer contains factual contradictions
- NOT_ATTEMPTED: Missing important information but no contradictions

Supports retrieval augmentation:
- retrieval_method="bm25"|"dense"|"hybrid": Use RAG from Wikipedia
- This converts SimpleQA from a parametric knowledge test to a RAG benchmark

Uses the openai/simple-evals dataset from HuggingFace.
Reference: https://github.com/openai/simple-evals
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.retrieval import (
    BenchmarkRetrievalMixin,
    RetrievalConfig,
    RetrievalMethod,
    load_simpleqa_corpus,
)
from evals.benchmarks.simpleqa.dataset import SimpleQASample, load_simpleqa_samples
from evals.benchmarks.simpleqa.scorer import grade_answer_async, GradeType


@dataclass
class SimpleQAResult:
    """Result from evaluating a single SimpleQA sample."""

    original_index: int
    question: str
    ground_truth: str
    model_answer: str
    topic: str
    answer_type: str
    grade: GradeType  # CORRECT, INCORRECT, NOT_ATTEMPTED
    is_correct: bool
    is_incorrect: bool
    is_not_attempted: bool
    response_time_seconds: float = 0.0
    grading_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleQABenchmark(BenchmarkRetrievalMixin, DatasetBenchmark):
    """
    SimpleQA Verified benchmark for evaluating factual knowledge.

    This benchmark:
    1. Loads questions from the openai/simple-evals dataset
    2. Optionally loads Wikipedia corpus for RAG augmentation
    3. Generates model responses using the provided orchestrator
    4. Grades responses using LLM-as-judge (CORRECT/INCORRECT/NOT_ATTEMPTED)

    Uses the official OpenAI grading approach with gpt-4o as the default grader.

    Two modes:
    - Closed-book (default): Tests parametric knowledge only
    - RAG-augmented: Retrieves from Wikipedia before answering

    Example usage:
        >>> # Closed-book (parametric knowledge)
        >>> benchmark = SimpleQABenchmark(limit=10)
        >>> results = benchmark.run_benchmark(orchestrator)

        >>> # RAG-augmented
        >>> benchmark = SimpleQABenchmark(
        ...     retrieval_method="bm25",
        ...     max_corpus_documents=5000,
        ...     limit=10,
        ... )
        >>> results = benchmark.run_benchmark(orchestrator)
    """

    def __init__(
        self,
        *,
        # Dataset options
        split: str = "simpleqa",
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        topics: Optional[List[str]] = None,
        answer_types: Optional[List[str]] = None,
        input_prompt: Optional[str] = None,
        # Retrieval options
        retrieval_method: RetrievalMethod = "none",
        retrieval_top_k: int = 5,
        max_corpus_documents: int = 10000,
        corpus_source: str = "wikipedia_simple",
        retrieval_cache_dir: Optional[Path] = None,
        telemetry_collector: Optional[Any] = None,
        # Grading options
        grader_model: str = "gpt-4o",
        grader_api_key: Optional[str] = None,
        max_concurrent_grading: int = 10,
        # Execution options
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SimpleQA benchmark.

        Args:
            split: Dataset split/config to use (default: "simpleqa")
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            limit: Maximum number of samples to evaluate
            topics: Filter to specific topics (e.g., ["Politics", "Art"])
            answer_types: Filter to specific answer types (e.g., ["Person", "Date"])
            input_prompt: Custom input prompt template
            retrieval_method: "none", "bm25", "dense", "hybrid" (default: "none")
            retrieval_top_k: Number of documents to retrieve (default: 5)
            max_corpus_documents: Max Wikipedia articles to index (default: 10000)
            corpus_source: Source for corpus ("wikipedia_simple", "wikipedia")
            retrieval_cache_dir: Cache directory for retrieval index
            telemetry_collector: Energy monitor collector for telemetry
            grader_model: Model for LLM-as-judge grading (default: "gpt-4o")
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
        self.topics = topics
        self.answer_types = answer_types
        self.input_prompt = input_prompt

        # Retrieval config
        self.retrieval_method = retrieval_method
        self.retrieval_top_k = retrieval_top_k
        self.max_corpus_documents = max_corpus_documents
        self.corpus_source = corpus_source
        self.retrieval_cache_dir = retrieval_cache_dir
        self.telemetry_collector = telemetry_collector
        self._retrieval_initialized = False

        # Grading config
        self.grader_model = grader_model
        self.grader_api_key = grader_api_key
        self.max_concurrent_grading = max_concurrent_grading

        # Results storage
        self._samples: List[SimpleQASample] = []
        self._results: Dict[int, SimpleQAResult] = {}

    def _load_samples(self) -> List[SimpleQASample]:
        """Load samples from the dataset."""
        if not self._samples:
            self._samples = list(load_simpleqa_samples(
                split=self.split,
                shuffle=self.shuffle,
                seed=self.seed,
                limit=self.limit,
                topics=self.topics,
                answer_types=self.answer_types,
            ))
            self.logger.info(f"Loaded {len(self._samples)} SimpleQA samples")
            self.logger.info(f"Split: {self.split}")

            if self.topics:
                self.logger.info(f"Filtered to topics: {self.topics}")
            if self.answer_types:
                self.logger.info(f"Filtered to answer types: {self.answer_types}")

        return self._samples

    def _init_retrieval_if_needed(self) -> None:
        """Initialize retrieval system if configured and not already done."""
        if self._retrieval_initialized:
            return
        self._retrieval_initialized = True

        if self.retrieval_method == "none":
            return

        # Create retrieval config
        config = RetrievalConfig(
            method=self.retrieval_method,
            top_k=self.retrieval_top_k,
            cache_dir=self.retrieval_cache_dir,
            telemetry_collector=self.telemetry_collector,
        )

        # Load corpus from Wikipedia
        self.logger.info(
            f"Loading {self.corpus_source} corpus "
            f"(max {self.max_corpus_documents} documents)..."
        )
        documents = load_simpleqa_corpus(
            topics=self.topics,
            max_documents=self.max_corpus_documents,
            source=self.corpus_source,
        )

        if not documents:
            self.logger.warning(
                "No documents loaded for SimpleQA retrieval. "
                "Check corpus source configuration."
            )
            return

        # Initialize retrieval
        self.init_retrieval(config, documents, benchmark_name="simpleqa")

    def _build_prompt_with_retrieval(self, sample: SimpleQASample) -> str:
        """Build prompt with retrieved context.

        Args:
            sample: SimpleQA sample

        Returns:
            Prompt with retrieved context
        """
        # Get base prompt
        base_prompt = sample.get_prompt(self.input_prompt)

        # If retrieval is enabled, augment with retrieved context
        if self.retrieval_enabled:
            results = self.search_retrieval(sample.problem, top_k=self.retrieval_top_k)
            if results:
                context = self._retrieval.format_context(results, include_scores=False)
                # Insert context before the question
                context_section = (
                    "Here is some relevant information that may help answer the question:\n\n"
                    f"{context}\n\n"
                )
                # Insert before "Question:" if present
                if "Question:" in base_prompt:
                    parts = base_prompt.split("Question:")
                    return f"{parts[0]}{context_section}Question:{parts[1]}"
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
        Generate model responses for all SimpleQA samples.

        Args:
            orchestrator: An orchestrator instance with a run() method.

        Returns:
            Dict mapping sample index to response data including raw response and timing.
        """
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        # Initialize retrieval if needed
        self._init_retrieval_if_needed()

        self.logger.info(f"Generating responses for {len(samples)} SimpleQA samples")
        if self.retrieval_method != "none":
            self.logger.info(
                f"Retrieval: {self.retrieval_method} (top_k={self.retrieval_top_k})"
            )

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing sample {sample.original_index} "
                f"(topic: {sample.topic}, type: {sample.answer_type})"
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

                results[str(sample.original_index)] = {
                    "original_index": sample.original_index,
                    "question": sample.problem,
                    "ground_truth": sample.answer,
                    "model_answer": response_text,
                    "topic": sample.topic,
                    "answer_type": sample.answer_type,
                    "response_time_seconds": elapsed,
                    "error": None,
                }

            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"[{idx}/{len(samples)}] Error generating response: {e}"
                )

                results[str(sample.original_index)] = {
                    "original_index": sample.original_index,
                    "question": sample.problem,
                    "ground_truth": sample.answer,
                    "model_answer": "",
                    "topic": sample.topic,
                    "answer_type": sample.answer_type,
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
                gold_target=data.get("ground_truth", ""),
                predicted_answer=data.get("model_answer", ""),
                grader_model=self.grader_model,
                api_key=self.grader_api_key,
            )

            grading_time = time.time() - start_time

            return {
                "key": key,
                "grade": result["grade"],
                "is_correct": result["is_correct"],
                "is_incorrect": result["is_incorrect"],
                "is_not_attempted": result["is_not_attempted"],
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

        Uses the official OpenAI SimpleQA grading approach with three categories:
        - CORRECT: Answer contains important information without contradictions
        - INCORRECT: Answer contains factual contradictions
        - NOT_ATTEMPTED: Missing important information but no contradictions

        Args:
            results: Dict from generate_responses() with response data.

        Returns:
            Dict with aggregate metrics (accuracy, topic breakdown, type breakdown, etc.)
        """
        self.logger.info(f"Starting SimpleQA evaluation with {self.grader_model}")

        # Run async grading
        grading_results = asyncio.run(self._grade_all_responses_async(results))

        # Build SimpleQAResult objects
        self._results = {}
        for key, data in results.items():
            grading = grading_results.get(key, {})

            # Default to NOT_ATTEMPTED for errors or empty responses
            grade = grading.get("grade", "NOT_ATTEMPTED")
            is_correct = grading.get("is_correct", False)
            is_incorrect = grading.get("is_incorrect", False)
            is_not_attempted = grading.get("is_not_attempted", True)

            if data.get("error"):
                grade = "NOT_ATTEMPTED"
                is_correct = False
                is_incorrect = False
                is_not_attempted = True

            self._results[int(key)] = SimpleQAResult(
                original_index=data.get("original_index", 0),
                question=data.get("question", ""),
                ground_truth=data.get("ground_truth", ""),
                model_answer=data.get("model_answer", ""),
                topic=data.get("topic", ""),
                answer_type=data.get("answer_type", ""),
                grade=grade,
                is_correct=is_correct,
                is_incorrect=is_incorrect,
                is_not_attempted=is_not_attempted,
                response_time_seconds=data.get("response_time_seconds", 0.0),
                grading_time_seconds=grading.get("grading_time_seconds", 0.0),
                error=data.get("error") or grading.get("error"),
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
        incorrect_count = sum(1 for r in all_results if r.is_incorrect)
        not_attempted_count = sum(1 for r in all_results if r.is_not_attempted)

        # Standard accuracy
        accuracy = (correct_count / total) * 100

        # Accuracy given attempted (excluding NOT_ATTEMPTED)
        attempted_count = correct_count + incorrect_count
        accuracy_given_attempted = (
            (correct_count / attempted_count) * 100 if attempted_count > 0 else 0.0
        )

        # Calculate per-topic metrics
        topic_results: Dict[str, Dict[str, int]] = {}
        for r in all_results:
            topic = r.topic or "Unknown"
            if topic not in topic_results:
                topic_results[topic] = {"correct": 0, "incorrect": 0, "not_attempted": 0}
            if r.is_correct:
                topic_results[topic]["correct"] += 1
            elif r.is_incorrect:
                topic_results[topic]["incorrect"] += 1
            else:
                topic_results[topic]["not_attempted"] += 1

        # Calculate per-answer-type metrics
        type_results: Dict[str, Dict[str, int]] = {}
        for r in all_results:
            atype = r.answer_type or "Other"
            if atype not in type_results:
                type_results[atype] = {"correct": 0, "incorrect": 0, "not_attempted": 0}
            if r.is_correct:
                type_results[atype]["correct"] += 1
            elif r.is_incorrect:
                type_results[atype]["incorrect"] += 1
            else:
                type_results[atype]["not_attempted"] += 1

        metrics = {
            "total_samples": float(total),
            "correct_count": float(correct_count),
            "incorrect_count": float(incorrect_count),
            "not_attempted_count": float(not_attempted_count),
            "accuracy": round(accuracy, 2),
            "accuracy_given_attempted": round(accuracy_given_attempted, 2),
            "attempted_rate": round((attempted_count / total) * 100, 2),
        }

        # Add per-topic metrics
        for topic, counts in topic_results.items():
            topic_key = topic.lower().replace(" ", "_").replace("-", "_")
            topic_total = sum(counts.values())
            topic_accuracy = (counts["correct"] / topic_total) * 100 if topic_total > 0 else 0
            metrics[f"topic_{topic_key}_accuracy"] = round(topic_accuracy, 2)
            metrics[f"topic_{topic_key}_total"] = float(topic_total)

        # Add per-answer-type metrics
        for atype, counts in type_results.items():
            type_key = atype.lower().replace(" ", "_").replace("-", "_")
            type_total = sum(counts.values())
            type_accuracy = (counts["correct"] / type_total) * 100 if type_total > 0 else 0
            metrics[f"answer_type_{type_key}_accuracy"] = round(type_accuracy, 2)
            metrics[f"answer_type_{type_key}_total"] = float(type_total)

        # Calculate timing metrics
        total_grading_time = sum(r.grading_time_seconds for r in all_results)
        avg_response_time = sum(r.response_time_seconds for r in all_results) / total

        metrics["total_grading_time_seconds"] = round(total_grading_time, 2)
        metrics["avg_response_time_seconds"] = round(avg_response_time, 2)

        # Add retrieval metrics
        retrieval_metrics = self.get_retrieval_metrics()
        metrics.update(retrieval_metrics)

        self.logger.info(
            f"SimpleQA Evaluation Complete: {accuracy:.1f}% accuracy "
            f"({correct_count} correct, {incorrect_count} incorrect, "
            f"{not_attempted_count} not attempted)"
        )

        return metrics

    def get_results(self) -> Dict[int, SimpleQAResult]:
        """Get detailed results for each sample."""
        return self._results.copy()

    def get_topic_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get accuracy breakdown by topic."""
        breakdown: Dict[str, Dict[str, Any]] = {}

        for r in self._results.values():
            topic = r.topic or "Unknown"
            if topic not in breakdown:
                breakdown[topic] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_attempted": 0,
                }

            breakdown[topic]["total"] += 1
            if r.is_correct:
                breakdown[topic]["correct"] += 1
            elif r.is_incorrect:
                breakdown[topic]["incorrect"] += 1
            else:
                breakdown[topic]["not_attempted"] += 1

        # Calculate accuracy
        for topic, data in breakdown.items():
            if data["total"] > 0:
                data["accuracy"] = round((data["correct"] / data["total"]) * 100, 2)
            else:
                data["accuracy"] = 0.0

        return breakdown

    def get_answer_type_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get accuracy breakdown by answer type."""
        breakdown: Dict[str, Dict[str, Any]] = {}

        for r in self._results.values():
            atype = r.answer_type or "Other"
            if atype not in breakdown:
                breakdown[atype] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_attempted": 0,
                }

            breakdown[atype]["total"] += 1
            if r.is_correct:
                breakdown[atype]["correct"] += 1
            elif r.is_incorrect:
                breakdown[atype]["incorrect"] += 1
            else:
                breakdown[atype]["not_attempted"] += 1

        # Calculate accuracy
        for atype, data in breakdown.items():
            if data["total"] > 0:
                data["accuracy"] = round((data["correct"] / data["total"]) * 100, 2)
            else:
                data["accuracy"] = 0.0

        return breakdown


@register_benchmark("simpleqa")
def _create_simpleqa_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> SimpleQABenchmark:
    """
    Create a SimpleQA benchmark instance.

    Options:
        Dataset options:
        - split: Dataset split/config (default: "simpleqa")
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - topics: List of topics to filter (default: None = all)
        - answer_types: List of answer types to filter (default: None = all)
        - input_prompt: Custom prompt template (default: None)

        Retrieval options:
        - retrieval_method: "none", "bm25", "dense", "hybrid" (default: "none")
        - retrieval_top_k: Number of documents to retrieve (default: 5)
        - max_corpus_documents: Max Wikipedia articles to index (default: 10000)
        - corpus_source: "wikipedia_simple" or "wikipedia" (default: "wikipedia_simple")
        - retrieval_cache_dir: Cache directory for retrieval index

        Grading options:
        - grader_model: LLM for grading (default: "gpt-4o")
        - grader_api_key: API key for grader (default: None, uses env vars)
        - max_concurrent_grading: Concurrent grading calls (default: 10)

    Example (closed-book):
        >>> benchmark = get_benchmark("simpleqa")(options={
        ...     "limit": 100,
        ...     "topics": ["Science", "History"],
        ... })

    Example (RAG-augmented):
        >>> benchmark = get_benchmark("simpleqa")(options={
        ...     "retrieval_method": "bm25",
        ...     "retrieval_top_k": 5,
        ...     "max_corpus_documents": 5000,
        ...     "limit": 100,
        ... })
    """
    options = options or {}
    return SimpleQABenchmark(**options)
