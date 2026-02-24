from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.retrieval import (
    BenchmarkRetrievalMixin,
    RetrievalConfig,
    RetrievalMethod,
    load_financebench_corpus,
)

from .dataset import load_financebench_samples
from .prompts import SYSTEM_INSTRUCTION, format_query
from .scorer import FinanceBenchScorer, compute_metrics, parse_model_response
from .types import FinanceBenchResult, FinanceBenchSample

logger = logging.getLogger(__name__)


class FinanceBenchBenchmark(BenchmarkRetrievalMixin, DatasetBenchmark):
    """FinanceBench benchmark for evaluating financial QA capabilities.

    Supports three context modes:
    1. with_context=False: No context provided (closed-book QA)
    2. with_context=True: Ground truth evidence provided (oracle setting)
    3. retrieval_method="bm25"|"dense"|"hybrid": RAG from indexed evidence

    Example with retrieval:
        >>> benchmark = FinanceBenchBenchmark(
        ...     retrieval_method="bm25",
        ...     retrieval_top_k=5,
        ...     limit=100,
        ... )
        >>> results = benchmark.run_benchmark(orchestrator)
    """

    def __init__(
        self,
        *,
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        with_context: bool = False,
        # Retrieval options
        retrieval_method: RetrievalMethod = "none",
        retrieval_top_k: int = 5,
        retrieval_cache_dir: Optional[Path] = None,
        telemetry_collector: Optional[Any] = None,
        # Scoring options
        judge_orchestrator: Optional[Any] = None,
        judge_model: Optional[Any] = None,
        scoring_mode: Literal["llm"] = "llm",
        logger: Optional[logging.Logger] = None,
        save_incremental: bool = True,
        run_id: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=SYSTEM_INSTRUCTION)
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.with_context = with_context
        self.retrieval_method = retrieval_method
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_cache_dir = retrieval_cache_dir
        self.telemetry_collector = telemetry_collector
        self.judge_orchestrator = judge_orchestrator
        self.judge_model = judge_model
        self.scoring_mode = scoring_mode
        self.save_incremental = save_incremental
        self.run_id = run_id or f"run_{int(time.time())}"
        self.samples_cache: List[FinanceBenchSample] = []
        self.results_cache: Dict[str, FinanceBenchResult] = {}
        self.scorer_instance: Optional[FinanceBenchScorer] = None
        self._retrieval_initialized = False
    
    def load_samples(self) -> List[FinanceBenchSample]:
        if not self.samples_cache:
            self.samples_cache = load_financebench_samples(limit=self.limit, shuffle=self.shuffle, seed=self.seed, cache_dir=self.cache_dir)
            self.logger.info(f"Loaded {len(self.samples_cache)} FinanceBench samples")
        return self.samples_cache

    def _init_retrieval_if_needed(self) -> None:
        """Initialize retrieval system if configured and not already done."""
        if self._retrieval_initialized:
            return
        self._retrieval_initialized = True

        if self.retrieval_method == "none":
            return

        # Load samples first (needed for corpus)
        samples = self.load_samples()

        # Create retrieval config
        config = RetrievalConfig(
            method=self.retrieval_method,
            top_k=self.retrieval_top_k,
            cache_dir=self.retrieval_cache_dir or self.cache_dir,
            telemetry_collector=self.telemetry_collector,
        )

        # Load corpus from evidence
        documents = load_financebench_corpus(
            samples=samples,
            cache_dir=self.cache_dir,
            include_full_page=False,
        )

        # Initialize retrieval
        self.init_retrieval(config, documents, benchmark_name="financebench")
    
    def get_scorer(self) -> FinanceBenchScorer:
        if self.scorer_instance is None:
            self.scorer_instance = FinanceBenchScorer(judge_model=self.judge_model, judge_orchestrator=self.judge_orchestrator)
        return self.scorer_instance
    
    def extract_response_text(self, response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "content") and response.content is not None:
            return str(response.content)
        return str(response)
    
    def build_prompt(self, sample: FinanceBenchSample) -> str:
        """Build prompt for a sample, optionally with context or retrieval.

        Priority:
        1. If with_context=True, use ground truth evidence (oracle)
        2. If retrieval_method != "none", use retrieved context (RAG)
        3. Otherwise, no context (closed-book)
        """
        context = ""

        # Oracle context from ground truth evidence
        if self.with_context and sample.evidence:
            context_parts = [ev.evidence_text for ev in sample.evidence if ev.evidence_text]
            context = "\n\n".join(context_parts)

        # RAG context from retrieval (if no oracle context)
        elif self.retrieval_method != "none":
            self._init_retrieval_if_needed()
            if self.retrieval_enabled:
                results = self.search_retrieval(sample.question, top_k=self.retrieval_top_k)
                if results:
                    context = self._retrieval.format_context(results, include_scores=False)

        return format_query(sample.question, context)
    
    def save_incremental_result(self, result: dict) -> None:
        if not self.save_incremental:
            return
        runs_dir = (self.cache_dir or Path.home() / ".cache" / "ipw" / "financebench") / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        with open(runs_dir / f"{self.run_id}.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    
    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Generate model responses for all FinanceBench samples."""
        samples = self.load_samples()
        results: Dict[str, Any] = {}

        # Initialize retrieval if needed
        self._init_retrieval_if_needed()

        self.logger.info(f"Generating responses for {len(samples)} FinanceBench samples")
        self.logger.info(f"With context: {self.with_context}")
        if self.retrieval_method != "none":
            self.logger.info(f"Retrieval: {self.retrieval_method} (top_k={self.retrieval_top_k})")
        
        for idx, sample in enumerate(samples, 1):
            self.logger.info(f"[{idx}/{len(samples)}] Processing {sample.uid} ({sample.company})")
            start = time.time()
            
            try:
                response = orchestrator.run(self.build_prompt(sample))
                response_text = self.extract_response_text(response)
                elapsed = time.time() - start
                self.logger.info(f"[{idx}/{len(samples)}] Generated response in {elapsed:.2f}s")
                
                parsed = parse_model_response(response_text)
                result = {
                    "uid": sample.uid, "question": sample.question, "ground_truth": sample.answer,
                    "model_response": response_text, "extracted_answer": parsed.exact_answer,
                    "confidence": parsed.confidence, "response_time_seconds": elapsed,
                    "company": sample.company, "question_type": sample.question_type,
                    "error": None, "parse_errors": parsed.parse_errors,
                }
            except Exception as e:
                self.logger.error(f"[{idx}/{len(samples)}] Error: {e}")
                result = {
                    "uid": sample.uid, "question": sample.question, "ground_truth": sample.answer,
                    "model_response": "", "extracted_answer": "", "confidence": 100.0,
                    "response_time_seconds": time.time() - start, "company": sample.company,
                    "question_type": sample.question_type, "error": str(e), "parse_errors": [],
                }
            
            results[sample.uid] = result
            self.save_incremental_result(result)
        
        return results
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate responses using LLM-based judging."""
        self.logger.info("Starting FinanceBench evaluation (LLM judging)")
        scorer = self.get_scorer()
        scored = []
        
        for uid, data in results.items():
            if data.get("error"):
                scored.append({
                    "uid": uid, "is_correct": False, "confidence": data.get("confidence", 100.0),
                    "question_type": data.get("question_type", ""), "error": data.get("error"),
                    "response_time_seconds": data.get("response_time_seconds", 0.0)
                })
                continue
            
            try:
                judge = scorer.judge_response(data["question"], data["model_response"], data["ground_truth"])
                self.results_cache[uid] = FinanceBenchResult(
                    uid=uid, question=data["question"], ground_truth=data["ground_truth"],
                    model_response=data["model_response"], extracted_answer=data.get("extracted_answer", ""),
                    confidence=data.get("confidence", 100.0), is_correct=judge.is_correct,
                    company=data.get("company", ""), question_type=data.get("question_type", ""),
                    response_time_seconds=data.get("response_time_seconds", 0.0),
                    metadata={"judge_response": judge.raw_response, "judge_reasoning": judge.reasoning},
                )
                scored.append({
                    "uid": uid, "is_correct": judge.is_correct, "confidence": data.get("confidence", 100.0),
                    "question_type": data.get("question_type", ""), "error": None,
                    "response_time_seconds": data.get("response_time_seconds", 0.0)
                })
            except Exception as e:
                self.logger.error(f"Error judging {uid}: {e}")
                scored.append({
                    "uid": uid, "is_correct": False, "confidence": data.get("confidence", 100.0),
                    "question_type": data.get("question_type", ""), "error": f"judge_error: {e}",
                    "response_time_seconds": data.get("response_time_seconds", 0.0)
                })
        
        metrics = compute_metrics(scored)

        # Add retrieval metrics
        retrieval_metrics = self.get_retrieval_metrics()
        metrics.update(retrieval_metrics)

        self.logger.info(f"FinanceBench Evaluation Complete: {metrics['accuracy']:.1f}% accuracy ({metrics['correct_count']}/{metrics['valid_count']} correct), ECE: {metrics['ece_10bin']:.1f}%")
        return metrics
    
    def get_results(self) -> Dict[str, FinanceBenchResult]:
        return self.results_cache.copy()
    
    def get_samples(self) -> List[FinanceBenchSample]:
        return self.load_samples()


@register_benchmark("financebench")
def create_financebench_benchmark(options: Optional[Dict[str, Any]] = None) -> FinanceBenchBenchmark:
    """Create a FinanceBench benchmark instance.

    Options:
        Dataset options:
        - shuffle: Whether to shuffle samples (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - cache_dir: Cache directory for dataset

        Context options (mutually exclusive):
        - with_context: Use ground truth evidence (oracle setting)
        - retrieval_method: "none", "bm25", "dense", "hybrid" (RAG setting)
        - retrieval_top_k: Number of documents to retrieve (default: 5)
        - retrieval_cache_dir: Cache directory for retrieval index

        Scoring options:
        - judge_orchestrator: Orchestrator for LLM judge
        - judge_model: Model for LLM judge
        - scoring_mode: Scoring mode (default: "llm")

    Example with retrieval:
        >>> benchmark = create_financebench_benchmark({
        ...     "retrieval_method": "bm25",
        ...     "retrieval_top_k": 5,
        ...     "limit": 100,
        ... })
    """
    return FinanceBenchBenchmark(**(options or {}))
