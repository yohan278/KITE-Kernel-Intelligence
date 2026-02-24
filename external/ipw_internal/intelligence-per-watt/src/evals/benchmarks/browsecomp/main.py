# BrowseComp Benchmark Implementation.
# Paper: https://arxiv.org/pdf/2504.12516
# 1,266 questions requiring web browsing to find hard-to-find information.

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark

from .dataset import load_browsecomp_samples
from .prompts import BROWSING_INSTRUCTION, format_query, parse_model_response
from .scorer import BrowseCompScorer, compute_metrics
from .types import BrowseCompResult, BrowseCompSample


class BrowseCompBenchmark(DatasetBenchmark):
    """BrowseComp benchmark for evaluating web browsing agents."""
    
    def __init__(
        self,
        *,
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        with_browsing: bool = True,
        judge_orchestrator: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        save_incremental: bool = True,
        run_id: Optional[str] = None,
        concurrency: int = 1,  # Number of parallel questions
        sample_timeout: int = 300,  # Timeout per sample in seconds (default 5 min)
    ):
        super().__init__(logger=logger, system_instruction=None)
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.with_browsing = with_browsing
        self.judge_orchestrator = judge_orchestrator
        self.save_incremental = save_incremental
        self.run_id = run_id or f"run_{int(time.time())}"
        self.concurrency = max(1, concurrency)
        self.sample_timeout = sample_timeout
        self._results: Dict[str, BrowseCompResult] = {}
        self._save_lock = Lock()
        self._completed_count = 0
        self._completed_lock = Lock()
    
    def _create_error_result(self, sample: BrowseCompSample, error: str, elapsed: float) -> dict:
        """Create error result dict."""
        return {
            "uid": sample.uid, "question": sample.question, "answer": sample.answer,
            "model_response": "", "extracted_answer": "", "confidence": 100.0,
            "response_time_seconds": elapsed, "error": error, "parse_errors": [],
        }
    
    def _save_incremental_result(self, result: dict) -> None:
        if not self.save_incremental:
            return
        runs_dir = (self.cache_dir or Path.home() / ".cache" / "ipw" / "browsecomp") / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        with self._save_lock:
            with open(runs_dir / f"{self.run_id}.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
    
    def _process_single_sample(
        self, 
        sample: BrowseCompSample, 
        orchestrator_factory: Callable[[], Any],
        idx: int, 
        total: int
    ) -> dict:
        """Process a single sample with timeout using multiprocessing."""
        import multiprocessing as mp
        
        self.logger.info(f"Starting {sample.uid} (sample {idx} of {total})")
        start = time.time()
        error = None
        
        def worker(conn, factory, prompt):
            """Worker that creates orchestrator and runs it."""
            try:
                orchestrator = factory()
                response = orchestrator.run(prompt)
                conn.send(("ok", response.content if hasattr(response, 'content') else str(response)))
            except Exception as e:
                conn.send(("error", str(e)))
        
        try:
            query = format_query(sample.question)
            prompt = f"{BROWSING_INSTRUCTION}\n\n{query}" if self.with_browsing else query
            parent_conn, child_conn = mp.Pipe()
            
            ctx = mp.get_context('fork')
            proc = ctx.Process(target=worker, args=(child_conn, orchestrator_factory, prompt))
            proc.start()
            
            if not parent_conn.poll(timeout=self.sample_timeout):
                self.logger.warning(f"Timeout on {sample.uid} after {self.sample_timeout}s")
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                raise TimeoutError(f"Sample exceeded {self.sample_timeout}s timeout")
            
            status, response_text = parent_conn.recv()
            proc.join(timeout=5)
            if status != "ok":
                raise Exception(response_text)
            
            parsed = parse_model_response(response_text)
            result = {
                "uid": sample.uid, "question": sample.question, "answer": sample.answer,
                "model_response": response_text, "extracted_answer": parsed.exact_answer,
                "confidence": parsed.confidence, "response_time_seconds": time.time() - start,
                "error": None, "parse_errors": parsed.parse_errors,
            }
        except Exception as e:
            error = str(e)
            if not isinstance(e, TimeoutError):
                self.logger.error(f"Error on {sample.uid}: {error[:100]}")
            result = self._create_error_result(sample, error, time.time() - start)
        
        with self._completed_lock:
            self._completed_count += 1
            completed = self._completed_count
        status = "timeout" if isinstance(error, str) and "timeout" in error.lower() else ("error" if error else "ok")
        self.logger.info(f"Completed {completed}/{total} - {sample.uid} in {time.time() - start:.1f}s ({status})")
        
        self._save_incremental_result(result)
        return result
    
    def generate_responses(self, orchestrator_factory: Callable[[], Any]) -> Dict[str, Any]:
        """Generate model responses for all BrowseComp samples.
        
        Args:
            orchestrator_factory: Factory function that creates orchestrator instances.
                                  Called once per sample (fresh instance for each).
        """
        samples = load_browsecomp_samples(limit=self.limit, shuffle=self.shuffle, seed=self.seed, cache_dir=self.cache_dir)
        results: Dict[str, Any] = {}
        total = len(samples)
        self._completed_count = 0
        
        self.logger.info(f"Generating responses for {total} BrowseComp samples (browsing={self.with_browsing}, concurrency={self.concurrency}, timeout={self.sample_timeout}s)")
        
        if self.concurrency == 1:
            # Sequential execution
            for idx, sample in enumerate(samples, 1):
                result = self._process_single_sample(sample, orchestrator_factory, idx, total)
                results[sample.uid] = result
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(self._process_single_sample, sample, orchestrator_factory, idx, total): sample
                    for idx, sample in enumerate(samples, 1)
                }
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        result = future.result()
                        results[sample.uid] = result
                    except Exception as e:
                        self.logger.error(f"Unexpected error for {sample.uid}: {e}")
                        results[sample.uid] = self._create_error_result(sample, str(e), 0.0)
        
        return results
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate responses using LLM-based judging."""
        self.logger.info("Starting BrowseComp evaluation (LLM judging)")
        scorer = BrowseCompScorer(judge_orchestrator=self.judge_orchestrator)
        scored = []
        
        def score_entry(uid: str, is_correct: bool, data: dict, error: Optional[str] = None) -> dict:
            return {"uid": uid, "is_correct": is_correct, "confidence": data.get("confidence", 100.0), 
                    "error": error, "response_time_seconds": data.get("response_time_seconds", 0.0)}
        
        for uid, data in results.items():
            if data.get("error"):
                scored.append(score_entry(uid, False, data, data.get("error")))
                continue
            
            try:
                judge = scorer.judge_response(data["question"], data["model_response"], data["answer"])
                self._results[uid] = BrowseCompResult(
                    uid=uid, question=data["question"], answer=data["answer"],
                    model_response=data["model_response"], extracted_answer=data.get("extracted_answer", ""),
                    confidence=data.get("confidence", 100.0), is_correct=judge.is_correct,
                    response_time_seconds=data.get("response_time_seconds", 0.0),
                    metadata={"judge_response": judge.raw_response, "judge_reasoning": judge.reasoning},
                )
                scored.append(score_entry(uid, judge.is_correct, data))
            except Exception as e:
                self.logger.error(f"Error judging {uid}: {e}")
                scored.append(score_entry(uid, False, data, f"judge_error: {e}"))
        
        metrics = compute_metrics(scored)
        self.logger.info(f"BrowseComp Evaluation Complete: {metrics['accuracy']:.1f}% accuracy ({metrics['correct_count']}/{metrics['valid_count']} correct), ECE: {metrics['ece_10bin']:.1f}%")
        return metrics
    
    def get_results(self) -> Dict[str, BrowseCompResult]:
        return self._results.copy()


@register_benchmark("browsecomp")
def _create_browsecomp_benchmark(options: Optional[Dict[str, Any]] = None) -> BrowseCompBenchmark:
    """Create a BrowseComp benchmark instance."""
    return BrowseCompBenchmark(**(options or {}))
