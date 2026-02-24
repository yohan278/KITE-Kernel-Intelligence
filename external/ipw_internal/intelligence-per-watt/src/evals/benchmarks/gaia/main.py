# benchmarks/gaia/main.py
"""
GAIA (General AI Assistant) Benchmark.

Evaluates AI models on real-world questions requiring multi-step reasoning,
tool use, and information synthesis across diverse domains.

Uses the gaia-benchmark/GAIA dataset from HuggingFace.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.gaia.dataset import GAIASample, load_gaia_samples
from evals.benchmarks.gaia.scorer import question_scorer


@dataclass
class GAIAResult:
    """Result from evaluating a single GAIA sample."""
    
    task_id: str
    question: str
    ground_truth: str
    model_answer: str
    level: int
    is_correct: bool
    response_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GAIABenchmark(DatasetBenchmark):
    """
    GAIA benchmark for evaluating AI on real-world questions.
    
    This benchmark:
    1. Loads questions from the gaia-benchmark/GAIA dataset
    2. Generates model responses using the provided orchestrator
    3. Scores responses using the GAIA evaluation metrics
    
    Example usage:
        >>> from agents import React
        >>> benchmark = GAIABenchmark(limit=10, subset="2023_level1")
        >>> orchestrator = ReactOrchestrater(model="gpt-4")
        >>> results = benchmark.run_benchmark(orchestrator)
    """
    
    def __init__(
        self,
        *,
        # Dataset options
        subset: Literal[
            "2023_all", "2023_level1", "2023_level2", "2023_level3"
        ] = "2023_all",
        split: Literal["test", "validation"] = "validation",
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        input_prompt: Optional[str] = None,
        # Execution options
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the GAIA benchmark.
        
        Args:
            subset: Dataset subset (level filtering)
            split: Dataset split ("test" or "validation")
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            limit: Maximum number of samples to evaluate
            cache_dir: Cache directory for dataset files
            input_prompt: Custom input prompt template
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=None)
        
        # Dataset config
        self.subset = subset
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.cache_dir = cache_dir
        self.input_prompt = input_prompt
        
        # Results storage
        self._samples: List[GAIASample] = []
        self._results: Dict[str, GAIAResult] = {}
    
    def _load_samples(self) -> List[GAIASample]:
        """Load samples from the dataset."""
        if not self._samples:
            self._samples = list(load_gaia_samples(
                subset=self.subset,
                split=self.split,
                shuffle=self.shuffle,
                seed=self.seed,
                limit=self.limit,
                cache_dir=self.cache_dir,
            ))
            self.logger.info(f"Loaded {len(self._samples)} GAIA samples")
            self.logger.info(f"Subset: {self.subset}, Split: {self.split}")
        
        return self._samples
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text from orchestrator response.
        
        Handles different response types:
        - Agno RunOutput objects with .content attribute
        - Plain strings
        - Other objects with __str__
        """
        if response is None:
            return ""
        
        # Try to get the content attribute (Agno RunOutput, etc.)
        if hasattr(response, 'content'):
            content = getattr(response, 'content', None)
            if content is not None:
                return str(content)
        
        # Fallback to string conversion
        return str(response)
    
    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """
        Generate model responses for all GAIA samples.
        
        Args:
            orchestrator: An orchestrator instance with a run() method.
        
        Returns:
            Dict mapping task_id to response data including raw response and timing.
        """
        samples = self._load_samples()
        results: Dict[str, Any] = {}
        
        self.logger.info(f"Generating responses for {len(samples)} GAIA samples")
        
        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing {sample.task_id} "
                f"(Level {sample.level})"
            )
            
            start_time = time.time()
            try:
                # Get the formatted prompt for the model
                prompt = sample.get_prompt(self.input_prompt)
                
                # Run the orchestrator
                response = orchestrator.run(prompt)
                response_text = self._extract_response_text(response)
                
                elapsed = time.time() - start_time
                self.logger.info(
                    f"[{idx}/{len(samples)}] Generated response in {elapsed:.2f}s"
                )
                
                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "question": sample.question,
                    "ground_truth": sample.final_answer,
                    "model_answer": response_text,
                    "level": sample.level,
                    "response_time_seconds": elapsed,
                    "error": None,
                    "annotator_metadata": sample.annotator_metadata,
                }
                
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"[{idx}/{len(samples)}] Error generating response: {e}"
                )
                
                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "question": sample.question,
                    "ground_truth": sample.final_answer,
                    "model_answer": "",
                    "level": sample.level,
                    "response_time_seconds": elapsed,
                    "error": str(e),
                    "annotator_metadata": sample.annotator_metadata,
                }
        
        return results
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using GAIA scoring.
        
        Args:
            results: Dict from generate_responses() with response data.
        
        Returns:
            Dict with aggregate metrics (accuracy, level breakdown, etc.)
        """
        self.logger.info("Starting GAIA evaluation (scoring responses)")
        
        # Score each response
        self._results = {}
        for task_id, data in results.items():
            model_answer = data.get("model_answer", "")
            ground_truth = data.get("ground_truth", "")
            
            # Score the response
            is_correct = False
            if not data.get("error") and model_answer:
                try:
                    is_correct = question_scorer(model_answer, ground_truth)
                except Exception as e:
                    self.logger.error(f"Error scoring {task_id}: {e}")
            
            self._results[task_id] = GAIAResult(
                task_id=task_id,
                question=data.get("question", ""),
                ground_truth=ground_truth,
                model_answer=model_answer,
                level=data.get("level", 0),
                is_correct=is_correct,
                response_time_seconds=data.get("response_time_seconds", 0.0),
                error=data.get("error"),
                metadata={"annotator_metadata": data.get("annotator_metadata", "")},
            )
        
        # Calculate aggregate metrics
        valid_results = [
            r for r in self._results.values() 
            if not r.error
        ]
        
        if not valid_results:
            self.logger.warning("No valid results to evaluate")
            return {
                "total_samples": float(len(results)),
                "valid_samples": 0.0,
                "accuracy": 0.0,
                "correct_count": 0.0,
            }
        
        correct_count = sum(1 for r in valid_results if r.is_correct)
        accuracy = (correct_count / len(valid_results)) * 100
        
        # Calculate per-level metrics
        level_results: Dict[int, List[bool]] = {}
        for r in valid_results:
            if r.level not in level_results:
                level_results[r.level] = []
            level_results[r.level].append(r.is_correct)
        
        metrics = {
            "total_samples": float(len(results)),
            "valid_samples": float(len(valid_results)),
            "error_samples": float(len(results) - len(valid_results)),
            "accuracy": round(accuracy, 2),
            "correct_count": float(correct_count),
            "incorrect_count": float(len(valid_results) - correct_count),
        }
        
        # Add per-level accuracy
        for level, correctness_list in level_results.items():
            level_correct = sum(correctness_list)
            level_total = len(correctness_list)
            level_accuracy = (level_correct / level_total) * 100
            
            metrics[f"level_{level}_accuracy"] = round(level_accuracy, 2)
            metrics[f"level_{level}_correct"] = float(level_correct)
            metrics[f"level_{level}_total"] = float(level_total)
        
        # Calculate average response time
        avg_response_time = sum(
            r.response_time_seconds for r in valid_results
        ) / len(valid_results)
        metrics["avg_response_time_seconds"] = round(avg_response_time, 2)
        
        self.logger.info(
            f"GAIA Evaluation Complete: {accuracy:.1f}% accuracy "
            f"({correct_count}/{len(valid_results)} correct)"
        )
        
        return metrics
    
    def get_results(self) -> Dict[str, GAIAResult]:
        """Get detailed results for each sample."""
        return self._results.copy()
    
    def get_level_breakdown(self) -> Dict[int, Dict[str, Any]]:
        """Get accuracy breakdown by difficulty level."""
        breakdown: Dict[int, Dict[str, Any]] = {}
        
        for r in self._results.values():
            if r.error:
                continue
                
            if r.level not in breakdown:
                breakdown[r.level] = {
                    "total": 0,
                    "correct": 0,
                    "questions": [],
                }
            
            breakdown[r.level]["total"] += 1
            if r.is_correct:
                breakdown[r.level]["correct"] += 1
            
            breakdown[r.level]["questions"].append({
                "task_id": r.task_id,
                "is_correct": r.is_correct,
                "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
            })
        
        # Calculate accuracy
        for level, data in breakdown.items():
            if data["total"] > 0:
                data["accuracy"] = round((data["correct"] / data["total"]) * 100, 2)
            else:
                data["accuracy"] = 0.0
        
        return breakdown


@register_benchmark("gaia")
def _create_gaia_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> GAIABenchmark:
    """
    Create a GAIA benchmark instance.
    
    Options:
        Dataset options:
        - subset: Dataset subset (default: "2023_all")
                 Options: "2023_all", "2023_level1", "2023_level2", "2023_level3"
        - split: Dataset split (default: "validation")
                Options: "test", "validation"
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - cache_dir: Cache directory (default: ~/.cache/gaia_benchmark)
        - input_prompt: Custom prompt template (default: None)
    
    Example:
        >>> benchmark = get_benchmark("gaia")(options={
        ...     "limit": 10,
        ...     "subset": "2023_level1",
        ...     "split": "validation",
        ... })
    """
    options = options or {}
    return GAIABenchmark(**options)
