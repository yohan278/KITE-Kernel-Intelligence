# benchmarks/apex/main.py
"""
APEX (AI Productivity Index Extended) Benchmark.

Evaluates AI models on economically valuable tasks across professional domains:
- Investment Banking
- Management Consulting  
- Law
- Medicine

Uses the Mercor APEX-v1-extended dataset from HuggingFace.
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
from evals.benchmarks.apex.dataset import (
    APEXSample,
    get_apex_domains,
    load_apex_samples,
)
from evals.benchmarks.apex.agents import (
    APEXAgentsSample,
    get_apex_agents_job_categories,
    load_apex_agents_samples,
)
from evals.benchmarks.apex.grading.config import (
    GradingModelConfig,
    GradingResult,
    GradingTask,
)
from evals.benchmarks.apex.grading.executor import (
    grade_solution_against_rubric,
    run_grading_task_async,
)
from evals.benchmarks.apex.prompt import load_grading_prompt


@dataclass
class APEXResult:
    """Result from evaluating a single APEX sample."""
    
    task_id: str
    domain: str
    response: str
    grading_result: Optional[GradingResult] = None
    response_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def score(self) -> float:
        """Get the percentage score (0-100)."""
        if self.grading_result:
            return self.grading_result.percentage_score
        return 0.0
    
    @property
    def passed(self) -> bool:
        """Check if the sample passed (>= 50% score)."""
        return self.score >= 50.0


class APEXBenchmark(DatasetBenchmark):
    """
    APEX benchmark for evaluating AI on professional domain tasks.
    
    This benchmark:
    1. Loads tasks from the Mercor APEX-v1-extended dataset
    2. Generates model responses using the provided orchestrator
    3. Grades responses against task-specific rubrics using LLM-as-judge
    
    Example usage:
        >>> from agents import React
        >>> benchmark = APEXBenchmark(limit=10, domains=["Law"])
        >>> orchestrator = ReactOrchestrater(model="gpt-4")
        >>> results = benchmark.run_benchmark(orchestrator)
    """
    
    def __init__(
        self,
        *,
        # Dataset options
        split: str = "train",
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        domains: Optional[List[str]] = None,
        # Grading options
        grading_model: str = "gemini-2.5-flash",
        grading_temperature: float = 0.01,
        grading_max_tokens: int = 4096,
        grading_api_key: Optional[str] = None,
        grading_prompt_template: Optional[str] = None,
        # Execution options
        max_concurrent_grading: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the APEX benchmark.
        
        Args:
            split: Dataset split to use (default: "train")
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            limit: Maximum number of samples to evaluate
            domains: Filter to specific domains (e.g., ["Law", "Medicine"])
            grading_model: LLM model for grading (default: "gemini-2.5-flash")
            grading_temperature: Temperature for grading model
            grading_max_tokens: Max tokens for grading responses
            grading_api_key: API key for grading model (optional, uses env vars)
            grading_prompt_template: Custom grading prompt template (optional)
            max_concurrent_grading: Max concurrent grading calls
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=None)
        
        # Dataset config
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.domains = domains
        
        # Grading config
        self.grading_model_config = GradingModelConfig(
            model_id=grading_model,
            temperature=grading_temperature,
            max_tokens=grading_max_tokens,
            api_key=grading_api_key,
        )
        self.grading_prompt_template = grading_prompt_template or load_grading_prompt()
        self.max_concurrent_grading = max_concurrent_grading
        
        # Results storage
        self._samples: List[APEXSample] = []
        self._results: Dict[str, APEXResult] = {}
    
    def _load_samples(self) -> List[APEXSample]:
        """Load samples from the dataset."""
        if not self._samples:
            self._samples = list(load_apex_samples(
                split=self.split,
                shuffle=self.shuffle,
                seed=self.seed,
                limit=self.limit,
                domains=self.domains,
            ))
            self.logger.info(f"Loaded {len(self._samples)} APEX samples")
            
            if self.domains:
                self.logger.info(f"Filtered to domains: {self.domains}")
        
        return self._samples
    
    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """
        Generate model responses for all APEX samples.
        
        Args:
            orchestrator: An orchestrator instance with a run() method.
        
        Returns:
            Dict mapping task_id to response data including raw response and timing.
        """
        samples = self._load_samples()
        results: Dict[str, Any] = {}
        
        self.logger.info(f"Generating responses for {len(samples)} APEX samples")
        
        for idx, sample in enumerate(samples, 1):
            self.logger.info(f"[{idx}/{len(samples)}] Processing {sample.task_id} ({sample.domain})")
            
            start_time = time.time()
            try:
                # Get the formatted prompt for the model
                prompt = sample.get_full_prompt()
                
                # Run the orchestrator
                response = orchestrator.run(prompt)
                response_text = str(response) if response else ""
                
                elapsed = time.time() - start_time
                self.logger.info(f"[{idx}/{len(samples)}] Generated response in {elapsed:.2f}s")
                
                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "domain": sample.domain,
                    "response": response_text,
                    "response_time_seconds": elapsed,
                    "rubric": sample.rubric,
                    "error": None,
                }
                
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"[{idx}/{len(samples)}] Error generating response: {e}")
                
                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "domain": sample.domain,
                    "response": "",
                    "response_time_seconds": elapsed,
                    "rubric": sample.rubric,
                    "error": str(e),
                }
        
        return results
    
    async def _grade_single_response(
        self,
        task_id: str,
        response: str,
        rubric: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> GradingResult:
        """Grade a single response against its rubric."""
        async with semaphore:
            self.logger.debug(f"Grading {task_id}")
            
            result = await grade_solution_against_rubric(
                solution=response,
                rubric=rubric,
                grading_model_config=self.grading_model_config.model_dump(),
                grading_prompt_template=self.grading_prompt_template,
            )
            
            return GradingResult(**result)
    
    async def _grade_all_responses_async(
        self,
        response_results: Dict[str, Any],
    ) -> Dict[str, GradingResult]:
        """Grade all responses concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent_grading)
        
        tasks = {}
        for task_id, data in response_results.items():
            if data.get("error"):
                # Skip errored responses
                continue
            
            response = data.get("response", "")
            rubric = data.get("rubric", {})
            
            if not rubric:
                self.logger.warning(f"No rubric for {task_id}, skipping grading")
                continue
            
            tasks[task_id] = self._grade_single_response(
                task_id=task_id,
                response=response,
                rubric=rubric,
                semaphore=semaphore,
            )
        
        self.logger.info(f"Grading {len(tasks)} responses with max {self.max_concurrent_grading} concurrent")
        
        results = {}
        if tasks:
            grading_results = await asyncio.gather(
                *tasks.values(),
                return_exceptions=True,
            )
            
            for task_id, result in zip(tasks.keys(), grading_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Grading error for {task_id}: {result}")
                    results[task_id] = GradingResult(
                        points_earned=0.0,
                        points_possible=0,
                        percentage_score=0.0,
                        criteria_results=[],
                        grading_error=str(result),
                        execution_time_seconds=0.0,
                        total_grading_tokens=0,
                        total_grading_cost=0.0,
                    )
                else:
                    results[task_id] = result
        
        return results
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using LLM-as-judge grading.
        
        Args:
            results: Dict from generate_responses() with response data.
        
        Returns:
            Dict with aggregate metrics (scores, pass rates, etc.)
        """
        self.logger.info("Starting APEX evaluation (grading responses)")
        
        # Run async grading
        grading_results = asyncio.run(self._grade_all_responses_async(results))
        
        # Build APEXResult objects
        self._results = {}
        for task_id, data in results.items():
            grading_result = grading_results.get(task_id)
            
            self._results[task_id] = APEXResult(
                task_id=task_id,
                domain=data.get("domain", ""),
                response=data.get("response", ""),
                grading_result=grading_result,
                response_time_seconds=data.get("response_time_seconds", 0.0),
                error=data.get("error"),
            )
        
        # Calculate aggregate metrics
        valid_results = [r for r in self._results.values() if r.grading_result and not r.error]
        
        if not valid_results:
            self.logger.warning("No valid grading results to evaluate")
            return {
                "total_samples": float(len(results)),
                "graded_samples": 0.0,
                "average_score": 0.0,
                "pass_rate": 0.0,
            }
        
        total_score = sum(r.score for r in valid_results)
        passed_count = sum(1 for r in valid_results if r.passed)
        
        average_score = total_score / len(valid_results)
        pass_rate = (passed_count / len(valid_results)) * 100
        
        # Calculate per-domain metrics
        domain_scores: Dict[str, List[float]] = {}
        for r in valid_results:
            if r.domain not in domain_scores:
                domain_scores[r.domain] = []
            domain_scores[r.domain].append(r.score)
        
        metrics = {
            "total_samples": float(len(results)),
            "graded_samples": float(len(valid_results)),
            "error_samples": float(len(results) - len(valid_results)),
            "average_score": round(average_score, 2),
            "pass_rate": round(pass_rate, 2),
            "passed_count": float(passed_count),
            "failed_count": float(len(valid_results) - passed_count),
        }
        
        # Add per-domain scores
        for domain, scores in domain_scores.items():
            domain_key = domain.lower().replace(" ", "_")
            metrics[f"domain_{domain_key}_average"] = round(sum(scores) / len(scores), 2)
            metrics[f"domain_{domain_key}_count"] = float(len(scores))
        
        # Calculate token/cost metrics
        total_tokens = sum(
            r.grading_result.total_grading_tokens 
            for r in valid_results 
            if r.grading_result
        )
        total_cost = sum(
            r.grading_result.total_grading_cost 
            for r in valid_results 
            if r.grading_result
        )
        total_grading_time = sum(
            r.grading_result.execution_time_seconds 
            for r in valid_results 
            if r.grading_result
        )
        
        metrics["total_grading_tokens"] = float(total_tokens)
        metrics["total_grading_cost"] = round(total_cost, 4)
        metrics["total_grading_time_seconds"] = round(total_grading_time, 2)
        
        self.logger.info(f"APEX Evaluation Complete: {average_score:.1f}% average, {pass_rate:.1f}% pass rate")
        
        return metrics
    
    def get_results(self) -> Dict[str, APEXResult]:
        """Get detailed results for each sample."""
        return self._results.copy()
    
    def get_domain_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get score breakdown by domain."""
        breakdown: Dict[str, Dict[str, Any]] = {}
        
        for r in self._results.values():
            if r.domain not in breakdown:
                breakdown[r.domain] = {
                    "total": 0,
                    "passed": 0,
                    "scores": [],
                }
            
            breakdown[r.domain]["total"] += 1
            if r.passed:
                breakdown[r.domain]["passed"] += 1
            breakdown[r.domain]["scores"].append(r.score)
        
        # Calculate averages
        for domain, data in breakdown.items():
            if data["scores"]:
                data["average_score"] = round(sum(data["scores"]) / len(data["scores"]), 2)
                data["pass_rate"] = round((data["passed"] / data["total"]) * 100, 2)
            else:
                data["average_score"] = 0.0
                data["pass_rate"] = 0.0
        
        return breakdown


@dataclass
class APEXAgentsResult:
    """Result from evaluating a single APEX-Agents sample."""

    task_id: str
    job_category: str
    response: str
    grading_result: Optional[GradingResult] = None
    response_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Get the percentage score (0-100)."""
        if self.grading_result:
            return self.grading_result.percentage_score
        return 0.0

    @property
    def passed(self) -> bool:
        """Check if the sample passed (100% score = all criteria met)."""
        # APEX-Agents uses Pass@1: all criteria must be met
        return self.score == 100.0

    @property
    def all_criteria_met(self) -> bool:
        """Check if all rubric criteria were met (Pass@1 metric)."""
        if self.grading_result:
            return self.grading_result.points_earned == self.grading_result.points_possible
        return False


class APEXAgentsBenchmark(DatasetBenchmark):
    """
    APEX-Agents benchmark for evaluating AI on agentic professional tasks.

    This benchmark:
    1. Loads tasks from the mercor/apex-agents dataset
    2. Generates model responses using the provided orchestrator
    3. Grades responses against binary rubric criteria using LLM-as-judge

    Key difference from APEX-v1: Uses Pass@1 metric (all criteria must pass).

    Example usage:
        >>> benchmark = APEXAgentsBenchmark(limit=10, job_categories=["law"])
        >>> orchestrator = SomeOrchestrator(model="gpt-4o")
        >>> results = benchmark.run_benchmark(orchestrator)
    """

    def __init__(
        self,
        *,
        # Dataset options
        split: str = "train",
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        job_categories: Optional[List[str]] = None,
        world_ids: Optional[List[str]] = None,
        min_criteria: Optional[int] = None,
        max_criteria: Optional[int] = None,
        # Grading options
        grading_model: str = "gemini-2.5-pro",  # APEX-Agents recommends Gemini 2.5 Pro
        grading_temperature: float = 0.01,
        grading_max_tokens: int = 4096,
        grading_api_key: Optional[str] = None,
        grading_prompt_template: Optional[str] = None,
        # Execution options
        max_concurrent_grading: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the APEX-Agents benchmark.

        Args:
            split: Dataset split to use (default: "train")
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            limit: Maximum number of samples to evaluate
            job_categories: Filter to specific job categories
                           (e.g., ["investment_banking", "law", "management_consulting"])
            world_ids: Filter to specific world/context IDs
            min_criteria: Minimum number of rubric criteria
            max_criteria: Maximum number of rubric criteria
            grading_model: LLM model for grading (default: "gemini-2.5-pro")
            grading_temperature: Temperature for grading model
            grading_max_tokens: Max tokens for grading responses
            grading_api_key: API key for grading model (optional, uses env vars)
            grading_prompt_template: Custom grading prompt template (optional)
            max_concurrent_grading: Max concurrent grading calls
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=None)

        # Dataset config
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.job_categories = job_categories
        self.world_ids = world_ids
        self.min_criteria = min_criteria
        self.max_criteria = max_criteria

        # Grading config
        self.grading_model_config = GradingModelConfig(
            model_id=grading_model,
            temperature=grading_temperature,
            max_tokens=grading_max_tokens,
            api_key=grading_api_key,
        )
        self.grading_prompt_template = grading_prompt_template or load_grading_prompt()
        self.max_concurrent_grading = max_concurrent_grading

        # Results storage
        self._samples: List[APEXAgentsSample] = []
        self._results: Dict[str, APEXAgentsResult] = {}

    def _load_samples(self) -> List[APEXAgentsSample]:
        """Load samples from the dataset."""
        if not self._samples:
            self._samples = list(load_apex_agents_samples(
                split=self.split,
                shuffle=self.shuffle,
                seed=self.seed,
                limit=self.limit,
                job_categories=self.job_categories,
                world_ids=self.world_ids,
                min_criteria=self.min_criteria,
                max_criteria=self.max_criteria,
            ))
            self.logger.info(f"Loaded {len(self._samples)} APEX-Agents samples")

            if self.job_categories:
                self.logger.info(f"Filtered to job categories: {self.job_categories}")

        return self._samples

    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """
        Generate model responses for all APEX-Agents samples.

        Args:
            orchestrator: An orchestrator instance with a run() method.

        Returns:
            Dict mapping task_id to response data including raw response and timing.
        """
        samples = self._load_samples()
        results: Dict[str, Any] = {}

        self.logger.info(f"Generating responses for {len(samples)} APEX-Agents samples")

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"[{idx}/{len(samples)}] Processing {sample.task_id} "
                f"({sample.job_category}, {sample.num_criteria} criteria)"
            )

            start_time = time.time()
            try:
                # Get the formatted prompt for the model
                prompt = sample.get_full_prompt()

                # Run the orchestrator
                response = orchestrator.run(prompt)
                response_text = str(response) if response else ""

                elapsed = time.time() - start_time
                self.logger.info(f"[{idx}/{len(samples)}] Generated response in {elapsed:.2f}s")

                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "job_category": sample.job_category,
                    "response": response_text,
                    "response_time_seconds": elapsed,
                    "rubric": sample.rubric,
                    "error": None,
                    "num_criteria": sample.num_criteria,
                    "world_id": sample.world_id,
                }

            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"[{idx}/{len(samples)}] Error generating response: {e}")

                results[sample.task_id] = {
                    "task_id": sample.task_id,
                    "job_category": sample.job_category,
                    "response": "",
                    "response_time_seconds": elapsed,
                    "rubric": sample.rubric,
                    "error": str(e),
                    "num_criteria": sample.num_criteria,
                    "world_id": sample.world_id,
                }

        return results

    async def _grade_single_response(
        self,
        task_id: str,
        response: str,
        rubric: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> GradingResult:
        """Grade a single response against its rubric."""
        async with semaphore:
            self.logger.debug(f"Grading {task_id}")

            result = await grade_solution_against_rubric(
                solution=response,
                rubric=rubric,
                grading_model_config=self.grading_model_config.model_dump(),
                grading_prompt_template=self.grading_prompt_template,
            )

            return GradingResult(**result)

    async def _grade_all_responses_async(
        self,
        response_results: Dict[str, Any],
    ) -> Dict[str, GradingResult]:
        """Grade all responses concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent_grading)

        tasks = {}
        for task_id, data in response_results.items():
            if data.get("error"):
                continue

            response = data.get("response", "")
            rubric = data.get("rubric", {})

            if not rubric:
                self.logger.warning(f"No rubric for {task_id}, skipping grading")
                continue

            tasks[task_id] = self._grade_single_response(
                task_id=task_id,
                response=response,
                rubric=rubric,
                semaphore=semaphore,
            )

        self.logger.info(
            f"Grading {len(tasks)} responses with max {self.max_concurrent_grading} concurrent"
        )

        results = {}
        if tasks:
            grading_results = await asyncio.gather(
                *tasks.values(),
                return_exceptions=True,
            )

            for task_id, result in zip(tasks.keys(), grading_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Grading error for {task_id}: {result}")
                    results[task_id] = GradingResult(
                        points_earned=0.0,
                        points_possible=0,
                        percentage_score=0.0,
                        criteria_results=[],
                        grading_error=str(result),
                        execution_time_seconds=0.0,
                        total_grading_tokens=0,
                        total_grading_cost=0.0,
                    )
                else:
                    results[task_id] = result

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using LLM-as-judge grading.

        Uses Pass@1 metric: all rubric criteria must be Met for a task to pass.

        Args:
            results: Dict from generate_responses() with response data.

        Returns:
            Dict with aggregate metrics (pass@1, average score, breakdown by category)
        """
        self.logger.info("Starting APEX-Agents evaluation (grading responses)")

        # Run async grading
        grading_results = asyncio.run(self._grade_all_responses_async(results))

        # Build APEXAgentsResult objects
        self._results = {}
        for task_id, data in results.items():
            grading_result = grading_results.get(task_id)

            self._results[task_id] = APEXAgentsResult(
                task_id=task_id,
                job_category=data.get("job_category", ""),
                response=data.get("response", ""),
                grading_result=grading_result,
                response_time_seconds=data.get("response_time_seconds", 0.0),
                error=data.get("error"),
                metadata={
                    "num_criteria": data.get("num_criteria", 0),
                    "world_id": data.get("world_id", ""),
                },
            )

        # Calculate aggregate metrics
        valid_results = [
            r for r in self._results.values()
            if r.grading_result and not r.error
        ]

        if not valid_results:
            self.logger.warning("No valid grading results to evaluate")
            return {
                "total_samples": float(len(results)),
                "graded_samples": 0.0,
                "pass_at_1": 0.0,
                "average_score": 0.0,
            }

        # Pass@1: all criteria must be met
        pass_at_1_count = sum(1 for r in valid_results if r.all_criteria_met)
        pass_at_1 = (pass_at_1_count / len(valid_results)) * 100

        # Average percentage score
        total_score = sum(r.score for r in valid_results)
        average_score = total_score / len(valid_results)

        # Calculate per-job-category metrics
        category_results: Dict[str, Dict[str, Any]] = {}
        for r in valid_results:
            cat = r.job_category
            if cat not in category_results:
                category_results[cat] = {"scores": [], "passed": 0, "total": 0}
            category_results[cat]["scores"].append(r.score)
            category_results[cat]["total"] += 1
            if r.all_criteria_met:
                category_results[cat]["passed"] += 1

        metrics = {
            "total_samples": float(len(results)),
            "graded_samples": float(len(valid_results)),
            "error_samples": float(len(results) - len(valid_results)),
            "pass_at_1": round(pass_at_1, 2),
            "pass_at_1_count": float(pass_at_1_count),
            "average_score": round(average_score, 2),
        }

        # Add per-category metrics
        for category, data in category_results.items():
            cat_key = category.lower().replace(" ", "_")
            cat_pass_at_1 = (data["passed"] / data["total"]) * 100 if data["total"] > 0 else 0
            cat_avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            metrics[f"category_{cat_key}_pass_at_1"] = round(cat_pass_at_1, 2)
            metrics[f"category_{cat_key}_average"] = round(cat_avg, 2)
            metrics[f"category_{cat_key}_count"] = float(data["total"])

        # Calculate token/cost metrics
        total_tokens = sum(
            r.grading_result.total_grading_tokens
            for r in valid_results
            if r.grading_result
        )
        total_cost = sum(
            r.grading_result.total_grading_cost
            for r in valid_results
            if r.grading_result
        )
        total_grading_time = sum(
            r.grading_result.execution_time_seconds
            for r in valid_results
            if r.grading_result
        )

        metrics["total_grading_tokens"] = float(total_tokens)
        metrics["total_grading_cost"] = round(total_cost, 4)
        metrics["total_grading_time_seconds"] = round(total_grading_time, 2)

        self.logger.info(
            f"APEX-Agents Evaluation Complete: {pass_at_1:.1f}% pass@1, "
            f"{average_score:.1f}% average score"
        )

        return metrics

    def get_results(self) -> Dict[str, APEXAgentsResult]:
        """Get detailed results for each sample."""
        return self._results.copy()

    def get_category_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get score breakdown by job category."""
        breakdown: Dict[str, Dict[str, Any]] = {}

        for r in self._results.values():
            cat = r.job_category
            if cat not in breakdown:
                breakdown[cat] = {
                    "total": 0,
                    "passed": 0,
                    "scores": [],
                }

            breakdown[cat]["total"] += 1
            if r.all_criteria_met:
                breakdown[cat]["passed"] += 1
            breakdown[cat]["scores"].append(r.score)

        # Calculate averages
        for cat, data in breakdown.items():
            if data["scores"]:
                data["average_score"] = round(sum(data["scores"]) / len(data["scores"]), 2)
                data["pass_at_1"] = round((data["passed"] / data["total"]) * 100, 2)
            else:
                data["average_score"] = 0.0
                data["pass_at_1"] = 0.0

        return breakdown


@register_benchmark("apex")
def _create_apex_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> APEXBenchmark:
    """
    Create an APEX benchmark instance.
    
    Options:
        Dataset options:
        - split: Dataset split (default: "train")
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - domains: List of domains to filter (default: None = all)
        
        Grading options:
        - grading_model: LLM for grading (default: "gemini-2.5-flash")
        - grading_temperature: Temperature (default: 0.01)
        - grading_max_tokens: Max tokens (default: 4096)
        - grading_api_key: API key (default: None, uses env vars)
        - grading_prompt_template: Custom prompt (default: None)
        
        Execution options:
        - max_concurrent_grading: Concurrent grading calls (default: 5)
    
    Example:
        >>> benchmark = get_benchmark("apex")(options={
        ...     "limit": 10,
        ...     "domains": ["Law", "Medicine"],
        ...     "grading_model": "gpt-4o",
        ... })
    """
    options = options or {}
    return APEXBenchmark(**options)


@register_benchmark("apex-agents")
def _create_apex_agents_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> APEXAgentsBenchmark:
    """
    Create an APEX-Agents benchmark instance.

    Options:
        Dataset options:
        - split: Dataset split (default: "train")
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed for shuffling (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - job_categories: List of job categories to filter
                         (["investment_banking", "law", "management_consulting"])
        - world_ids: List of world IDs to filter
        - min_criteria: Minimum number of rubric criteria
        - max_criteria: Maximum number of rubric criteria

        Grading options:
        - grading_model: LLM for grading (default: "gemini-2.5-pro")
        - grading_temperature: Temperature (default: 0.01)
        - grading_max_tokens: Max tokens (default: 4096)
        - grading_api_key: API key (default: None, uses env vars)
        - grading_prompt_template: Custom prompt (default: None)

        Execution options:
        - max_concurrent_grading: Concurrent grading calls (default: 5)

    Example:
        >>> benchmark = get_benchmark("apex-agents")(options={
        ...     "limit": 10,
        ...     "job_categories": ["law", "investment_banking"],
        ...     "grading_model": "gemini-2.5-pro",
        ... })
    """
    options = options or {}
    return APEXAgentsBenchmark(**options)

