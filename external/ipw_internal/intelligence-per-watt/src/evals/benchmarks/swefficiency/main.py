# benchmarks/swefficiency/main.py
"""
SWEfficiency Benchmark.

Evaluates AI agents on software performance optimization tasks by:
1. Loading tasks from the SWEfficiency dataset
2. Running agents to generate optimization patches
3. Evaluating patches using Docker containers (similar to SWE-bench)

Metrics:
- resolve_rate: Fraction of tasks where optimization passes tests
- speedup_achieved: Actual speedup compared to expected

Uses the swefficiency/swefficiency dataset from HuggingFace.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.swefficiency.dataset import (
    SWEfficiencySample,
    load_swefficiency_samples,
)


@dataclass
class SWEfficiencyResult:
    """Result from evaluating a single SWEfficiency sample."""

    instance_id: str
    repo: str
    patch: str              # Generated optimization patch
    agent_output: str
    response_time_seconds: float = 0.0
    error: Optional[str] = None
    # Evaluation results (populated after running evaluation)
    resolved: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    speedup_achieved: float = 0.0  # Actual speedup from optimization
    expected_speedup: float = 0.0  # Expected speedup from ground truth
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SWEfficiencyConfig:
    """Configuration for the SWEfficiency benchmark."""

    # Dataset options
    split: str = "test"
    shuffle: bool = False
    seed: int = 42
    limit: Optional[int] = None
    repos: Optional[List[str]] = None
    instance_ids: Optional[List[str]] = None
    min_speedup: Optional[float] = None
    max_speedup: Optional[float] = None

    # Agent options
    agent_type: str = "react"  # "react" or "openhands"
    model: str = "gpt-4o"
    provider: str = "openai"
    base_url: Optional[str] = None  # Base URL for OpenAI-compatible APIs (e.g. vLLM)
    max_iterations: int = 10

    # Container options
    agent_timeout: int = 1800  # 30 minutes per task

    # Execution options
    num_workers: int = 1  # Parallel workers for generation
    eval_workers: int = 4  # Parallel workers for evaluation

    # Evaluation options
    run_evaluation: bool = True  # Whether to run evaluation harness
    benchmark_iterations: int = 3  # Number of iterations for performance measurement
    benchmark_warmup: int = 1  # Warmup iterations before measurement

    # Output options
    output_dir: Optional[Path] = None
    run_id: Optional[str] = None


class SWEfficiencyBenchmark(DatasetBenchmark):
    """
    SWEfficiency benchmark for evaluating AI on performance optimization.

    This benchmark:
    1. Loads tasks from the SWEfficiency dataset
    2. For each task, runs an agent to generate an optimization patch
    3. Optionally evaluates patches in Docker containers

    Example usage:
        >>> from evals.benchmarks.swefficiency import SWEfficiencyBenchmark, SWEfficiencyConfig
        >>> config = SWEfficiencyConfig(limit=5)
        >>> benchmark = SWEfficiencyBenchmark(config)
        >>> results = benchmark.run_benchmark()
    """

    def __init__(
        self,
        config: Optional[SWEfficiencyConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SWEfficiency benchmark.

        Args:
            config: Benchmark configuration
            logger: Optional logger instance
        """
        super().__init__(logger=logger)
        self.config = config or SWEfficiencyConfig()
        self.results: List[SWEfficiencyResult] = []

        # Set up output directory
        if self.config.output_dir is None:
            self.config.output_dir = Path(__file__).parent / "outputs"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID if not provided
        if self.config.run_id is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_short = self.config.model.replace("/", "-").replace(":", "-")
            self.config.run_id = f"{self.config.agent_type}_{model_short}_{timestamp}"

    def load_samples(self) -> List[SWEfficiencySample]:
        """Load samples from the dataset."""
        samples = list(load_swefficiency_samples(
            split=self.config.split,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
            limit=self.config.limit,
            repos=self.config.repos,
            min_speedup=self.config.min_speedup,
            max_speedup=self.config.max_speedup,
        ))

        # Filter by instance_ids if specified
        if self.config.instance_ids:
            samples = [s for s in samples if s.instance_id in self.config.instance_ids]

        self.logger.info(f"Loaded {len(samples)} samples")
        return samples

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from orchestrator response."""
        if response is None:
            return ""

        if hasattr(response, 'content'):
            content = getattr(response, 'content', None)
            if content is not None:
                return str(content)

        return str(response)

    def _extract_patch(self, agent_output: str) -> str:
        """
        Extract a git patch from agent output.

        Looks for diff/patch content between common markers.
        """
        import re

        # Try to find diff content
        patterns = [
            r"```diff\n(.*?)```",
            r"```patch\n(.*?)```",
            r"```\n(diff --git.*?)```",
            r"(diff --git.*?)(?=\n```|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, agent_output, re.DOTALL)
            if matches:
                return matches[0].strip()

        # If no explicit diff markers, check if the whole response looks like a diff
        if agent_output.strip().startswith("diff --git") or agent_output.strip().startswith("---"):
            return agent_output.strip()

        return ""

    def _run_builtin_agent(self, sample: SWEfficiencySample) -> tuple[str, str]:
        """
        Run a built-in agent on a sample to generate an optimization patch.

        For SWE Efficiency, we use a simple completion-based approach since
        we're generating patches, not executing container commands.

        Returns:
            Tuple of (agent_output, patch)
        """
        from evals.benchmarks.swebench.custom_runner import create_model

        model = create_model(
            self.config.provider, self.config.model, base_url=self.config.base_url
        )

        system_prompt = (
            "You are a software performance engineer. "
            "Analyze the provided problem and produce a unified diff patch "
            "that optimizes performance. Output your patch inside a ```diff code block."
        )
        prompt = sample.get_prompt()

        # Use direct model completion instead of React agent
        from agno.agent import Agent
        agent = Agent(model=model, instructions=system_prompt)
        response = agent.run(prompt)
        agent_output = self._extract_response_text(response)
        patch = self._extract_patch(agent_output)
        return agent_output, patch

    def generate_responses(self, orchestrator: Any = None) -> Dict[str, Any]:
        """
        Run agents on all samples to generate optimization patches.

        Args:
            orchestrator: Optional orchestrator instance with a run() method.
                         If provided, uses the orchestrator instead of internal agents.

        Returns:
            Dictionary with results and predictions file path
        """
        samples = self.load_samples()
        total = len(samples)
        results = []

        self.logger.info(f"Generating responses for {total} samples")

        for i, sample in enumerate(samples):
            self.logger.info(f"[{i+1}/{total}] {sample.instance_id} ({sample.repo})")
            start_time = time.time()

            try:
                if orchestrator is not None:
                    # Use provided orchestrator
                    prompt = sample.get_prompt()
                    response = orchestrator.run(prompt)
                    agent_output = self._extract_response_text(response)
                    patch = self._extract_patch(agent_output)
                else:
                    # Create a built-in agent using config
                    agent_output, patch = self._run_builtin_agent(sample)

                elapsed = time.time() - start_time
                self.logger.info(
                    f"[{i+1}/{total}] Generated response in {elapsed:.1f}s, "
                    f"patch length: {len(patch)}"
                )

                result = SWEfficiencyResult(
                    instance_id=sample.instance_id,
                    repo=sample.repo,
                    patch=patch,
                    agent_output=agent_output,
                    response_time_seconds=elapsed,
                    expected_speedup=sample.speedup,
                )
                results.append(result)
                self.results.append(result)

            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Error processing {sample.instance_id}: {e}")
                result = SWEfficiencyResult(
                    instance_id=sample.instance_id,
                    repo=sample.repo,
                    patch="",
                    agent_output="",
                    response_time_seconds=elapsed,
                    error=str(e),
                    expected_speedup=sample.speedup,
                )
                results.append(result)
                self.results.append(result)

        # Write predictions to JSONL file
        predictions_path = self.config.output_dir / f"{self.config.run_id}_predictions.jsonl"
        self._write_predictions(results, predictions_path)

        return {
            "results": results,
            "predictions_path": str(predictions_path),
            "run_id": self.config.run_id,
        }

    def _write_predictions(self, results: List[SWEfficiencyResult], path: Path) -> None:
        """Write results to JSONL file."""
        with open(path, "w") as f:
            for result in results:
                prediction = {
                    "instance_id": result.instance_id,
                    "model_name_or_path": f"{self.config.agent_type}_{self.config.model}",
                    "model_patch": result.patch,
                }
                f.write(json.dumps(prediction) + "\n")
        self.logger.info(f"Wrote predictions to {path}")

    def evaluate_responses(self, generation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated patches using Docker containers.

        For each patch:
        1. Start Docker container from sample's image_name
        2. Measure baseline performance (before patch)
        3. Apply patch and rebuild
        4. Run tests to verify correctness
        5. Measure optimized performance (after patch)
        6. Calculate speedup ratio

        Args:
            generation_results: Output from generate_responses()

        Returns:
            Dictionary with evaluation metrics
        """
        results: List[SWEfficiencyResult] = generation_results["results"]

        if not self.config.run_evaluation:
            return self._compute_generation_stats(results)

        # Load samples for evaluation context
        samples = {s.instance_id: s for s in self.load_samples()}

        # Filter results that have patches and matching samples
        evaluable = [
            (r, samples[r.instance_id])
            for r in results
            if r.patch and not r.error and r.instance_id in samples
        ]

        if not evaluable:
            self.logger.warning("No patches to evaluate")
            return self._compute_generation_stats(results)

        self.logger.info(
            f"Evaluating {len(evaluable)} patches with Docker "
            f"({self.config.eval_workers} workers)"
        )

        # Run evaluations in parallel
        results_lock = Lock()
        evaluated_results: Dict[str, Dict[str, Any]] = {}

        def evaluate_single(item: Tuple[SWEfficiencyResult, SWEfficiencySample]) -> None:
            result, sample = item
            eval_result = self._evaluate_single_patch(result, sample)
            with results_lock:
                evaluated_results[result.instance_id] = eval_result

        if self.config.eval_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.eval_workers) as executor:
                futures = {
                    executor.submit(evaluate_single, item): item[0].instance_id
                    for item in evaluable
                }
                for future in as_completed(futures):
                    instance_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Evaluation failed for {instance_id}: {e}")
                        with results_lock:
                            evaluated_results[instance_id] = {
                                "resolved": False,
                                "error": str(e),
                            }
        else:
            for item in evaluable:
                try:
                    evaluate_single(item)
                except Exception as e:
                    instance_id = item[0].instance_id
                    self.logger.error(f"Evaluation failed for {instance_id}: {e}")
                    evaluated_results[instance_id] = {
                        "resolved": False,
                        "error": str(e),
                    }

        # Update results with evaluation data
        for r in results:
            if r.instance_id in evaluated_results:
                eval_data = evaluated_results[r.instance_id]
                r.resolved = eval_data.get("resolved", False)
                r.tests_passed = eval_data.get("tests_passed", 0)
                r.tests_failed = eval_data.get("tests_failed", 0)
                r.speedup_achieved = eval_data.get("speedup_achieved", 0.0)
                if eval_data.get("error"):
                    r.error = eval_data["error"]

        return self._compute_evaluation_stats(results)

    def _evaluate_single_patch(
        self,
        result: SWEfficiencyResult,
        sample: SWEfficiencySample,
    ) -> Dict[str, Any]:
        """
        Evaluate a single patch in Docker.

        Returns dict with: resolved, tests_passed, tests_failed, speedup_achieved, error
        """
        from evals.benchmarks.swefficiency.env_wrapper import SWEfficiencyEnv

        self.logger.info(f"Evaluating {result.instance_id}")

        try:
            with SWEfficiencyEnv.from_sample(sample) as env:
                # 1. Measure baseline performance (before patch)
                self.logger.debug(f"[{result.instance_id}] Measuring baseline performance")
                baseline_result = env.measure_performance(
                    iterations=self.config.benchmark_iterations,
                    warmup=self.config.benchmark_warmup,
                )
                baseline_time = baseline_result.execution_time

                if baseline_time <= 0:
                    self.logger.warning(
                        f"[{result.instance_id}] Invalid baseline time: {baseline_time}"
                    )
                    baseline_time = 1.0  # Avoid division by zero

                # 2. Apply patch
                self.logger.debug(f"[{result.instance_id}] Applying patch")
                patch_success, patch_output = env.apply_patch(result.patch)

                if not patch_success:
                    return {
                        "resolved": False,
                        "tests_passed": 0,
                        "tests_failed": 0,
                        "speedup_achieved": 0.0,
                        "error": f"Patch application failed: {patch_output[:200]}",
                    }

                # 3. Rebuild
                self.logger.debug(f"[{result.instance_id}] Rebuilding")
                rebuild_success, rebuild_output = env.rebuild()

                if not rebuild_success:
                    return {
                        "resolved": False,
                        "tests_passed": 0,
                        "tests_failed": 0,
                        "speedup_achieved": 0.0,
                        "error": f"Rebuild failed: {rebuild_output[:200]}",
                    }

                # 4. Run tests
                self.logger.debug(f"[{result.instance_id}] Running tests")
                test_result = env.run_tests()

                if not test_result.all_passed:
                    return {
                        "resolved": False,
                        "tests_passed": test_result.passed,
                        "tests_failed": test_result.failed,
                        "speedup_achieved": 0.0,
                        "error": f"Tests failed: {test_result.failed} failures",
                    }

                # 5. Measure optimized performance (after patch)
                self.logger.debug(f"[{result.instance_id}] Measuring optimized performance")
                optimized_result = env.measure_performance(
                    iterations=self.config.benchmark_iterations,
                    warmup=self.config.benchmark_warmup,
                )
                optimized_time = optimized_result.execution_time

                if optimized_time <= 0:
                    optimized_time = baseline_time  # No speedup if invalid

                # 6. Calculate speedup
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0

                self.logger.info(
                    f"[{result.instance_id}] Evaluation complete: "
                    f"tests_passed={test_result.passed}, "
                    f"speedup={speedup:.2f}x (expected: {result.expected_speedup:.2f}x)"
                )

                return {
                    "resolved": True,
                    "tests_passed": test_result.passed,
                    "tests_failed": test_result.failed,
                    "speedup_achieved": speedup,
                    "baseline_time": baseline_time,
                    "optimized_time": optimized_time,
                    "error": None,
                }

        except Exception as e:
            self.logger.error(f"[{result.instance_id}] Evaluation error: {e}")
            return {
                "resolved": False,
                "tests_passed": 0,
                "tests_failed": 0,
                "speedup_achieved": 0.0,
                "error": str(e),
            }

    def _compute_evaluation_stats(self, results: List[SWEfficiencyResult]) -> Dict[str, float]:
        """Compute statistics from evaluated results."""
        total = len(results)
        if total == 0:
            return {"total_samples": 0.0}

        errors = sum(1 for r in results if r.error)
        resolved = sum(1 for r in results if r.resolved)
        with_patch = sum(1 for r in results if r.patch and not r.error)

        # Speedup statistics (only for resolved samples)
        resolved_results = [r for r in results if r.resolved]
        speedups_achieved = [r.speedup_achieved for r in resolved_results if r.speedup_achieved > 0]
        expected_speedups = [r.expected_speedup for r in resolved_results if r.expected_speedup > 0]

        avg_speedup = sum(speedups_achieved) / len(speedups_achieved) if speedups_achieved else 0.0
        avg_expected = sum(expected_speedups) / len(expected_speedups) if expected_speedups else 0.0

        # Calculate how many achieved their target speedup
        target_met = sum(
            1 for r in resolved_results
            if r.speedup_achieved >= r.expected_speedup * 0.9  # Within 90% of target
        )

        avg_time = sum(r.response_time_seconds for r in results) / total

        return {
            "total_samples": float(total),
            "errors": float(errors),
            "patches_generated": float(with_patch),
            "patch_rate": with_patch / total if total > 0 else 0.0,
            "resolved": float(resolved),
            "resolve_rate": resolved / total if total > 0 else 0.0,
            "avg_speedup_achieved": round(avg_speedup, 3),
            "avg_expected_speedup": round(avg_expected, 3),
            "target_speedup_met": float(target_met),
            "target_speedup_rate": target_met / resolved if resolved > 0 else 0.0,
            "avg_response_time_seconds": round(avg_time, 2),
        }

    def _compute_generation_stats(self, results: List[SWEfficiencyResult]) -> Dict[str, float]:
        """Compute basic statistics from generation results."""
        total = len(results)
        errors = sum(1 for r in results if r.error)
        with_patch = sum(1 for r in results if r.patch and not r.error)
        avg_time = sum(r.response_time_seconds for r in results) / total if total > 0 else 0

        # Calculate speedup statistics
        expected_speedups = [r.expected_speedup for r in results if r.expected_speedup > 0]
        avg_expected_speedup = sum(expected_speedups) / len(expected_speedups) if expected_speedups else 0

        return {
            "total_samples": float(total),
            "errors": float(errors),
            "patches_generated": float(with_patch),
            "patch_rate": with_patch / total if total > 0 else 0.0,
            "avg_response_time_seconds": avg_time,
            "avg_expected_speedup": avg_expected_speedup,
            # Placeholder for actual evaluation metrics
            "resolved": 0.0,
            "resolve_rate": 0.0,
        }

    def run_benchmark(self, orchestrator: Any = None) -> Dict[str, float]:
        """
        Run the full SWEfficiency benchmark.

        Args:
            orchestrator: Optional orchestrator instance with a run() method.

        Returns:
            Dictionary with benchmark metrics
        """
        self.logger.info(f"Running SWEfficiency benchmark")
        self.logger.info(f"Agent: {self.config.agent_type}, Model: {self.config.model}")

        generation_results = self.generate_responses(orchestrator)
        evaluation_results = self.evaluate_responses(generation_results)

        # Save final results
        results_path = self.config.output_dir / f"{self.config.run_id}_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": {
                    "split": self.config.split,
                    "agent_type": self.config.agent_type,
                    "model": self.config.model,
                    "provider": self.config.provider,
                },
                "metrics": evaluation_results,
                "results": [
                    {
                        "instance_id": r.instance_id,
                        "repo": r.repo,
                        "resolved": r.resolved,
                        "error": r.error,
                        "response_time_seconds": r.response_time_seconds,
                        "patch_length": len(r.patch),
                        "expected_speedup": r.expected_speedup,
                        "speedup_achieved": r.speedup_achieved,
                    }
                    for r in self.results
                ],
            }, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")
        return evaluation_results


@register_benchmark("swefficiency")
def _create_swefficiency_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> SWEfficiencyBenchmark:
    """
    Create a SWEfficiency benchmark instance.

    Options:
        Dataset options:
        - split: Dataset split (default: "test")
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - repos: List of repos to filter (default: None = all)
        - instance_ids: List of instance IDs to filter (default: None = all)
        - min_speedup: Minimum expected speedup (default: None)
        - max_speedup: Maximum expected speedup (default: None)

        Agent options:
        - agent_type: Agent type (default: "react")
        - model: Model name (default: "gpt-4o")
        - provider: Model provider (default: "openai")
        - max_iterations: Max agent iterations (default: 10)

        Execution options:
        - agent_timeout: Timeout per task in seconds (default: 1800)
        - num_workers: Parallel workers (default: 1)
        - run_evaluation: Whether to run full evaluation (default: True)

        Output options:
        - output_dir: Output directory path (default: ./outputs)
        - run_id: Run identifier (default: auto-generated)

    Example:
        >>> benchmark = get_benchmark("swefficiency")(options={
        ...     "limit": 10,
        ...     "model": "gpt-4o",
        ...     "min_speedup": 2.0,
        ... })
    """
    options = options or {}

    # Extract config options
    config_fields = {
        "split", "shuffle", "seed", "limit", "repos", "instance_ids",
        "min_speedup", "max_speedup", "agent_type", "model", "provider",
        "base_url", "max_iterations", "agent_timeout", "num_workers",
        "run_evaluation", "output_dir", "run_id"
    }
    config_options = {k: v for k, v in options.items() if k in config_fields}

    # Handle output_dir conversion
    if "output_dir" in config_options and config_options["output_dir"]:
        config_options["output_dir"] = Path(config_options["output_dir"])

    config = SWEfficiencyConfig(**config_options)
    return SWEfficiencyBenchmark(config)
