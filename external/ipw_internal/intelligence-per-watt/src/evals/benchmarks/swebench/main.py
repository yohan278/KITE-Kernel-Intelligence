# benchmarks/swebench/main.py
"""
SWE-bench Benchmark.

Evaluates AI agents on real-world GitHub issues by:
1. Loading tasks from SWE-bench Verified or Verified Mini
2. Running agents in Docker containers to generate patches
3. Evaluating patches using the swebench harness

Supports two dataset variants:
- verified: princeton-nlp/SWE-bench_Verified (500 tasks)
- verified_mini: MariusHobbhahn/swe-bench-verified-mini (50 tasks)
"""
from __future__ import annotations

import glob
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from evals.base import DatasetBenchmark
from evals.registry import register_benchmark

from .dataset import (
    DatasetVariant,
    SWEBenchSample,
    load_swebench_samples,
)
from .custom_runner import run_custom_on_sample
from .openhands_runner import run_openhands_on_sample


@dataclass
class SWEBenchResult:
    """Result from evaluating a single SWE-bench sample."""
    
    instance_id: str
    repo: str
    patch: str  # Generated patch
    agent_output: str
    response_time_seconds: float = 0.0
    error: Optional[str] = None
    # Evaluation results (populated after running swebench harness)
    resolved: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SWEBenchConfig:
    """Configuration for the SWE-bench benchmark."""
    
    # Dataset options
    dataset: DatasetVariant = "verified_mini"
    split: str = "test"
    shuffle: bool = False
    seed: int = 42
    limit: Optional[int] = None
    repos: Optional[List[str]] = None
    instance_ids: Optional[List[str]] = None
    
    # Agent options
    agent_type: str = "react"  # "react" or "openhands"
    model: str = "gpt-4o"
    provider: str = "openai"
    base_url: Optional[str] = None  # Base URL for OpenAI-compatible APIs (e.g. vLLM)
    max_iterations: int = 10  # Max agent iterations per task
    max_retries: int = 0  # Max retries on exceptions (OpenHands only)
    
    # Container options
    agent_timeout: int = 1800  # 30 minutes per task
    
    # Execution options
    num_workers: int = 1  # Parallel workers for running agents (1 = sequential)
    
    # Evaluation options
    run_evaluation: bool = True  # Whether to run swebench harness
    eval_workers: int = 4  # Parallel workers for swebench evaluation harness
    
    # Output options
    output_dir: Optional[Path] = None
    run_id: Optional[str] = None


class SWEBenchBenchmark(DatasetBenchmark):
    """
    SWE-bench benchmark for evaluating AI agents on GitHub issues.
    
    This benchmark:
    1. Loads tasks from SWE-bench Verified (500) or Verified Mini (50)
    2. For each task, runs an agent in a Docker container
    3. Collects the generated patch (git diff)
    4. Optionally evaluates patches using swebench harness
    
    Example usage:
        >>> from evals.benchmarks.swebench import SWEBenchBenchmark, SWEBenchConfig
        >>> config = SWEBenchConfig(dataset="verified_mini", limit=5)
        >>> benchmark = SWEBenchBenchmark(config)
        >>> results = benchmark.run_benchmark()
    """
    
    def __init__(
        self,
        config: Optional[SWEBenchConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SWE-bench benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Optional logger instance
        """
        super().__init__(logger=logger)
        self.config = config or SWEBenchConfig()
        self.results: List[SWEBenchResult] = []
        
        # Set up output directory (relative to this module)
        if self.config.output_dir is None:
            self.config.output_dir = Path(__file__).parent / "outputs"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID if not provided
        # Format: {agent}_{model}_{dataset}_{timestamp}
        # Example: react_gpt-4o_mini_20260119_143022
        if self.config.run_id is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_short = self.config.model.replace("/", "-").replace(":", "-")
            dataset_short = "mini" if "mini" in self.config.dataset else "verified"
            self.config.run_id = f"{self.config.agent_type}_{model_short}_{dataset_short}_{timestamp}"
    
    def load_samples(self) -> List[SWEBenchSample]:
        """Load samples from the dataset."""
        samples = list(load_swebench_samples(
            dataset=self.config.dataset,
            split=self.config.split,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
            limit=self.config.limit,
            repos=self.config.repos,
        ))
        
        # Filter by instance_ids if specified
        if self.config.instance_ids:
            samples = [s for s in samples if s.instance_id in self.config.instance_ids]
        
        self.logger.info(f"Loaded {len(samples)} samples from {self.config.dataset}")
        return samples
    
    def run_agent_on_sample(self, sample: SWEBenchSample) -> SWEBenchResult:
        """
        Run an agent on a single sample.
        
        Args:
            sample: SWEBenchSample to process
        
        Returns:
            SWEBenchResult with the generated patch
        """
        self.logger.info(f"Processing {sample.instance_id} ({sample.repo})")
        start_time = time.time()
        
        try:
            # Run agent based on config
            if self.config.agent_type == "openhands":
                agent_output, patch = run_openhands_on_sample(
                    sample=sample,
                    model_name=self.config.model,
                    provider=self.config.provider,
                    max_iterations=self.config.max_iterations,
                    max_retries=self.config.max_retries,
                    timeout=self.config.agent_timeout,
                    output_dir=str((self.config.output_dir / "openhands").resolve()),
                    base_url=self.config.base_url,
                )
            elif self.config.agent_type == "react":
                agent_output, patch = run_custom_on_sample(
                    sample=sample,
                    model_name=self.config.model,
                    provider=self.config.provider,
                    max_iterations=self.config.max_iterations,
                    timeout=self.config.agent_timeout,
                    base_url=self.config.base_url,
                )
            else:
                raise ValueError(
                    f"Unknown agent_type: '{self.config.agent_type}'. "
                    f"Supported types: 'react', 'openhands'"
                )
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed {sample.instance_id} in {elapsed:.1f}s, patch length: {len(patch)}")
            
            return SWEBenchResult(
                instance_id=sample.instance_id,
                repo=sample.repo,
                patch=patch,
                agent_output=agent_output,
                response_time_seconds=elapsed,
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Error processing {sample.instance_id}: {e}")
            return SWEBenchResult(
                instance_id=sample.instance_id,
                repo=sample.repo,
                patch="",
                agent_output="",
                response_time_seconds=elapsed,
                error=str(e),
            )
    
    def generate_responses(self, orchestrator: Any = None) -> Dict[str, Any]:
        """
        Run agents on all samples to generate patches.
        
        Note: The orchestrator parameter is ignored - agents run inside containers.
        Supports parallel execution via config.num_workers.
        
        Returns:
            Dictionary with results and predictions file path
        """
        samples = self.load_samples()
        total = len(samples)
        
        if self.config.num_workers > 1:
            # Parallel execution
            results = self._run_parallel(samples)
        else:
            # Sequential execution
            results = []
            for i, sample in enumerate(samples):
                self.logger.info(f"[{i+1}/{total}] {sample.instance_id}")
                result = self.run_agent_on_sample(sample)
                results.append(result)
                self.results.append(result)
        
        # Write predictions to JSONL file (for swebench harness)
        predictions_path = self.config.output_dir / f"{self.config.run_id}_predictions.jsonl"
        self._write_predictions(results, predictions_path)
        
        return {
            "results": results,
            "predictions_path": str(predictions_path),
            "run_id": self.config.run_id,
        }
    
    def _run_parallel(self, samples: List[SWEBenchSample]) -> List[SWEBenchResult]:
        """
        Run agents on samples in parallel using ThreadPoolExecutor.
        
        Args:
            samples: List of samples to process
            
        Returns:
            List of results in the same order as input samples
        """
        total = len(samples)
        completed = 0
        lock = Lock()
        
        # Map to store results by instance_id to preserve order
        results_map: Dict[str, SWEBenchResult] = {}
        
        def process_sample(sample: SWEBenchSample) -> SWEBenchResult:
            nonlocal completed
            result = self.run_agent_on_sample(sample)
            with lock:
                completed += 1
                self.results.append(result)
                self.logger.info(f"[{completed}/{total}] Completed {sample.instance_id}")
            return result
        
        self.logger.info(f"Running {total} samples with {self.config.num_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_sample, sample): sample 
                for sample in samples
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    results_map[sample.instance_id] = result
                except Exception as e:
                    self.logger.error(f"Error processing {sample.instance_id}: {e}")
                    # Create error result
                    results_map[sample.instance_id] = SWEBenchResult(
                        instance_id=sample.instance_id,
                        repo=sample.repo,
                        patch="",
                        agent_output="",
                        error=str(e),
                    )
        
        # Return results in original sample order
        return [results_map[s.instance_id] for s in samples]
    
    def _write_predictions(self, results: List[SWEBenchResult], path: Path) -> None:
        """Write results to JSONL file in swebench format."""
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
        Evaluate generated patches using swebench harness.
        
        Args:
            generation_results: Output from generate_responses()
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.config.run_evaluation:
            # Skip evaluation, just return generation stats
            results = generation_results["results"]
            return self._compute_generation_stats(results)
        
        predictions_path = generation_results["predictions_path"]
        run_id = generation_results["run_id"]
        
        self.logger.info("Running swebench evaluation harness...")
        
        try:
            from swebench.harness.run_evaluation import main as run_evaluation
            import swebench.harness.constants
            
            # Redirect swebench logs to our output directory
            swebench.harness.constants.RUN_EVALUATION_LOG_DIR = self.config.output_dir / "logs" / "run_evaluation"
            
            # Map dataset variant to swebench dataset name
            dataset_name = {
                "verified": "princeton-nlp/SWE-bench_Verified",
                "verified_mini": "princeton-nlp/SWE-bench_Verified",  # Use same for eval
            }.get(self.config.dataset, "princeton-nlp/SWE-bench_Verified")
            
            # Get instance IDs from our predictions
            instance_ids = [r.instance_id for r in generation_results["results"]]
            
            # Run evaluation - put logs in output directory
            report_dir = str(self.config.output_dir)
            run_evaluation(
                dataset_name=dataset_name,
                split=self.config.split,
                instance_ids=instance_ids,
                predictions_path=predictions_path,
                max_workers=self.config.eval_workers,
                force_rebuild=False,
                cache_level="env",
                clean=True,
                open_file_limit=4096,
                run_id=run_id,
                timeout=self.config.agent_timeout,
                namespace="swebench",
                rewrite_reports=False,
                modal=False,
                report_dir=report_dir,
            )
            
            # Move detailed logs from cwd to output directory
            cwd_logs = Path("logs/run_evaluation") / run_id
            if cwd_logs.exists():
                dest_logs = self.config.output_dir / "logs" / run_id
                dest_logs.parent.mkdir(parents=True, exist_ok=True)
                if dest_logs.exists():
                    shutil.rmtree(dest_logs)
                shutil.move(str(cwd_logs), str(dest_logs))
                self.logger.info(f"Moved evaluation logs to {dest_logs}")
                # Clean up empty parent directories
                try:
                    Path("logs/run_evaluation").rmdir()
                    Path("logs").rmdir()
                except OSError:
                    pass  # Not empty, that's fine
            
            # Parse evaluation results
            return self._parse_evaluation_results(run_id, generation_results["results"], report_dir)
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            # Return generation stats only
            return self._compute_generation_stats(generation_results["results"])
    
    def _parse_agent_output(self, agent_output: str) -> Any:
        """Parse agent_output JSON string into an object for cleaner output."""
        if not agent_output:
            return None
        try:
            return json.loads(agent_output)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, return as-is
            return agent_output
    
    def _compute_generation_stats(self, results: List[SWEBenchResult]) -> Dict[str, float]:
        """Compute basic statistics from generation results."""
        total = len(results)
        errors = sum(1 for r in results if r.error)
        with_patch = sum(1 for r in results if r.patch and not r.error)
        avg_time = sum(r.response_time_seconds for r in results) / total if total > 0 else 0
        
        return {
            "total_samples": float(total),
            "errors": float(errors),
            "patches_generated": float(with_patch),
            "patch_rate": with_patch / total if total > 0 else 0.0,
            "avg_response_time_seconds": avg_time,
        }
    
    def _parse_evaluation_results(
        self,
        run_id: str,
        results: List[SWEBenchResult],
        report_dir: str = ".",
    ) -> Dict[str, float]:
        """Parse evaluation results from swebench harness output."""
        resolved_count = 0
        total = len(results)
        report_found = False
        
        # swebench harness writes reports in multiple possible locations/formats:
        # 1. Current dir: {model_name}.{run_id}.json (e.g., react_gpt-4o.react_gpt-4o_mini_20260120_001229.json)
        # 2. Logs dir: logs/run_evaluation/{run_id}/report.json
        
        # Try to find and parse the report from current directory first
        model_name = f"{self.config.agent_type}_{self.config.model}"
        report_patterns = [
            f"./{model_name}.{run_id}.json",
            f"./*.{run_id}.json",
            f"{report_dir}/logs/run_evaluation/{run_id}/report.json",
        ]
        
        for pattern in report_patterns:
            matching_files = glob.glob(pattern)
            for report_file in matching_files:
                report_path = Path(report_file)
                if report_path.exists():
                    try:
                        with open(report_path) as f:
                            report = json.load(f)
                        
                        # Update results with resolution status
                        # Note: swebench uses "resolved_ids" in schema v2, "resolved" in older versions
                        resolved_ids = set(report.get("resolved_ids", report.get("resolved", [])))
                        for result in results:
                            if result.instance_id in resolved_ids:
                                result.resolved = True
                                resolved_count += 1
                        
                        report_found = True
                        self.logger.info(f"Parsed evaluation report from {report_path}")
                        
                        # Move report to output dir
                        dest = self.config.output_dir / report_path.name
                        if report_path != dest:
                            shutil.move(str(report_path), dest)
                            self.logger.info(f"Moved report to {dest}")
                        
                        break
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Could not parse report {report_path}: {e}")
            
            if report_found:
                break
        
        if not report_found:
            self.logger.warning(f"Could not find evaluation report for run_id={run_id}")
        
        # Compute metrics
        base_stats = self._compute_generation_stats(results)
        base_stats.update({
            "resolved": float(resolved_count),
            "resolve_rate": resolved_count / total if total > 0 else 0.0,
        })
        
        return base_stats
    
    def run_benchmark(self, orchestrator: Any = None) -> Dict[str, float]:
        """
        Run the full SWE-bench benchmark.
        
        Note: The orchestrator parameter is ignored - agents run inside containers
        with their own model configuration.
        
        Returns:
            Dictionary with benchmark metrics
        """
        self.logger.info(f"Running SWE-bench benchmark: {self.config.dataset}")
        self.logger.info(f"Agent: {self.config.agent_type}, Model: {self.config.model}")
        
        generation_results = self.generate_responses(orchestrator)
        evaluation_results = self.evaluate_responses(generation_results)
        
        # Save final results
        results_path = self.config.output_dir / f"{self.config.run_id}_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": {
                    "dataset": self.config.dataset,
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
                        "agent_output": self._parse_agent_output(r.agent_output),
                    }
                    for r in self.results
                ],
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        return evaluation_results


@register_benchmark("swebench")
def _create_swebench_benchmark(
    options: Optional[Dict[str, Any]] = None,
) -> SWEBenchBenchmark:
    """
    Create a SWE-bench benchmark instance.

    Options:
        Dataset options:
        - dataset: Dataset variant ("verified_mini" or "verified", default: "verified_mini")
        - split: Dataset split (default: "test")
        - shuffle: Whether to shuffle (default: False)
        - seed: Random seed (default: 42)
        - limit: Max samples to evaluate (default: None = all)
        - repos: List of repos to filter (default: None = all)
        - instance_ids: List of instance IDs to filter (default: None = all)

        Agent options:
        - agent_type: Agent type ("react" or "openhands", default: "react")
        - model: Model name (default: "gpt-4o")
        - provider: Model provider (default: "openai")
        - max_iterations: Max agent iterations (default: 10)
        - max_retries: Max retries on exceptions, OpenHands only (default: 0)

        Execution options:
        - agent_timeout: Timeout per task in seconds (default: 1800)
        - num_workers: Parallel workers (default: 1)
        - run_evaluation: Whether to run swebench harness (default: True)
        - eval_workers: Parallel workers for evaluation (default: 4)

        Output options:
        - output_dir: Output directory path (default: ./outputs)
        - run_id: Run identifier (default: auto-generated)
    """
    options = options or {}

    config_fields = {
        "dataset", "split", "shuffle", "seed", "limit", "repos", "instance_ids",
        "agent_type", "model", "provider", "base_url", "max_iterations", "max_retries",
        "agent_timeout", "num_workers", "run_evaluation", "eval_workers",
        "output_dir", "run_id",
    }
    config_options = {k: v for k, v in options.items() if k in config_fields}

    if "output_dir" in config_options and config_options["output_dir"]:
        config_options["output_dir"] = Path(config_options["output_dir"])

    config = SWEBenchConfig(**config_options)
    return SWEBenchBenchmark(config)

