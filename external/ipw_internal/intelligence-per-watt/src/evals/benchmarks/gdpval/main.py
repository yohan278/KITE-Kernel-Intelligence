# src/gdpval/benchmark.py
from __future__ import annotations

from typing import Any, Dict, Optional

from evals.base import BaseBenchmark
from evals.registry import register_benchmark
from evals.benchmarks.gdpval.dataset import load_gdpval_samples
from evals.benchmarks.gdpval.sandbox import DockerSandbox, SandboxHandle
from evals.benchmarks.gdpval.util import GDPvalResult, prepare_hf_folder, upload_folder_to_hf


def create_sandbox_tools(env: SandboxHandle):
    """Create callable tool functions for the sandbox environment."""
    
    def bash(command: str) -> Dict[str, Any]:
        """Execute shell commands inside the GDPval sandbox."""
        return env.run_tool("bash", command)
    
    def python(code: str) -> Dict[str, Any]:
        """Execute Python code inside the GDPval sandbox."""
        return env.run_tool("python", code)
    
    return [bash, python]


class GDPvalBenchmark(BaseBenchmark):
    def __init__(
        self,
        *,
        sandbox: Optional[DockerSandbox] = None,
        shuffle: bool = False,
        upload_to_hf: bool = False,
        limit: Optional[int] = None,
        logger=None,
    ):
        super().__init__(logger=logger, system_instruction=None)
        self.sandbox = sandbox or DockerSandbox()
        self.shuffle = shuffle
        self.upload_to_hf = upload_to_hf
        self.limit = limit

    def generate_responses(self, orchestrator_factory: Any) -> Dict[str, GDPvalResult]:
        """Run the orchestrator on GDPval samples.
        
        Args:
            orchestrator_factory: A callable that takes a list of tools and returns an
                                 orchestrator instance with a run() method.
        """
        results: Dict[str, GDPvalResult] = {}

        for sample in load_gdpval_samples(shuffle=self.shuffle, limit=self.limit):
            with self.sandbox.run(sample.files) as env:
                # Create tools that run in this sandbox
                tools = create_sandbox_tools(env)
                
                # Create orchestrator instance with tools for this sample
                orchestrator = orchestrator_factory(tools)
                
                # Run the orchestrator - it handles all tool calling internally
                response = orchestrator.run(sample.prompt)

                # Extract final response text
                final_text = str(response) if response else ""
                
                files = env.collect_deliverables()

            results[sample.task_id] = GDPvalResult(
                task_id=sample.task_id,
                deliverable_text=final_text,
                deliverable_files=files,
            )

        return results

    def evaluate_responses(
        self,
        results: Dict[str, GDPvalResult],
    ) -> Dict[str, float]:
        folder = prepare_hf_folder(list(results.values()))
        self.logger.info(f"GDPval deliverables ready in {folder}")
        if self.upload_to_hf:
            dataset_url = upload_folder_to_hf(folder)
            self.logger.info(f"Uploaded GDPval dataset to {dataset_url}")
        return {"samples_completed": float(len(results))}


@register_benchmark("gdpval")
def _create_gdpval_benchmark(
    options: Optional[Dict[str, Any]] | None = None,
) -> GDPvalBenchmark:
    options = options or {}
    return GDPvalBenchmark(**options)