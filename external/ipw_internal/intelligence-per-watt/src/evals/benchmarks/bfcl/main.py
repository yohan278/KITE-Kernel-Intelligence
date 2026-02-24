# benchmarks/bfcl/main.py
"""
BFCL (Berkeley Function Calling Leaderboard) benchmark.

Runs bfcl generate + evaluate against an OpenAI-compatible server.
Install: pip install bfcl-eval
"""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

from evals.base import CLIBenchmark
from evals.registry import register_benchmark


class BFCLBenchmark(CLIBenchmark):
    """BFCL benchmark for testing function calling capabilities."""
    
    def __init__(
        self,
        *,
        model: str = "agent",
        test_category: str = "all",
        base_url: str = "http://localhost:8000/v1",
        timeout: Optional[int] = None,
        logger=None,
    ):
        super().__init__(logger=logger, system_instruction=None)
        self.model = model
        self.test_category = test_category
        self.base_url = base_url
        self.timeout = timeout

    def _run_cmd(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a subprocess command."""
        self.logger.info(f"Running: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )

    def run_benchmark(self, orchestrator: Any) -> Dict[str, float]:
        """Run BFCL generate + evaluate pipeline."""
        
        # Step 1: Generate responses
        generate_cmd = [
            "bfcl", "generate",
            "--model", self.model,
            "--test-category", self.test_category,
            "--base-url", self.base_url,
        ]
        
        try:
            gen_result = self._run_cmd(generate_cmd)
            if gen_result.returncode != 0:
                self.logger.error(f"bfcl generate failed: {gen_result.stderr}")
                return {"success": 0.0, "generate_exit_code": float(gen_result.returncode)}
            
            # Step 2: Evaluate responses
            evaluate_cmd = [
                "bfcl", "evaluate",
                "--model", self.model,
                "--test-category", self.test_category,
            ]
            
            eval_result = self._run_cmd(evaluate_cmd)
            if eval_result.returncode != 0:
                self.logger.error(f"bfcl evaluate failed: {eval_result.stderr}")
                return {"success": 0.0, "evaluate_exit_code": float(eval_result.returncode)}
            
            return {
                "success": 1.0,
                "generate_exit_code": float(gen_result.returncode),
                "evaluate_exit_code": float(eval_result.returncode),
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"BFCL timed out after {self.timeout}s")
            return {"success": 0.0, "timeout": 1.0}
        except FileNotFoundError:
            self.logger.error("bfcl not found. Install with: pip install bfcl-eval")
            return {"success": 0.0, "error": 1.0}
        except Exception as e:
            self.logger.error(f"BFCL error: {e}")
            return {"success": 0.0, "error": 1.0}


@register_benchmark("bfcl")
def _create_bfcl_benchmark(
    options: Optional[Dict[str, Any]] | None = None,
) -> BFCLBenchmark:
    """Create BFCL benchmark.
    
    Options:
        - model: Model name (default: "agent")
        - test_category: BFCL test category (default: "all")
        - base_url: OpenAI-compatible API base URL (default: "http://localhost:8000/v1")
        - timeout: Command timeout in seconds
    """
    options = options or {}
    return BFCLBenchmark(**options)

