"""
OpenHands runner for SWE-bench evaluation.

This module provides a wrapper around the openhands-benchmarks CLI tool
(swebench-infer) to run OpenHands on SWE-bench instances.

The openhands-benchmarks repo is a git submodule at swebench/openhands-benchmarks
and uses its own Docker images and evaluation pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv

if TYPE_CHECKING:
    from .dataset import SWEBenchSample

logger = logging.getLogger(__name__)

# Load environment variables from evals/.env
_EVALS_DIR = Path(__file__).parent.parent.parent.parent
_ENV_FILE = _EVALS_DIR / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
    logger.debug(f"Loaded environment from {_ENV_FILE}")

# Path to the cloned openhands-benchmarks repo (now in same directory as this file)
OPENHANDS_BENCHMARKS_DIR = Path(__file__).parent / "openhands-benchmarks"


def create_llm_config(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Path:
    """
    Create a temporary LLM config file for OpenHands.
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "anthropic/claude-sonnet-4-20250514")
        api_key: Optional API key (defaults to env var based on model)
        base_url: Optional base URL for the API
        
    Returns:
        Path to the created config file
    """
    config = {"model": model}
    
    # Set API key from environment if not provided
    if api_key is None:
        if model.startswith("anthropic/") or "claude" in model.lower():
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        config["api_key"] = api_key
    
    if base_url:
        config["base_url"] = base_url
        if not api_key:
            config["api_key"] = "dummy"
    
    # Write config to a temp file in the benchmarks dir
    config_dir = OPENHANDS_BENCHMARKS_DIR / ".llm_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "eval_config.json"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created LLM config at {config_path}")
    return config_path


def run_openhands_on_sample(
    sample: "SWEBenchSample",
    model_name: str = "gpt-4o",
    provider: str = "openai",
    max_iterations: int = 10,
    max_retries: int = 0,
    timeout: int = 3600,
    output_dir: Optional[str] = None,
    dataset: str = "MariusHobbhahn/swe-bench-verified-mini",
    base_url: Optional[str] = None,
) -> tuple[str, str]:
    """
    Run OpenHands agent on a single SWE-bench sample.

    This is a convenience wrapper that runs the full openhands-benchmarks
    pipeline for a single instance.

    Args:
        sample: SWEBenchSample to process
        model_name: Model identifier
        provider: Model provider (openai, anthropic)
        max_iterations: Maximum agent iterations
        max_retries: Maximum retries on exceptions (default: 1 = no retries)
        timeout: Overall timeout in seconds for subprocess
        output_dir: Directory to save OpenHands outputs
        dataset: HuggingFace dataset name
        base_url: Optional base URL for OpenAI-compatible APIs (e.g. vLLM)

    Returns:
        Tuple of (agent_output, patch)
    """
    logger.info(f"Running OpenHands on {sample.instance_id}")
    logger.info(f"Model: {provider}/{model_name}")
    
    # Format model name for OpenHands
    if provider == "anthropic":
        model = f"anthropic/{model_name}"
    else:
        model = model_name
    
    # Run OpenHands via CLI
    result = run_openhands(
        instance_ids=[sample.instance_id],
        model=model,
        max_iterations=max_iterations,
        max_retries=max_retries,
        dataset=dataset,
        output_dir=output_dir,
        timeout=timeout,
        base_url=base_url,
    )
    
    # Extract output and patch from result
    if result["success"]:
        output_dir = Path(result["output_dir"])
        # Look for output JSONL - search recursively since the actual files are in subdirs
        output_files = list(output_dir.rglob("output.jsonl"))
        if not output_files:
            # Fallback to any jsonl file
            output_files = list(output_dir.rglob("*.jsonl"))
        if output_files:
            # Use the most recent file
            output_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            with open(output_files[0]) as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("instance_id") == sample.instance_id:
                        patch = data.get("test_result", {}).get("git_patch", "")
                        return json.dumps(data.get("history", [])), patch
        return "Evaluation completed but no output found", ""
    else:
        return f"Error: {result.get('error', 'Unknown error')}", ""


def run_openhands(
    instance_ids: list[str],
    model: str = "gpt-4o",
    max_iterations: int = 10,
    max_retries: int = 0,
    dataset: str = "MariusHobbhahn/swe-bench-verified-mini",
    output_dir: Optional[str] = None,
    num_workers: int = 1,
    workspace_type: str = "docker",
    timeout: int = 3600,
    base_url: Optional[str] = None,
) -> dict:
    """
    Run OpenHands SWE-bench evaluation via CLI.
    
    Args:
        instance_ids: List of SWE-bench instance IDs to evaluate
        model: Model identifier for LiteLLM
        max_iterations: Maximum agent iterations per instance
        max_retries: Maximum retries on exceptions (default: 1 = no retries)
        dataset: HuggingFace dataset name
        output_dir: Optional output directory (auto-generated if not provided)
        num_workers: Number of parallel workers
        workspace_type: Workspace type (docker or remote)
        timeout: Overall timeout in seconds for subprocess (default: 1 hour)
        
    Returns:
        Dict with success status, output_dir, and any error message
    """
    if not OPENHANDS_BENCHMARKS_DIR.exists():
        return {
            "success": False,
            "error": f"openhands-benchmarks not found at {OPENHANDS_BENCHMARKS_DIR}. "
                     "Run: git clone https://github.com/OpenHands/benchmarks.git vendor/openhands-benchmarks",
        }
    
    # Clear existing outputs to force fresh runs
    if output_dir:
        # Use specified output dir
        output_path = Path(output_dir)
        if output_path.exists():
            logger.info(f"Clearing existing output at {output_path}")
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # Default: clear eval_outputs in openhands-benchmarks
        eval_outputs_dir = OPENHANDS_BENCHMARKS_DIR / "eval_outputs"
        if eval_outputs_dir.exists():
            logger.info(f"Clearing existing eval_outputs at {eval_outputs_dir}")
            shutil.rmtree(eval_outputs_dir)
    
    # Create LLM config
    config_path = create_llm_config(model, base_url=base_url)
    
    # Create instance selection file
    select_file = OPENHANDS_BENCHMARKS_DIR / ".instance_selection.txt"
    with open(select_file, "w") as f:
        f.write("\n".join(instance_ids))
    
    # Build command
    # Note: llm_config_path is a positional argument, not a named argument
    cmd = [
        "uv", "run", "swebench-infer",
        str(config_path),  # positional argument
        "--dataset", dataset,
        "--select", str(select_file),
        "--max-iterations", str(max_iterations),
        "--max-retries", str(max_retries),
        "--num-workers", str(num_workers),
        "--workspace", workspace_type,
    ]
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    # Set environment variables
    env = os.environ.copy()
    env["SKIP_BUILD"] = "1"  # Skip building agent server images
    
    logger.info(f"Running OpenHands CLI: {' '.join(cmd)}")
    logger.info(f"Working directory: {OPENHANDS_BENCHMARKS_DIR}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=OPENHANDS_BENCHMARKS_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        # Log output regardless of return code for debugging
        if result.stdout:
            logger.info(f"OpenHands CLI stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"OpenHands CLI stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            logger.error(f"OpenHands CLI failed with return code {result.returncode}")
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout,
            }
        
        # Find output directory from stdout
        output_dir_actual = None
        for line in result.stdout.split("\n"):
            if "output" in line.lower() and "/" in line:
                # Try to extract path
                parts = line.split()
                for part in parts:
                    if "/" in part and Path(part).exists():
                        output_dir_actual = part
                        break
        
        return {
            "success": True,
            "output_dir": output_dir_actual or output_dir or str(OPENHANDS_BENCHMARKS_DIR / "eval_outputs"),
            "stdout": result.stdout,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Evaluation timed out after {timeout} seconds",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        # Clean up temp files
        if select_file.exists():
            select_file.unlink()
        if config_path.exists():
            config_path.unlink()
