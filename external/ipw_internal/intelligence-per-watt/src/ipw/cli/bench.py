"""Run agent benchmarks with energy telemetry for Intelligence Per Watt measurement.

To extend:
  - Add client: MODEL_FACTORIES
  - Add dataset: DATASET_CONFIG
  - Add agent: AGENT_FACTORIES
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
import platform
import shlex
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import click

logger = logging.getLogger(__name__)

from ipw.cli._console import error, info, success, warning
from ipw.cli.server_manager import (
    InferenceServerManager,
    ServerConfig,
    build_server_configs,
)
from ipw.core.types import TelemetryReading
from ipw.execution.telemetry_session import TelemetrySample, TelemetrySession
from ipw.telemetry import EnergyMonitorCollector


# Default output directory for traces
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "outputs"


# Client Configuration
# To add a new client: add a factory function and register it in MODEL_FACTORIES

def _create_ollama_model(model: str, base_url: str | None = None):
    from agno.models.ollama import Ollama
    return Ollama(id=model, host=base_url or "http://localhost:11434")


def _create_vllm_model(model: str, base_url: str | None = None):
    from agno.models.openai import OpenAIChat
    return OpenAIChat(
        id=model,
        base_url=base_url or "http://localhost:8000/v1",
        collect_metrics_on_completion=True,  # Enable token metrics collection
    )


def _create_openai_model(model: str, base_url: str | None = None):
    from agno.models.openai import OpenAIChat
    if base_url:
        return OpenAIChat(id=model, base_url=base_url, collect_metrics_on_completion=True)
    return OpenAIChat(id=model, collect_metrics_on_completion=True)


MODEL_FACTORIES: Dict[str, Callable] = {
    "ollama": _create_ollama_model,
    "vllm": _create_vllm_model,
    "openai": _create_openai_model,
}

# Model Presets - convenient aliases for common configurations
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B-Instruct",
        "default_url": "http://localhost:8000/v1",
        "vllm_args": {"tensor_parallel_size": 1, "max_model_len": 32768},
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "default_url": "http://localhost:8000/v1",
        "vllm_args": {"tensor_parallel_size": 1, "max_model_len": 32768},
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
        "default_url": "http://localhost:8000/v1",
        "vllm_args": {"tensor_parallel_size": 4, "max_model_len": 32768},
    },
}


def resolve_model(model: str, vllm_url: str | None = None) -> tuple[str, str]:
    """Resolve model preset or raw name to model config.

    Args:
        model: Model preset name (qwen3-4b, gpt-oss-120b) or raw model name
        vllm_url: Optional vLLM server URL override

    Returns:
        Tuple of (model_id, base_url)

    Raises:
        ValueError: If raw model name used without --vllm-url
    """
    if model in MODEL_PRESETS:
        preset = MODEL_PRESETS[model]
        return preset["model_id"], vllm_url or preset["default_url"]
    else:
        # Raw model name - require vllm_url
        if not vllm_url:
            raise ValueError(
                f"--vllm-url required for custom model: {model}. "
                f"Use a preset ({', '.join(MODEL_PRESETS.keys())}) or provide --vllm-url."
            )
        return model, vllm_url


def get_model_alias(model_id: str) -> str:
    """Get a clean alias from a model ID.

    Args:
        model_id: Full model ID (e.g., "Qwen/Qwen3-4B-Instruct")

    Returns:
        Clean alias (e.g., "qwen3-4b-instruct")
    """
    # Check if it's a preset name
    if model_id in MODEL_PRESETS:
        return model_id

    # Extract the model name part after the organization
    if "/" in model_id:
        model_id = model_id.split("/")[-1]

    # Clean up the name
    return model_id.lower().replace("_", "-")


def _detect_orchestrator_submodels(available_tools: List[str]) -> List[str]:
    """Extract vLLM/Ollama models from orchestrator's available_tools.

    Orchestrator agents may have access to MCP tools that are actually
    inference endpoints (vLLM or Ollama). This function detects those
    tools and returns submodel specs for auto-management.

    Args:
        available_tools: List of tool names from orchestrator config

    Returns:
        List of submodel specs in format "alias:backend:model_id"
    """
    # Known vLLM model mappings (alias -> HuggingFace model ID)
    VLLM_MODEL_MAPPINGS = {
        "qwen3-32b": "Qwen/Qwen3-32B",
        "qwen3-4b": "Qwen/Qwen3-4B-Instruct",
        "qwen2.5-math-72b": "Qwen/Qwen2.5-Math-72B-Instruct",
        "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    }

    submodels = []
    for tool in available_tools:
        if tool.startswith("vllm:"):
            # Extract model alias (e.g., "vllm:qwen3-32b" -> "qwen3-32b")
            model_alias = tool.split(":", 1)[1]
            model_id = VLLM_MODEL_MAPPINGS.get(model_alias, model_alias)
            submodels.append(f"{model_alias}:vllm:{model_id}")
        elif tool.startswith("ollama:"):
            # Ollama model names are used directly
            model_name = tool.split(":", 1)[1]
            submodels.append(f"{model_name}:ollama:{model_name}")

    return submodels


def _get_orchestrator_available_tools(checkpoint_path: str) -> List[str]:
    """Load available_tools from orchestrator checkpoint config.

    Args:
        checkpoint_path: Path to orchestrator checkpoint

    Returns:
        List of tool names, or empty list if config not found
    """
    config_path = Path(checkpoint_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("available_tools", [])
    return []


# Dataset Configuration
# To add a new dataset: add tools factory, instructions, and register in DATASET_CONFIG

def _get_gaia_tools() -> list:
    from evals.benchmarks.gaia import create_gaia_tools
    return create_gaia_tools()


def _get_hle_tools() -> list:
    """HLE benchmark doesn't require special tools - reasoning focused."""
    return []


def _get_apex_tools() -> list:
    """APEX benchmark doesn't require special tools - professional domain focused."""
    return []


GAIA_INSTRUCTIONS = (
    "You are a helpful AI assistant that can read and analyze various file types. "
    "When a file path is provided in the question, use the appropriate file reading tool to access it. "
    "Answer questions accurately and concisely. Follow the instructions in the prompt carefully."
)

HLE_INSTRUCTIONS = (
    "You are a helpful assistant solving challenging questions from Humanity's Last Exam. "
    "These questions span academic domains including math, physics, chemistry, biology, "
    "computer science, and humanities. Think step by step and provide your best answer. "
    "For multiple choice questions, respond with just the letter (A, B, C, D, etc.)."
)

APEX_INSTRUCTIONS = (
    "You are a professional AI assistant helping with economically valuable tasks. "
    "You have expertise across Investment Banking, Management Consulting, Law, and Medicine. "
    "Provide detailed, accurate, and professional responses. Follow the task requirements carefully."
)

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "gaia": {
        "tools_factory": _get_gaia_tools,
        "instructions": GAIA_INSTRUCTIONS,
    },
    "hle": {
        "tools_factory": _get_hle_tools,
        "instructions": HLE_INSTRUCTIONS,
    },
    "apex": {
        "tools_factory": _get_apex_tools,
        "instructions": APEX_INSTRUCTIONS,
    },
}


# Agent Configuration
# To add a new agent: add a factory function and register it in AGENT_FACTORIES

def _create_react_agent(
    model,
    tools: list,
    instructions: str | None = None,
    event_recorder: Optional[Any] = None,
):
    from agents import React
    return React(
        model=model,
        tools=tools,
        instructions=instructions,
        event_recorder=event_recorder,
    )


def _create_openhands_agent(
    model,
    tools: list,
    instructions: str | None = None,
    event_recorder: Optional[Any] = None,
):
    from agents import OpenHands
    from openhands.sdk import LLM as OpenHandsLLM

    # OpenHands expects its own LLM type, not Agno models
    # Extract model ID and base_url from the Agno model
    model_id = getattr(model, 'id', str(model))
    base_url = getattr(model, 'base_url', None)

    # Add 'hosted_vllm/' prefix for litellm to recognize vLLM servers
    # See: https://docs.litellm.ai/docs/providers/vllm
    if base_url and not model_id.startswith(('openai/', 'anthropic/', 'azure/', 'hosted_vllm/')):
        model_id = f'hosted_vllm/{model_id}'

    openhands_llm = OpenHandsLLM(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    )

    return OpenHands(
        model=openhands_llm,
        tools=tools,
        event_recorder=event_recorder,
    )


def _create_terminus_agent(
    model,
    tools: list,
    instructions: str | None = None,
    event_recorder: Optional[Any] = None,
):
    from agents import Terminus
    return Terminus(
        model=model,
        event_recorder=event_recorder,
    )


def _create_orchestrator_agent(
    model,  # Ignored - Orchestrator uses internal policy
    tools: list,  # Ignored - Orchestrator manages its own MCP tools
    instructions: str | None = None,  # Ignored
    event_recorder: Optional[Any] = None,
    checkpoint_path: str | None = None,
):
    """Create Orchestrator agent with trained policy checkpoint.

    Unlike other agents, Orchestrator doesn't use an external LLM model or tools.
    It uses a trained policy checkpoint to route tasks to its internal MCP tools.
    """
    from agents import Orchestrator
    return Orchestrator(
        checkpoint_path=checkpoint_path,
        event_recorder=event_recorder,
    )


AGENT_FACTORIES: Dict[str, Callable] = {
    "react": _create_react_agent,
    "openhands": _create_openhands_agent,
    "terminus": _create_terminus_agent,
    "orchestrator": _create_orchestrator_agent,
}


# Factory Functions

def create_model(client_id: str, model: str, base_url: str | None = None):
    """Create an Agno model instance based on client type."""
    if client_id not in MODEL_FACTORIES:
        raise ValueError(f"Unknown client: {client_id}. Supported: {list(MODEL_FACTORIES.keys())}")
    return MODEL_FACTORIES[client_id](model, base_url)


def get_tools_for_benchmark(dataset_id: str) -> list:
    """Get the tools required for a benchmark dataset."""
    config = DATASET_CONFIG.get(dataset_id)
    if config and "tools_factory" in config:
        return config["tools_factory"]()
    return []


def get_agent_instructions(dataset_id: str) -> str | None:
    """Get dataset-specific agent instructions."""
    config = DATASET_CONFIG.get(dataset_id)
    if config:
        return config.get("instructions")
    return None


def create_agent(
    agent_id: str,
    model,
    tools: list,
    instructions: str | None = None,
    event_recorder: Optional[Any] = None,
    checkpoint_path: str | None = None,
):
    """Create an agent/orchestrator instance."""
    if agent_id not in AGENT_FACTORIES:
        raise ValueError(f"Unknown agent: {agent_id}. Supported: {list(AGENT_FACTORIES.keys())}")
    if agent_id == "orchestrator":
        return AGENT_FACTORIES[agent_id](
            model, tools, instructions, event_recorder, checkpoint_path=checkpoint_path
        )
    return AGENT_FACTORIES[agent_id](model, tools, instructions, event_recorder)


# Energy Telemetry Helpers

def _compute_energy_delta(values: List[Optional[float]]) -> Optional[float]:
    """Compute energy delta from cumulative values.

    Energy counters are cumulative, so we compute delta = end - start.
    Returns None if insufficient valid samples or negative delta (counter reset).
    """
    filtered = [v for v in values if v is not None and math.isfinite(v) and v >= 0]
    if len(filtered) < 2:
        return None
    delta = filtered[-1] - filtered[0]
    return delta if delta >= 0 else None


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Compute mean, returning None if no valid values."""
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(filtered) if filtered else None


def _safe_max(values: List[Optional[float]]) -> Optional[float]:
    """Compute max, returning None if no valid values."""
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    return max(filtered) if filtered else None


def _extract_hardware_info(samples: Sequence[TelemetrySample]) -> Dict[str, Any]:
    """Extract hardware configuration from telemetry samples.

    Args:
        samples: Telemetry samples from the benchmark run

    Returns:
        Dictionary with hardware info (gpu_count, cpu_count, hardware_stack)
    """
    hardware_info: Dict[str, Any] = {
        "gpu_count": None,
        "cpu_count": None,
        "hardware_stack": None,
    }

    if not samples:
        return hardware_info

    # Get info from first sample
    first_reading = samples[0].reading

    # Hardware stack from platform field
    if first_reading.platform:
        hardware_info["hardware_stack"] = first_reading.platform

    # CPU count from system_info
    if first_reading.system_info and first_reading.system_info.cpu_count:
        hardware_info["cpu_count"] = first_reading.system_info.cpu_count

    # GPU count from CUDA_VISIBLE_DEVICES or default to 1 if we have GPU telemetry
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Count GPUs from comma-separated list
        gpu_ids = [g.strip() for g in cuda_visible.split(",") if g.strip()]
        hardware_info["gpu_count"] = len(gpu_ids)
    elif first_reading.gpu_info and first_reading.gpu_info.name:
        # We have GPU info, default to 1 if not specified
        hardware_info["gpu_count"] = 1

    return hardware_info


def _compute_energy_metrics(
    samples: Sequence[TelemetrySample],
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """Compute energy metrics from telemetry samples.

    Args:
        samples: Telemetry samples from the benchmark run
        start_time: Unix timestamp when benchmark started
        end_time: Unix timestamp when benchmark ended

    Returns:
        Dictionary with energy metrics (joules, watts, duration)
    """
    readings = [s.reading for s in samples]

    # Compute energy deltas 
    gpu_energy = _compute_energy_delta([r.energy_joules for r in readings])
    cpu_energy = _compute_energy_delta([r.cpu_energy_joules for r in readings])

    # Compute average power from instantaneous samples
    gpu_power_samples = [r.power_watts for r in readings if r.power_watts is not None]
    cpu_power_samples = [r.cpu_power_watts for r in readings if r.cpu_power_watts is not None]

    duration = max(end_time - start_time, 0.0)
    total_energy = (gpu_energy or 0) + (cpu_energy or 0)

    return {
        "duration_seconds": duration,
        "gpu_energy_joules": gpu_energy,
        "cpu_energy_joules": cpu_energy,
        "total_energy_joules": total_energy if total_energy > 0 else None,
        "avg_gpu_power_watts": _safe_mean(gpu_power_samples),
        "max_gpu_power_watts": _safe_max(gpu_power_samples),
        "avg_cpu_power_watts": _safe_mean(cpu_power_samples),
        "telemetry_samples": len(samples),
    }

# Server Warmup Helpers

def _wait_for_server_ready(client_id: str, base_url: str | None = None, timeout: float = 60.0) -> bool:
    """Wait for inference server to be ready.

    Args:
        client_id: Model provider identifier (ollama, vllm, openai)
        base_url: Override default API endpoint (may include /v1 for OpenAI-compatible servers)
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False if timeout reached
    """
    import urllib.request
    import urllib.error

    if client_id == "ollama":
        url = (base_url or "http://localhost:11434").rstrip("/") + "/api/version"
    elif client_id == "vllm":
        base = (base_url or "http://localhost:8000").rstrip("/")
        # Handle URLs that already include /v1 (OpenAI-compatible format)
        if base.endswith("/v1"):
            url = base + "/models"
        else:
            url = base + "/v1/models"
    elif client_id == "openai":
        # OpenAI API is always available
        return True
    else:
        return True

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, Exception):
            time.sleep(0.5)

    return False


def _run_warmup_query(model, warmup_prompt: str = "Hello") -> None:
    """Run a warmup query to initialize model and exclude cold-start costs.

    This runs a simple query that doesn't count towards profiling metrics.
    It ensures the model is loaded into GPU memory and any initialization
    overhead is excluded from energy measurements.

    Args:
        model: Agno model instance
        warmup_prompt: Simple prompt for warmup (default: "Hello")
    """
    try:
        # Run a minimal inference to warm up the model
        response = model.response(warmup_prompt)
        # Consume the response (important for streaming models)
        _ = response.content if hasattr(response, 'content') else str(response)
    except Exception:
        # Warmup failures are non-fatal, proceed with benchmark
        pass


# Benchmark Execution

def _build_run_metadata() -> Dict[str, Any]:
    """Capture CLI invocation details and version information."""
    try:
        ipw_version = importlib_metadata.version("ipw")
    except importlib_metadata.PackageNotFoundError:
        ipw_version = "unknown"

    return {
        "cli_invocation": {
            "argv": list(sys.argv),
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        },
        "versions": {
            "ipw": ipw_version,
            "python": platform.python_version(),
        },
        "timestamp": datetime.now().isoformat(),
    }


def _get_output_path(benchmark: str, model: str, output_dir: str | None = None) -> Path:
    """Generate organized output path for benchmark results.

    Args:
        benchmark: Benchmark identifier (hle, gaia, apex)
        model: Model identifier
        output_dir: Optional override for output directory

    Returns:
        Path to output directory for this benchmark run
    """
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "bench"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace(":", "_")
    return base / f"{benchmark}_{safe_model}_{timestamp}"


def _save_trace(
    result: Dict[str, Any],
    output_dir: Path,
    client_id: str,
    model_name: str,
    dataset_id: str,
) -> Optional[Path]:
    """Save benchmark trace to output directory.

    Args:
        result: Benchmark result dictionary
        output_dir: Directory to save trace
        client_id: Model provider identifier
        model_name: Model identifier
        dataset_id: Benchmark dataset identifier

    Returns:
        Path to saved trace file, or None if save failed
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results file
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Save summary file with key metrics
        summary = {
            "benchmark": dataset_id,
            "model": model_name,
            "client": client_id,
            "timestamp": datetime.now().isoformat(),
            "accuracy": result.get("accuracy"),
            "total_energy_joules": result.get("total_energy_joules"),
            "ipw_score": result.get("ipw_score"),
            "duration_seconds": result.get("duration_seconds"),
            "telemetry_samples": result.get("telemetry_samples"),
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save per-action breakdown if present
        if "action_breakdown" in result:
            per_action_path = output_dir / "per_action.json"
            with open(per_action_path, "w") as f:
                json.dump({
                    "action_breakdown": result["action_breakdown"],
                    "energy_analysis": result.get("energy_analysis", {}),
                }, f, indent=2, default=str)
            info(f"Per-action breakdown saved to: {per_action_path}")

        info(f"Results saved to: {output_dir}")
        return results_path

    except Exception as e:
        warning(f"Failed to save trace: {e}")
        return None


def execute_benchmark(
    client_id: str,
    model_name: str,
    agent_id: str,
    dataset_id: str,
    max_samples: int | None = None,
    client_base_url: str | None = None,
    output_dir: str | None = None,
    enable_telemetry: bool = True,
    telemetry_granularity: str = "benchmark",
    skip_warmup: bool = False,
    checkpoint_path: str | None = None,
    auto_server: bool = False,
    submodels: Sequence[str] | None = None,
    base_port: int = 8000,
    seed: int | None = None,
    resource_config: str | None = None,
) -> Dict[str, Any]:
    """Execute a benchmark run with energy telemetry.

    Args:
        client_id: Model provider identifier (ollama, vllm, openai)
        model_name: Model identifier
        agent_id: Agent type identifier
        dataset_id: Benchmark dataset identifier
        max_samples: Maximum number of samples to evaluate
        client_base_url: Override default API endpoint
        output_dir: Directory to save results
        enable_telemetry: Whether to collect energy telemetry
        telemetry_granularity: Level of telemetry detail:
            - "benchmark": Overall energy for entire benchmark run
            - "per-action": Energy breakdown per agent action (tool calls, LM inference)
        skip_warmup: Skip server warmup phase (for testing)
        checkpoint_path: Path to trained policy checkpoint (required for orchestrator agent)
        auto_server: Whether to auto-start/stop inference servers
        submodels: List of submodel specs (alias:backend:model_id)
        base_port: Base port for vLLM servers when using auto_server
        seed: Random seed for reproducible benchmark sampling
        resource_config: Hardware resource configuration (1gpu_8cpu, 4gpu_32cpu)

    Returns:
        Dictionary with benchmark metrics and energy metrics
    """
    from evals.registry import get_benchmark

    # Validate orchestrator requirements
    if agent_id == "orchestrator" and checkpoint_path is None:
        raise ValueError(
            "Orchestrator agent requires --checkpoint-path to load trained policy. "
            "Example: ipw bench --agent orchestrator --checkpoint-path /path/to/checkpoint --benchmark hle"
        )

    server_manager: Optional[InferenceServerManager] = None
    managed_urls: Dict[str, str] = {}

    if auto_server:
        # Build list of all submodels to manage
        all_submodels = list(submodels or [])

        # For orchestrator: auto-detect submodels from available_tools
        if agent_id == "orchestrator" and checkpoint_path:
            orch_tools = _get_orchestrator_available_tools(checkpoint_path)
            if orch_tools:
                detected_submodels = _detect_orchestrator_submodels(orch_tools)
                all_submodels.extend(detected_submodels)
                if detected_submodels:
                    info(f"Auto-detected {len(detected_submodels)} submodels from orchestrator config")

        # Build server configurations
        model_alias = get_model_alias(model_name)
        configs = build_server_configs(
            main_model=model_name,
            main_alias=model_alias,
            submodel_specs=all_submodels,
            base_port=base_port,
            main_backend=client_id,
        )

        # Create server manager
        server_manager = InferenceServerManager(configs)

        # Start all servers BEFORE telemetry
        info("Starting inference servers (excluded from profiling)...")
        managed_urls = server_manager.start_all()

        # Update client_base_url to use managed server
        if model_alias in managed_urls:
            client_base_url = managed_urls[model_alias]
            info(f"Using managed server at {client_base_url}")

        # Warmup all models (excluded from profiling)
        if not skip_warmup:
            info("Running warmup queries (excluded from profiling)...")
            server_manager.warmup_all()

    try:
        # If not using auto_server, wait for external server
        if not auto_server:
            # Wait for server to be ready (excludes startup time from measurements)
            # Skip for orchestrator since it manages its own models internally
            if agent_id != "orchestrator" and not skip_warmup:
                info("Waiting for inference server...")
                if not _wait_for_server_ready(client_id, client_base_url, timeout=120.0):
                    warning("Server not responding, proceeding anyway...")

        # Orchestrator doesn't use external model - it has internal MCP tools
        model = None if agent_id == "orchestrator" else create_model(client_id, model_name, client_base_url)

        # Run warmup query to exclude cold-start costs from profiling
        # Skip if auto_server already did warmup, or for orchestrator
        if model is not None and not skip_warmup and not auto_server:
            info("Running warmup query (excluded from measurements)...")
            _run_warmup_query(model)

        dataset_params: Dict[str, Any] = {}
        if max_samples is not None:
            dataset_params["limit"] = max_samples
        # Note: seed is set globally via random.seed() at benchmark start,
        # not passed as a constructor argument (benchmarks use random.sample internally)

        benchmark_factory = get_benchmark(dataset_id)
        benchmark = benchmark_factory(dataset_params)

        tools = get_tools_for_benchmark(dataset_id)
        instructions = get_agent_instructions(dataset_id)

        # Create event recorder for per-action telemetry
        event_recorder = None
        if enable_telemetry and telemetry_granularity == "per-action":
            from ipw.telemetry.events import EventRecorder
            event_recorder = EventRecorder()

        orchestrator = create_agent(agent_id, model, tools, instructions, event_recorder, checkpoint_path)

        # Determine output directory using organized path structure
        actual_output_dir = _get_output_path(dataset_id, model_name, output_dir)

        # === TELEMETRY STARTS HERE ===
        result = _execute_with_telemetry(
            benchmark,
            orchestrator,
            str(actual_output_dir),
            enable_telemetry,
            telemetry_granularity,
            event_recorder,
        )
        # === TELEMETRY ENDS HERE ===

        # Add run metadata
        result["run_metadata"] = {
            **_build_run_metadata(),
            "client_id": client_id,
            "model_name": model_name,
            "agent_id": agent_id,
            "dataset_id": dataset_id,
            "max_samples": max_samples,
            "telemetry_granularity": telemetry_granularity,
            "warmup_excluded": not skip_warmup,
            "auto_server": auto_server,
            "submodels": list(submodels) if submodels else [],
            "managed_server_urls": managed_urls,
            "seed": seed,
            "resource_config": resource_config,
        }

        # Save trace to output directory
        _save_trace(result, actual_output_dir, client_id, model_name, dataset_id)

        return result

    finally:
        # Stop servers AFTER telemetry (excluded from profiling)
        if server_manager:
            info("Stopping inference servers (excluded from profiling)...")
            server_manager.stop_all()


def _execute_with_telemetry(
    benchmark,
    orchestrator,
    output_dir: str | None = None,
    enable_telemetry: bool = True,
    telemetry_granularity: str = "benchmark",
    event_recorder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute benchmark with optional energy telemetry collection.

    Args:
        benchmark: Benchmark instance to run
        orchestrator: Agent orchestrator
        output_dir: Directory to save results (unused currently)
        enable_telemetry: Whether to collect energy telemetry
        telemetry_granularity: Level of telemetry detail ("benchmark" or "per-action")
        event_recorder: EventRecorder instance for per-action tracking

    Returns:
        Dictionary with benchmark metrics and optional energy metrics
    """
    if not enable_telemetry:
        info("Running without energy telemetry (--no-telemetry)")
        start_time = time.time()
        benchmark_metrics = benchmark.run_benchmark(orchestrator)
        end_time = time.time()
        benchmark_metrics["duration_seconds"] = end_time - start_time
        return benchmark_metrics

    try:
        collector = EnergyMonitorCollector()
        # Use 1 hour buffer for benchmarks - they can run for extended periods
        # Also increase max_samples to accommodate long runs at ~20 samples/sec
        with TelemetrySession(
            collector, buffer_seconds=3600.0, max_samples=100_000
        ) as telemetry:
            start_time = time.time()
            benchmark_metrics = benchmark.run_benchmark(orchestrator)
            end_time = time.time()

            samples = list(telemetry.window(start_time, end_time))

        if samples:
            energy_metrics = _compute_energy_metrics(samples, start_time, end_time)

            # Compute IPW score if accuracy available
            accuracy = benchmark_metrics.get("accuracy")
            total_energy = energy_metrics.get("total_energy_joules")
            if accuracy is not None and total_energy and total_energy > 0:
                energy_metrics["ipw_score"] = accuracy / total_energy

            # Extract hardware info from first sample
            hardware_info = _extract_hardware_info(samples)
            energy_metrics.update(hardware_info)

            result = {**benchmark_metrics, **energy_metrics}

            # Add per-action breakdown if requested
            if telemetry_granularity == "per-action" and event_recorder:
                from ipw.telemetry.correlation import (
                    compute_analysis,
                    correlate_energy_to_events,
                )

                events = event_recorder.get_events()
                if events:
                    breakdowns = correlate_energy_to_events(samples, events)
                    analysis = compute_analysis(breakdowns)

                    # Extract turns (LM inference count) and tools used
                    turns = analysis.get("action_counts", {}).get("lm_inference", 0)
                    tool_call_count = analysis.get("action_counts", {}).get("tool_call", 0)

                    # Extract tool names from events
                    tools_used = []
                    for event in events:
                        if event.event_type == "tool_call_start":
                            tool_name = event.metadata.get("tool")
                            if tool_name and tool_name not in tools_used:
                                tools_used.append(tool_name)

                    result["turns"] = turns
                    result["tools_used"] = tools_used
                    result["tools_used_count"] = tool_call_count

                    # Extract tokens from events
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    for event in events:
                        if event.event_type == "lm_inference_end":
                            total_prompt_tokens += event.metadata.get("prompt_tokens", 0)
                            total_completion_tokens += event.metadata.get("completion_tokens", 0)

                    result["total_prompt_tokens"] = total_prompt_tokens
                    result["total_completion_tokens"] = total_completion_tokens
                    result["total_tokens"] = total_prompt_tokens + total_completion_tokens

                    result["action_breakdown"] = [
                        {
                            "action_type": b.action_type,
                            "step_number": b.step_number,
                            "gpu_energy_joules": b.gpu_energy_joules,
                            "cpu_energy_joules": b.cpu_energy_joules,
                            "total_energy_joules": b.total_energy_joules,
                            "duration_ms": b.duration_ms,
                            "max_power_watts": b.max_power_watts,
                            "avg_power_watts": b.avg_power_watts,
                            "memory_bandwidth_gbps": b.memory_bandwidth_gbps,
                            "metadata": b.metadata,
                        }
                        for b in breakdowns
                    ]
                    result["energy_analysis"] = analysis

            return result
        else:
            warning("No telemetry samples collected during benchmark")
            benchmark_metrics["duration_seconds"] = end_time - start_time
            return benchmark_metrics

    except Exception as e:
        warning(f"Telemetry unavailable: {e}. Running without energy measurement.")
        start_time = time.time()
        benchmark_metrics = benchmark.run_benchmark(orchestrator)
        end_time = time.time()
        benchmark_metrics["duration_seconds"] = end_time - start_time
        return benchmark_metrics


# CLI

def _print_list(ctx, param, value) -> None:
    """Print available options and exit."""
    if not value or ctx.resilient_parsing:
        return

    info("Available agents:")
    for agent in AGENT_FACTORIES:
        if agent == "orchestrator":
            info(f"  - {agent} (requires --checkpoint-path)")
        else:
            info(f"  - {agent}")

    info("\nAvailable model presets:")
    for preset, config in MODEL_PRESETS.items():
        info(f"  - {preset} -> {config['model_id']}")
    info("  (or use any HuggingFace model name with --vllm-url)")

    info("\nAvailable benchmarks:")
    for benchmark in DATASET_CONFIG:
        info(f"  - {benchmark}")

    info("\nModel providers:")
    for client in MODEL_FACTORIES:
        info(f"  - {client}")

    ctx.exit()


@click.command(help="Run agent benchmarks with energy telemetry for IPW measurement.")
@click.option(
    "--list", "-l",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_list,
    help="List available agents, models, and benchmarks",
)
@click.option(
    "--agent",
    "agent_id",
    required=True,
    type=click.Choice(list(AGENT_FACTORIES.keys())),
    help="Agent type (react, openhands, terminus, orchestrator)",
)
@click.option(
    "--model",
    required=False,
    help="Model preset (qwen3-4b, gpt-oss-120b) or raw HuggingFace model name (not required for orchestrator)",
)
@click.option(
    "--benchmark",
    "benchmark_id",
    required=True,
    type=click.Choice(list(DATASET_CONFIG.keys())),
    help="Benchmark to run (hle, gaia, apex)",
)
@click.option(
    "--limit",
    "max_samples",
    type=int,
    default=None,
    help="Maximum number of samples to evaluate",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--vllm-url",
    help="vLLM server URL (required for raw model names, default: http://localhost:8000/v1)",
)
@click.option(
    "--per-action",
    is_flag=True,
    default=False,
    help="Record per-action energy breakdown (tool calls, LM inference)",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    default=False,
    help="Disable energy telemetry collection",
)
@click.option(
    "--skip-warmup",
    is_flag=True,
    default=False,
    help="Skip server warmup phase (includes cold-start costs in measurements)",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True),
    help="Path to trained policy checkpoint (required for orchestrator agent)",
)
@click.option(
    "--auto-server",
    is_flag=True,
    default=False,
    help="Auto-start/stop inference servers (excludes startup/shutdown from profiling)",
)
@click.option(
    "--submodel",
    "submodels",
    multiple=True,
    help="Submodel spec: alias:backend:model_id (e.g., math:vllm:Qwen/Qwen2.5-Math-72B)",
)
@click.option(
    "--base-port",
    type=int,
    default=8000,
    help="Base port for vLLM servers when using --auto-server (default: 8000)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible benchmark sampling",
)
@click.option(
    "--resource-config",
    type=click.Choice(["1gpu_8cpu", "4gpu_32cpu"]),
    default=None,
    help="Hardware resource configuration for grid search experiments",
)
def bench(
    agent_id: str,
    model: str | None,
    benchmark_id: str,
    max_samples: int | None,
    output_dir: str | None,
    vllm_url: str | None,
    per_action: bool,
    no_telemetry: bool,
    skip_warmup: bool,
    checkpoint_path: str | None,
    auto_server: bool,
    submodels: tuple[str, ...],
    base_port: int,
    seed: int | None,
    resource_config: str | None,
) -> None:
    """Run agent benchmarks with energy telemetry for IPW measurement."""
    # Set random seed for reproducible benchmark sampling
    if seed is not None:
        random.seed(seed)
        info(f"  Random seed: {seed}")

    # Validate agent-specific requirements
    if agent_id == "orchestrator":
        if checkpoint_path is None:
            error("Orchestrator agent requires --checkpoint-path to load trained policy.")
            error("Example: ipw bench --agent orchestrator --checkpoint-path /path/to/checkpoint --benchmark hle")
            raise click.Abort()
        # Orchestrator doesn't use model - set dummy values
        model_id = "orchestrator-policy"
        base_url = None
    else:
        if model is None:
            error(f"--model is required for {agent_id} agent.")
            raise click.Abort()
        # Resolve model preset to actual model ID and URL
        try:
            model_id, base_url = resolve_model(model, vllm_url)
        except ValueError as e:
            error(str(e))
            raise click.Abort()

    # Determine telemetry granularity from --per-action flag
    telemetry_granularity = "per-action" if per_action else "benchmark"

    info(f"Running benchmark: {benchmark_id}")
    info(f"  Agent: {agent_id}")
    if agent_id == "orchestrator":
        info(f"  Checkpoint: {checkpoint_path}")
    else:
        info(f"  Model: {model_id}")
        if not auto_server:
            info(f"  vLLM URL: {base_url}")
    if max_samples:
        info(f"  Limit: {max_samples}")
    if no_telemetry:
        info("  Telemetry: disabled")
    else:
        info(f"  Telemetry: enabled ({telemetry_granularity})")
    if auto_server:
        info(f"  Auto-server: enabled (base port: {base_port})")
        if submodels:
            info(f"  Submodels: {len(submodels)}")
            for spec in submodels:
                info(f"    - {spec}")
    if agent_id != "orchestrator":
        if skip_warmup:
            info("  Warmup: skipped (cold-start included)")
        else:
            info("  Warmup: enabled (cold-start excluded)")

    try:
        metrics = execute_benchmark(
            client_id="vllm",  # Always use vLLM client for model presets
            model_name=model_id,
            agent_id=agent_id,
            dataset_id=benchmark_id,
            max_samples=max_samples,
            client_base_url=base_url,
            output_dir=output_dir,
            enable_telemetry=not no_telemetry,
            telemetry_granularity=telemetry_granularity,
            skip_warmup=skip_warmup,
            checkpoint_path=checkpoint_path,
            auto_server=auto_server,
            submodels=submodels,
            base_port=base_port,
            seed=seed,
            resource_config=resource_config,
        )

        success("Benchmark completed!")
        info("\nResults:")
        for key, value in metrics.items():
            if isinstance(value, float):
                info(f"  {key}: {value:.4f}")
            else:
                info(f"  {key}: {value}")

    except Exception as e:
        error(f"Benchmark failed: {e}")
        raise click.Abort()


__all__ = ["bench"]
