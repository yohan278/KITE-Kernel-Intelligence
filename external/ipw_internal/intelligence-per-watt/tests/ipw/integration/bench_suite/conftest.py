"""Fixtures for bench integration tests.

This module provides pytest fixtures for model configurations, agent configurations,
vLLM server health checks, and resource management.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import pytest

from .resource_utils import ResourceManager


# Model configurations for testing
# Note: vllm_url includes /v1 as expected by OpenAI-compatible clients
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "vllm_url": "http://localhost:8000/v1",
        "tensor_parallel_size": 1,
    },
    "gpt-oss-20b": {
        # MoE model with <3B active parameters - fits on 1 GPU
        "model_id": "openai/gpt-oss-20b",
        "vllm_url": "http://localhost:8000/v1",
        "tensor_parallel_size": 1,
    },
}


# Agent configurations for testing
AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "react": {
        "requires_model": True,
        "requires_checkpoint": False,
    },
    "openhands": {
        "requires_model": True,
        "requires_checkpoint": False,
    },
    "orchestrator": {
        "requires_model": False,
        "requires_checkpoint": True,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration by name.

    Args:
        model_name: Model name (e.g., "qwen3-8b", "gpt-oss-20b")

    Returns:
        Model configuration dictionary

    Raises:
        KeyError: If model name not found
    """
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise KeyError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_CONFIGS[model_name]


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get agent configuration by name.

    Args:
        agent_name: Agent name (e.g., "react", "openhands", "orchestrator")

    Returns:
        Agent configuration dictionary

    Raises:
        KeyError: If agent name not found
    """
    if agent_name not in AGENT_CONFIGS:
        available = ", ".join(AGENT_CONFIGS.keys())
        raise KeyError(f"Unknown agent: {agent_name}. Available: {available}")
    return AGENT_CONFIGS[agent_name]


def get_checkpoint_path(model: str, agent: str) -> Optional[str]:
    """Return checkpoint path for orchestrator, None for other agents.

    For orchestrator agent, looks for a checkpoint in this order:
    1. ORCHESTRATOR_CHECKPOINT environment variable
    2. Default checkpoint paths for known models
    3. None (test will be skipped)

    Args:
        model: Model name
        agent: Agent name

    Returns:
        Checkpoint path for orchestrator if found, None otherwise
    """
    if agent != "orchestrator":
        return None

    # Check environment variable first
    env_checkpoint = os.environ.get("ORCHESTRATOR_CHECKPOINT")
    if env_checkpoint and os.path.exists(env_checkpoint):
        return env_checkpoint

    # Check default checkpoint paths
    default_paths = [
        f"/data/checkpoints/{model}",
        f"./checkpoints/{model}",
        os.path.expanduser(f"~/.cache/ipw/checkpoints/{model}"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path

    # No checkpoint found - return a marker that will trigger skip
    return None


def _check_vllm_health(base_url: str = "http://localhost:8000", timeout: float = 5.0) -> bool:
    """Check if vLLM server is healthy and responding.

    Args:
        base_url: vLLM server base URL
        timeout: Request timeout in seconds

    Returns:
        True if server is healthy, False otherwise
    """
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, Exception):
        return False


def _check_model_loaded(model_id: str, base_url: str = "http://localhost:8000") -> bool:
    """Check if a specific model is loaded in vLLM.

    Args:
        model_id: Model identifier to check for
        base_url: vLLM server base URL

    Returns:
        True if model is loaded, False otherwise
    """
    import json

    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode())
            models = data.get("data", [])
            return any(m.get("id") == model_id for m in models)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception):
        return False


@pytest.fixture(scope="session")
def vllm_server():
    """Ensure vLLM server is running and accessible.

    Skips test if vLLM server is not available.

    Yields:
        Base URL of the vLLM server
    """
    base_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    if not _check_vllm_health(base_url):
        pytest.skip(
            f"vLLM server not available at {base_url}. "
            "Start vLLM server or set VLLM_URL environment variable."
        )
    yield base_url


def _start_vllm_server(model_id: str, timeout: int = 300) -> bool:
    """Start vLLM server with specified model.

    Args:
        model_id: Model identifier to load
        timeout: Timeout in seconds waiting for server to be ready

    Returns:
        True if server started successfully, False otherwise
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--port", "8000",
        "--gpu-memory-utilization", "0.9",
    ]

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _check_vllm_health("http://localhost:8000"):
            return True
        time.sleep(2.0)

    return False


def _stop_vllm_server() -> None:
    """Stop vLLM server."""
    subprocess.run(
        ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
        capture_output=True,
    )
    # Wait a moment for process to terminate
    time.sleep(2.0)


@pytest.fixture(scope="module")
def vllm_server_for_model(request):
    """Start vLLM server with a specific model, stop after test module.

    This fixture manages the vLLM server lifecycle:
    - Starts vLLM with the requested model before tests
    - Stops vLLM after tests complete

    Usage:
        @pytest.mark.parametrize("vllm_server_for_model", ["Qwen/Qwen3-8B-Instruct"], indirect=True)
        def test_something(vllm_server_for_model):
            model_id = vllm_server_for_model
            # model_id is the loaded model

    Yields:
        Model ID that was loaded
    """
    model_id = request.param

    # Check if server is already running with correct model
    if _check_vllm_health("http://localhost:8000"):
        if _check_model_loaded(model_id, "http://localhost:8000"):
            yield model_id
            return
        # Wrong model loaded, stop and restart
        _stop_vllm_server()

    # Start server with model
    if not _start_vllm_server(model_id, timeout=300):
        pytest.skip(f"Failed to start vLLM server with model {model_id}")

    yield model_id

    # Stop server after tests
    _stop_vllm_server()


@pytest.fixture
def resource_manager():
    """Provide resource configuration context manager.

    Returns:
        ResourceManager instance for configuring GPU/CPU resources
    """
    return ResourceManager()


@pytest.fixture
def model_configs():
    """Provide model configurations.

    Returns:
        Dictionary of model configurations
    """
    return MODEL_CONFIGS


@pytest.fixture
def agent_configs():
    """Provide agent configurations.

    Returns:
        Dictionary of agent configurations
    """
    return AGENT_CONFIGS


@pytest.fixture
def check_model_loaded():
    """Provide function to check if a model is loaded.

    Returns:
        Function that checks if a model is loaded in vLLM
    """
    return _check_model_loaded
