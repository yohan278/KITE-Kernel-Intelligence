"""Inference server management commands.

Provides commands to start, stop, and check status of inference servers
used for local model serving (Ollama, vLLM).

Usage:
    # Start servers
    ipw servers start --ollama
    ipw servers start --vllm --model llama3.2:1b

    # Launch and wait for ready (recommended for benchmarking)
    ipw servers launch --ollama
    ipw servers launch --vllm --model llama3.2:1b --wait-timeout 120

    # Stop servers
    ipw servers stop --all

    # Check status
    ipw servers status
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

import click

from ipw.cli._console import error, info, success, warning


# Server health check timeouts
DEFAULT_WAIT_TIMEOUT = 60  # seconds
POLL_INTERVAL = 1.0  # seconds


@click.group()
def servers() -> None:
    """Manage inference servers (Ollama, vLLM)."""
    pass


@servers.command()
@click.option(
    "--ollama",
    is_flag=True,
    help="Start Ollama server",
)
@click.option(
    "--vllm",
    is_flag=True,
    help="Start vLLM server",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model to load (required for vLLM)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to run server on (default: 11434 for Ollama, 8000 for vLLM)",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization for vLLM (default: 0.9)",
)
def start(
    ollama: bool,
    vllm: bool,
    model: Optional[str],
    port: Optional[int],
    gpu_memory_utilization: float,
) -> None:
    """Start inference server(s).

    Examples:
        ipw servers start --ollama
        ipw servers start --vllm --model llama3.2:1b
    """
    if not ollama and not vllm:
        error("Please specify --ollama or --vllm")
        raise click.Abort()

    if ollama and vllm:
        error("Please specify only one of --ollama or --vllm")
        raise click.Abort()

    if ollama:
        _start_ollama(port)
    elif vllm:
        if not model:
            error("--model is required for vLLM")
            raise click.Abort()
        _start_vllm(model, port, gpu_memory_utilization)


def _start_ollama(port: Optional[int]) -> None:
    """Start Ollama server."""
    actual_port = port or 11434

    info(f"Starting Ollama server on port {actual_port}...")

    try:
        # Check if ollama is installed
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error("Ollama not found. Install from https://ollama.ai")
            raise click.Abort()

        # Start ollama serve
        env = {"OLLAMA_HOST": f"0.0.0.0:{actual_port}"}
        subprocess.Popen(
            ["ollama", "serve"],
            env={**subprocess.os.environ, **env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        success(f"Ollama server started on http://localhost:{actual_port}")
        info("Use 'ollama pull <model>' to download models")

    except FileNotFoundError:
        error("Ollama not found. Install from https://ollama.ai")
        raise click.Abort()


def _start_vllm(
    model: str,
    port: Optional[int],
    gpu_memory_utilization: float,
) -> None:
    """Start vLLM server."""
    actual_port = port or 8000

    info(f"Starting vLLM server with model {model} on port {actual_port}...")

    try:
        # Build vLLM command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(actual_port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
        ]

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        success(f"vLLM server started on http://localhost:{actual_port}")
        info(f"Model: {model}")
        info("OpenAI-compatible API available at /v1/")

    except Exception as e:
        error(f"Failed to start vLLM: {e}")
        raise click.Abort()


@servers.command()
@click.option(
    "--ollama",
    is_flag=True,
    help="Launch Ollama server",
)
@click.option(
    "--vllm",
    is_flag=True,
    help="Launch vLLM server",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model to load (required for vLLM, optional for Ollama to pre-pull)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to run server on (default: 11434 for Ollama, 8000 for vLLM)",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization for vLLM (default: 0.9)",
)
@click.option(
    "--wait-timeout",
    type=int,
    default=DEFAULT_WAIT_TIMEOUT,
    help=f"Timeout in seconds waiting for server to be ready (default: {DEFAULT_WAIT_TIMEOUT})",
)
def launch(
    ollama: bool,
    vllm: bool,
    model: Optional[str],
    port: Optional[int],
    gpu_memory_utilization: float,
    wait_timeout: int,
) -> None:
    """Launch inference server and wait until ready.

    This command starts the server and blocks until it's ready to accept
    requests. Recommended for use before running benchmarks to ensure
    server warmup costs are excluded from measurements.

    Examples:
        ipw servers launch --ollama
        ipw servers launch --ollama --model llama3.2:1b
        ipw servers launch --vllm --model meta-llama/Llama-3.2-1B --wait-timeout 120
    """
    if not ollama and not vllm:
        error("Please specify --ollama or --vllm")
        raise click.Abort()

    if ollama and vllm:
        error("Please specify only one of --ollama or --vllm")
        raise click.Abort()

    if ollama:
        _launch_ollama(port, model, wait_timeout)
    elif vllm:
        if not model:
            error("--model is required for vLLM")
            raise click.Abort()
        _launch_vllm(model, port, gpu_memory_utilization, wait_timeout)


def _launch_ollama(port: Optional[int], model: Optional[str], timeout: int) -> None:
    """Launch Ollama server and wait for it to be ready."""
    actual_port = port or 11434

    # Check if already running
    if _check_ollama_status():
        info("Ollama server already running")
    else:
        # Start server
        _start_ollama(port)
        info("Waiting for Ollama to be ready...")

        # Wait for server to be ready
        if not _wait_for_server("ollama", actual_port, timeout):
            error(f"Ollama server not ready after {timeout}s")
            raise click.Abort()

    success(f"Ollama server ready at http://localhost:{actual_port}")

    # Optionally pull/warm up model
    if model:
        info(f"Pulling model {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for model download
            )
            if result.returncode == 0:
                success(f"Model {model} ready")
                # Run a warmup query
                info("Running warmup inference...")
                subprocess.run(
                    ["ollama", "run", model, "Hello"],
                    capture_output=True,
                    timeout=60,
                )
                success("Warmup complete")
            else:
                warning(f"Failed to pull model: {result.stderr}")
        except subprocess.TimeoutExpired:
            warning("Model pull timed out")
        except Exception as e:
            warning(f"Failed to pull model: {e}")


def _launch_vllm(
    model: str,
    port: Optional[int],
    gpu_memory_utilization: float,
    timeout: int,
) -> None:
    """Launch vLLM server and wait for it to be ready."""
    actual_port = port or 8000

    # Check if already running
    if _check_vllm_status():
        info("vLLM server already running")
        success(f"vLLM server ready at http://localhost:{actual_port}")
        return

    # Start server
    _start_vllm(model, port, gpu_memory_utilization)
    info(f"Waiting for vLLM to load model {model} (this may take a while)...")

    # Wait for server to be ready (vLLM takes longer to load models)
    if not _wait_for_server("vllm", actual_port, timeout):
        error(f"vLLM server not ready after {timeout}s")
        info("Tip: Try increasing --wait-timeout for larger models")
        raise click.Abort()

    success(f"vLLM server ready at http://localhost:{actual_port}")
    info("OpenAI-compatible API available at /v1/")


def _wait_for_server(server_type: str, port: int, timeout: int) -> bool:
    """Wait for server to become ready.

    Args:
        server_type: 'ollama' or 'vllm'
        port: Server port
        timeout: Maximum wait time in seconds

    Returns:
        True if server became ready, False if timeout
    """
    import urllib.request
    import urllib.error

    if server_type == "ollama":
        url = f"http://localhost:{port}/api/version"
    else:
        url = f"http://localhost:{port}/v1/models"

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, Exception):
            pass
        time.sleep(POLL_INTERVAL)

    return False


@servers.command()
@click.option(
    "--ollama",
    is_flag=True,
    help="Stop Ollama server",
)
@click.option(
    "--vllm",
    is_flag=True,
    help="Stop vLLM server",
)
@click.option(
    "--all",
    "stop_all",
    is_flag=True,
    help="Stop all managed inference servers",
)
def stop(ollama: bool, vllm: bool, stop_all: bool) -> None:
    """Stop inference server(s).

    Examples:
        ipw servers stop --ollama
        ipw servers stop --all
    """
    if not ollama and not vllm and not stop_all:
        error("Please specify --ollama, --vllm, or --all")
        raise click.Abort()

    if stop_all or ollama:
        _stop_ollama()

    if stop_all or vllm:
        _stop_vllm()


def _stop_ollama() -> None:
    """Stop Ollama server."""
    info("Stopping Ollama server...")
    try:
        subprocess.run(
            ["pkill", "-f", "ollama serve"],
            capture_output=True,
        )
        success("Ollama server stopped")
    except Exception as e:
        warning(f"Could not stop Ollama: {e}")


def _stop_vllm() -> None:
    """Stop vLLM server."""
    info("Stopping vLLM server...")
    try:
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
        )
        success("vLLM server stopped")
    except Exception as e:
        warning(f"Could not stop vLLM: {e}")


@servers.command()
def status() -> None:
    """Show status of inference servers."""
    info("Checking inference server status...\n")

    # Check Ollama
    ollama_status = _check_ollama_status()
    if ollama_status:
        success(f"Ollama: Running on {ollama_status}")
    else:
        warning("Ollama: Not running")

    # Check vLLM
    vllm_status = _check_vllm_status()
    if vllm_status:
        success(f"vLLM: Running on {vllm_status}")
    else:
        warning("vLLM: Not running")


def _check_ollama_status() -> Optional[str]:
    """Check if Ollama is running and return endpoint if so."""
    try:
        import urllib.request
        import urllib.error

        # Try default Ollama endpoint
        url = "http://localhost:11434/api/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return "http://localhost:11434"
    except (urllib.error.URLError, TimeoutError, Exception):
        pass
    return None


def _check_vllm_status() -> Optional[str]:
    """Check if vLLM is running and return endpoint if so."""
    try:
        import urllib.request
        import urllib.error

        # Try default vLLM endpoint
        url = "http://localhost:8000/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return "http://localhost:8000"
    except (urllib.error.URLError, TimeoutError, Exception):
        pass
    return None


__all__ = ["servers"]
