"""Main evaluation loop for grid search."""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from grid_eval.config import (
    GPU_TYPE_REGISTRY,
    MODEL_REGISTRY,
    OLLAMA_MODEL_MAPPING,
    RESOURCE_CONFIG_REGISTRY,
    AgentType,
    BenchmarkType,
    GridConfig,
    GpuType,
    ModelType,
    ResourceConfig,
)
from grid_eval.hardware import ResourceManager
from grid_eval.output import GridMetadata, JSONLWriter, QueryResult
from grid_eval.progress import ProgressTracker
from ipw.cli.server_manager import InferenceServerManager, PortConflictError, ServerConfig
from ipw.cli.vllm_lifecycle import (
    ModelMismatchError,
    VLLMProcessDetector,
    VLLMServerRegistry,
    cleanup_orphaned_servers,
)
from ipw.execution.telemetry_session import TelemetrySession
from ipw.telemetry.collector import EnergyMonitorCollector

logger = logging.getLogger(__name__)

# Agent MCP tool configuration for grid evaluation
# NOTE: Local LLM sub-delegation is DISABLED for pure single-model evaluation
# NOTE: Retrieval tools are DISABLED to avoid interference with LM energy measurements
AGENT_TOOLS = {
    # Utility tools (always available, zero or minimal cost)
    "utility": [
        "calculator",      # Math expression evaluation ($0)
        "think",           # Internal reasoning scratchpad ($0)
        "code_interpreter",  # Python execution (sandboxed, ~$0)
        "file_read",       # Read file contents ($0)
        "file_write",      # Write/append to files ($0)
        "web_search",      # Tavily web search API ($0.01/search)
    ],
    # Cloud LLM APIs for sub-delegation (enabled for delegation experiments)
    "cloud_llms": [
        "openai:gpt-4o",
        "openai:gpt-5-mini-2025-08-07",
        "anthropic:claude-sonnet-4-20250514",
        "anthropic:claude-3-5-haiku-20241022",
        "openrouter:google/gemini-2.5-flash",
        "openrouter:google/gemini-2.5-pro",
    ],
}


class GridEvalRunner:
    """Main evaluation runner for grid search.

    Loads benchmark datasets, instantiates agents, and runs evaluations
    across all configuration combinations.

    Grid Search Loop Order (outermost to innermost):
        1. GpuType (hardware choice) - e.g., A100, H100, MI300X
        2. ResourceConfig (resource allocation) - e.g., 1gpu_8cpu, 4gpu_32cpu
        3. AgentType (agent harness) - e.g., react, openhands
        4. ModelType (LMs) - e.g., qwen3-8b, gpt-oss-20b
        5. BenchmarkType (benchmarks) - e.g., gaia, hle [INNERMOST]

    Example:
        >>> config = GridConfig(queries_per_benchmark=10)
        >>> runner = GridEvalRunner(config)
        >>> runner.run(output_dir=Path("results/grid_eval"))
    """

    def __init__(
        self,
        config: GridConfig,
        vllm_base_url: str = "http://localhost:8000",
        openai_base_url: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        conflict_policy: str = "kill",
        cleanup_orphans: bool = True,
        backend: str = "auto",
        grader_model: Optional[str] = None,
        grader_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the runner.

        Args:
            config: Grid configuration
            vllm_base_url: Base URL for vLLM server (default, may be overridden)
            openai_base_url: Base URL for OpenAI-compatible API
            ollama_base_url: Base URL for Ollama server (for Apple Silicon)
            conflict_policy: How to handle port conflicts ("fail", "kill", "skip_port")
            cleanup_orphans: Whether to clean up orphaned servers on startup
            backend: Inference backend ("auto", "vllm", "ollama"). Auto selects based on GPU vendor.
            grader_model: Model to use for LLM judge scoring (default from config)
            grader_api_key: API key for grader model (default from config/env)
        """
        self.config = config
        self.vllm_base_url = vllm_base_url
        self.openai_base_url = openai_base_url
        self._ollama_base_url = ollama_base_url
        self._conflict_policy = conflict_policy
        self._cleanup_orphans = cleanup_orphans
        self._backend_override = backend

        # LLM Judge configuration
        self.grader_model = grader_model or config.grader_model
        self.grader_api_key = grader_api_key or config.grader_api_key

        # Cache for loaded datasets
        self._hle_samples: Optional[List[Any]] = None
        self._gaia_samples: Optional[List[Any]] = None
        self._swebench_samples: Optional[List[Any]] = None
        self._apex_samples: Optional[List[Any]] = None
        self._browsecomp_samples: Optional[List[Any]] = None
        self._deepresearch_samples: Optional[List[Any]] = None
        self._simpleqa_samples: Optional[List[Any]] = None
        self._swefficiency_samples: Optional[List[Any]] = None

        # vLLM server management
        self._vllm_manager: Optional[InferenceServerManager] = None
        self._current_gpu_count: Optional[int] = None
        self._current_model_id: Optional[str] = None
        self._current_resource_config: Optional[ResourceConfig] = None
        self._telemetry_session: Optional[TelemetrySession] = None

        # Ollama client for Apple Silicon
        self._ollama_client: Optional[Any] = None
        self._current_ollama_model: Optional[str] = None

        # Lifecycle management
        self._registry = VLLMServerRegistry()
        self._detector = VLLMProcessDetector()
        self._orphans_cleaned = False

    def _should_skip_combination(
        self, model_type: ModelType, resource_config: ResourceConfig
    ) -> Optional[str]:
        """Check if combination should be skipped due to hardware constraints.

        Args:
            model_type: Model type to check
            resource_config: Resource configuration

        Returns:
            Reason string if should skip, None if OK to run
        """
        model_config = MODEL_REGISTRY[model_type]
        min_gpus = model_config.get("min_gpus", 1)
        rc_config = RESOURCE_CONFIG_REGISTRY[resource_config]
        available_gpus = rc_config.get("gpu_count", 1)

        if available_gpus < min_gpus:
            return f"Insufficient GPUs: requires {min_gpus}, has {available_gpus}"
        return None

    def _ensure_vllm_server(
        self, resource_config: ResourceConfig, model_config: Dict[str, Any]
    ) -> str:
        """Ensure vLLM server is running with correct model and GPU configuration.

        Restarts the server if model or GPU count changes. Handles port conflicts
        and orphaned servers.

        Args:
            resource_config: Resource configuration (determines GPU count)
            model_config: Model configuration from MODEL_REGISTRY

        Returns:
            vLLM base URL

        Raises:
            PortConflictError: If port is in use and conflict_policy is "fail"
            ModelMismatchError: If server is running with wrong model
        """
        if model_config["type"] != "vllm":
            return self.vllm_base_url  # Not using vLLM

        # Cleanup orphaned servers on first call
        if self._cleanup_orphans and not self._orphans_cleaned:
            logger.info("Cleaning up orphaned vLLM servers...")
            cleaned = cleanup_orphaned_servers(self._registry, self._detector)
            if cleaned:
                logger.info(f"Cleaned up orphaned servers on ports: {cleaned}")
            self._orphans_cleaned = True

        rc_config = RESOURCE_CONFIG_REGISTRY[resource_config]
        gpu_count = rc_config.get("gpu_count", 1)
        model_id = model_config["model_id"]

        # Respect max_tp: cap tensor parallelism for models with head-count constraints
        max_tp = model_config.get("max_tp")
        if max_tp and gpu_count > max_tp:
            logger.info(
                f"Capping TP from {gpu_count} to {max_tp} for {model_id} "
                f"(max_tp={max_tp}, {gpu_count} GPUs available)"
            )
            gpu_count = max_tp

        # Check if we need to restart (model or GPU count changed)
        if (
            self._current_gpu_count == gpu_count
            and self._current_model_id == model_id
            and self._vllm_manager
        ):
            url = self._vllm_manager.get_url("main")
            if url and self._verify_model(url, model_id):
                return url
            # Model mismatch or server not responding - need to restart
            logger.warning(f"Server on {url} not serving expected model {model_id}")

        # Stop existing server
        if self._vllm_manager:
            logger.info(
                f"Stopping vLLM server (was {self._current_model_id} "
                f"on {self._current_gpu_count} GPUs)"
            )
            self._vllm_manager.stop_all()
            self._vllm_manager = None
            # Wait for GPU memory to be freed after server stop
            self._wait_for_gpu_memory_release()

        # Extract port from configured vllm_base_url
        from urllib.parse import urlparse
        _parsed = urlparse(self.vllm_base_url.replace("/v1", ""))
        port = _parsed.port or 8000

        # Check if server at our configured URL already serves the right model
        url_to_check = f"http://localhost:{port}/v1"
        if self._verify_model(url_to_check, model_id):
            logger.info(f"Reusing existing vLLM server on port {port}")
            self.vllm_base_url = url_to_check
            return url_to_check

        # Extract port from configured vllm_base_url
        from urllib.parse import urlparse
        _parsed = urlparse(self.vllm_base_url.replace("/v1", ""))
        port = _parsed.port or 8000

        # Check if server at our configured URL already serves the right model
        url_to_check = f"http://localhost:{port}/v1"
        if self._verify_model(url_to_check, model_id):
            logger.info(f"Reusing existing vLLM server on port {port}")
            self.vllm_base_url = url_to_check
            return url_to_check

        # Check for port conflict before starting
        if self._detector._is_port_in_use(port):
            lock_info = self._registry.get_lock_info(port)
            proc_info = self._detector.find_vllm_on_port(port)

            # Check if it's our server with correct model
            if lock_info and lock_info.model_id == model_id:
                url = f"http://localhost:{port}/v1"
                if self._verify_model(url, model_id):
                    logger.info(f"Reusing existing vLLM server on port {port}")
                    self.vllm_base_url = url
                    return url

            # Handle conflict
            if self._conflict_policy == "fail":
                raise PortConflictError(
                    port=port,
                    existing_model=proc_info.get("model") if proc_info else None,
                    requested_model=model_id,
                    owner=lock_info.owner if lock_info else "external",
                )
            elif self._conflict_policy == "kill":
                if proc_info and proc_info.get("pid"):
                    logger.info(f"Killing conflicting process {proc_info['pid']} on port {port}")
                    self._detector.kill_process(proc_info["pid"], force=True)
                    time.sleep(2)  # Wait for port to be released
                self._registry.release_lock(port)

        # Start new server with correct model and GPU count
        logger.info(f"Starting vLLM server: {model_id} with {gpu_count} GPUs")

        # Build extra_args for vLLM (e.g., quantization)
        extra_args = {}
        if model_config.get("quantization"):
            extra_args["quantization"] = model_config["quantization"]
            logger.info(f"  Using quantization: {model_config['quantization']}")
        if model_config.get("trust_remote_code"):
            extra_args["trust_remote_code"] = True
        if model_config.get("enforce_eager"):
            extra_args["enforce_eager"] = True
        if model_config.get("enable_expert_parallel"):
            extra_args["enable_expert_parallel"] = True
        if model_config.get("tokenizer"):
            extra_args["tokenizer"] = model_config["tokenizer"]
        if model_config.get("limit_mm_per_prompt"):
            extra_args["limit_mm_per_prompt"] = model_config["limit_mm_per_prompt"]
        # In batch mode, set max-num-seqs to match the model's batch size
        if self.config.batch_mode:
            max_batch = model_config.get("max_batch_size")
            if max_batch:
                extra_args["max_num_seqs"] = max_batch
                logger.info(f"  Batch mode: max-num-seqs={max_batch}")
        # Enable native tool calling if model has a tool_call_parser configured
        # (required for OpenHands agent which uses the OpenAI tool calling API)
        # The server_manager pops tool_call_parser and adds --enable-auto-tool-choice automatically
        if model_config.get("tool_call_parser"):
            extra_args["tool_call_parser"] = model_config["tool_call_parser"]
            logger.info(f"  Tool calling enabled: parser={model_config['tool_call_parser']}")
        # Note: On PCIe-only GPUs (no NVLink), you may need:
        #   extra_args["enforce_eager"] = True
        #   extra_args["disable_custom_all_reduce"] = True
        # Omitted here for NVLink-connected GPUs (H100 SXM, etc.)
        # Always use patched launcher for transformers 5.x compat
        # (tokenizer patch + config registration)
        launch_module = "grid_eval.vllm_launcher"

        # Use model-specific max_model_len if provided, otherwise ServerConfig default
        max_model_len = model_config.get("max_model_len")
        config_kwargs: dict = dict(
            model_id=model_id,
            alias="main",
            backend="vllm",
            port=port,
            tensor_parallel_size=gpu_count,
            gpu_ids=list(range(gpu_count)),
            extra_args=extra_args,
            launch_module=launch_module,
        )
        if max_model_len is not None:
            config_kwargs["max_model_len"] = max_model_len
        gpu_mem_util = model_config.get("gpu_memory_utilization")
        if gpu_mem_util is not None:
            config_kwargs["gpu_memory_utilization"] = gpu_mem_util
        env_vars = model_config.get("env_vars", {})
        if env_vars:
            config_kwargs["env_vars"] = env_vars
        config = ServerConfig(**config_kwargs)
        self._vllm_manager = InferenceServerManager(
            [config],
            auto_assign_gpus=False,
            conflict_policy=self._conflict_policy,
            owner="grid_eval",
        )
        # Scale timeout with model size: base 600s + 10s per billion params, capped at 5400s
        # e.g., 16B -> 760s, 353B -> 4130s, 1000B -> 5400s (90 min, for large MoE weight loading)
        total_params_b = model_config.get("total_params_b", 0) or 0
        wait_timeout = min(5400.0, max(600.0, 600.0 + total_params_b * 10.0))
        logger.info(f"  Server startup timeout: {wait_timeout:.0f}s (model: {total_params_b:.0f}B params)")
        urls = self._vllm_manager.start_all(wait_timeout=wait_timeout)
        self._current_gpu_count = gpu_count
        self._current_model_id = model_id
        self._current_resource_config = resource_config

        # Warmup the server
        self._vllm_manager.warmup_all()

        # Return base URL with /v1 suffix (agno's OpenAILike expects it)
        return urls["main"]

    def _kill_gpu_processes(self, gpu_indices: List[int]) -> int:
        """Find and kill processes holding GPU memory on the given devices.

        Uses nvidia-smi to discover compute processes, then SIGKILL them.

        Args:
            gpu_indices: GPU device indices to clean.

        Returns:
            Number of processes killed.
        """
        import subprocess as _sp

        try:
            result = _sp.run(
                ["nvidia-smi",
                 "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return 0

            # Map GPU UUIDs to indices
            idx_result = _sp.run(
                ["nvidia-smi", "--query-gpu=index,uuid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            uuid_to_idx: dict[str, int] = {}
            if idx_result.returncode == 0:
                for line in idx_result.stdout.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        uuid_to_idx[parts[1].strip()] = int(parts[0].strip())

            pids_to_kill: set[int] = set()
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    gpu_uuid = parts[0].strip()
                    pid = int(parts[1].strip())
                    gpu_idx = uuid_to_idx.get(gpu_uuid, -1)
                    if gpu_idx in gpu_indices:
                        pids_to_kill.add(pid)

            killed = 0
            for pid in pids_to_kill:
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"  Killed orphan GPU process {pid}")
                    killed += 1
                except ProcessLookupError:
                    pass  # Already dead
                except PermissionError:
                    logger.warning(f"  Permission denied killing GPU process {pid}")
            return killed

        except Exception as e:
            logger.warning(f"  Failed to enumerate GPU processes: {e}")
            return 0

    def _wait_for_gpu_memory_release(
        self, max_wait: float = 180.0, min_free_gb: float = 60.0
    ) -> None:
        """Wait for GPU memory to be freed after stopping a vLLM server.

        Strategy:
          1. Wait up to 30s for graceful release (SIGTERM propagation).
          2. If memory still held, actively kill orphan GPU processes.
          3. Wait remaining time for memory to actually free.

        Args:
            max_wait: Maximum seconds to wait for memory release.
            min_free_gb: Minimum free memory (GB) per GPU to consider released.
        """
        import subprocess as _sp

        cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_devs:
            return

        gpu_indices = [int(x.strip()) for x in cuda_devs.split(",") if x.strip()]
        if not gpu_indices:
            return

        def _gpus_are_free() -> bool:
            try:
                result = _sp.run(
                    ["nvidia-smi", "--query-gpu=index,memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode != 0:
                    return False
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        idx = int(parts[0].strip())
                        free_mb = float(parts[1].strip())
                        if idx in gpu_indices and free_mb < min_free_gb * 1024:
                            return False
                return True
            except Exception:
                return False

        start = time.time()

        # Phase 1: brief grace period for normal cleanup (up to 30s)
        grace_deadline = min(start + 30.0, start + max_wait)
        while time.time() < grace_deadline:
            if _gpus_are_free():
                elapsed = time.time() - start
                logger.info(f"GPU memory released successfully ({elapsed:.0f}s)")
                return
            time.sleep(2)

        # Phase 2: kill orphan processes still holding GPU memory
        logger.info("GPU memory still held after 30s — killing orphan GPU processes")
        killed = self._kill_gpu_processes(gpu_indices)
        if killed:
            logger.info(f"  Killed {killed} orphan process(es), waiting for memory release")

        # Phase 3: wait for memory to actually free after kills
        while time.time() - start < max_wait:
            if _gpus_are_free():
                elapsed = time.time() - start
                logger.info(f"GPU memory released successfully ({elapsed:.0f}s)")
                return
            time.sleep(2)

        logger.warning(
            f"GPU memory not fully released after {max_wait:.0f}s, proceeding anyway"
        )

    def _verify_model(self, url: str, expected_model: str) -> bool:
        """Verify the expected model is loaded on the server.

        Args:
            url: Server URL
            expected_model: Expected model ID

        Returns:
            True if model matches
        """
        actual_model = self._detector.query_loaded_model(
            int(url.split(":")[-1].split("/")[0])  # Extract port from URL
        )
        if actual_model is None:
            return False
        return expected_model in actual_model or actual_model in expected_model

    def _get_backend_for_gpu(self, gpu_type: GpuType) -> str:
        """Determine the inference backend for a GPU type.

        Args:
            gpu_type: GPU hardware type

        Returns:
            Backend name ("vllm" or "ollama")
        """
        if self._backend_override != "auto":
            return self._backend_override

        gpu_config = GPU_TYPE_REGISTRY.get(gpu_type, {})
        return gpu_config.get("default_backend", "vllm")

    def _ensure_ollama_ready(
        self, model_type: ModelType, model_config: Dict[str, Any]
    ) -> str:
        """Ensure Ollama is running and model is available.

        Args:
            model_type: Model type enum
            model_config: Model configuration from MODEL_REGISTRY

        Returns:
            Ollama base URL

        Raises:
            RuntimeError: If Ollama is not running or model not available
        """
        from ipw.clients.ollama import OllamaClient

        # Initialize Ollama client if needed
        if self._ollama_client is None:
            self._ollama_client = OllamaClient(self._ollama_base_url)

        # Check Ollama is running
        if not self._ollama_client.health():
            raise RuntimeError(
                f"Ollama not running at {self._ollama_base_url}. "
                "Start with: ollama serve"
            )

        # Get Ollama model name
        ollama_model = OLLAMA_MODEL_MAPPING.get(model_type)
        if not ollama_model:
            raise RuntimeError(
                f"No Ollama mapping for model {model_type.value}. "
                f"Available mappings: {list(OLLAMA_MODEL_MAPPING.keys())}"
            )

        # Check if model is available
        available_models = self._ollama_client.list_models()
        # Check for exact match or partial match (ollama uses tags like "qwen2.5:7b")
        model_available = any(
            ollama_model in m or m.startswith(ollama_model.split(":")[0])
            for m in available_models
        )

        if not model_available:
            logger.warning(
                f"Ollama model '{ollama_model}' not found locally. "
                f"Available: {available_models}. "
                f"Pull with: ollama pull {ollama_model}"
            )
            raise RuntimeError(
                f"Ollama model '{ollama_model}' not available. "
                f"Pull with: ollama pull {ollama_model}"
            )

        logger.info(f"Ollama ready with model: {ollama_model}")
        self._current_ollama_model = ollama_model
        return self._ollama_base_url

    def run(self, output_dir: Path) -> None:
        """Run the full grid evaluation.

        Args:
            output_dir: Directory to write results to
        """
        logger.info("Starting grid evaluation")
        logger.info(self.config.describe())

        # Initialize progress tracker
        progress = ProgressTracker(output_dir / "progress.csv")

        # Load datasets
        self._load_datasets()

        # Track current GPU type and resource config for logging
        current_gpu_type: Optional[GpuType] = None
        current_resource_config: Optional[ResourceConfig] = None

        try:
            # Create output writer
            with JSONLWriter(output_dir) as writer:
                # Iterate through all combinations (5-tuple)
                combo_num = 0
                total_combos = self.config.total_combinations()

                for gpu_type, resource_config, agent, model, benchmark in self.config.get_all_combinations():
                    combo_num += 1

                    # Track hardware changes (outermost loop)
                    if gpu_type != current_gpu_type:
                        logger.info(f"=== GPU Type: {gpu_type.value} ===")
                        current_gpu_type = gpu_type

                    # Track resource config changes (2nd loop)
                    if resource_config != current_resource_config:
                        logger.info(f"  Resource Config: {resource_config.value}")
                        current_resource_config = resource_config

                    combo_desc = (
                        f"{gpu_type.value}/{resource_config.value}/"
                        f"{agent.value}/{model.value}/{benchmark.value}"
                    )

                    # Check if already completed
                    if progress.is_completed(
                        gpu_type.value,
                        resource_config.value,
                        agent.value,
                        model.value,
                        benchmark.value,
                    ):
                        logger.info(
                            f"[{combo_num}/{total_combos}] "
                            f"Skipping {combo_desc} - already completed"
                        )
                        continue

                    # Check hardware constraints
                    skip_reason = self._should_skip_combination(model, resource_config)
                    if skip_reason:
                        logger.info(
                            f"[{combo_num}/{total_combos}] "
                            f"Skipping {combo_desc} - {skip_reason}"
                        )
                        progress.mark_skipped(
                            gpu_type.value,
                            resource_config.value,
                            agent.value,
                            model.value,
                            benchmark.value,
                            skip_reason,
                        )
                        continue

                    logger.info(
                        f"[{combo_num}/{total_combos}] Running {combo_desc}"
                    )

                    model_config = MODEL_REGISTRY[model]

                    # Determine GPU vendor and backend
                    gpu_config = GPU_TYPE_REGISTRY.get(gpu_type, {})
                    gpu_vendor = gpu_config.get("vendor", "nvidia")
                    backend = self._get_backend_for_gpu(gpu_type)

                    try:
                        # Apply resource config FIRST (sets CUDA_VISIBLE_DEVICES)
                        # so vLLM server launches on the correct GPUs
                        with ResourceManager(resource_config, gpu_vendor=gpu_vendor):
                            # Ensure inference server is ready based on backend
                            if backend == "ollama":
                                base_url = self._ensure_ollama_ready(model, model_config)
                            else:
                                # Restart vLLM if needed for this resource/model config
                                base_url = self._ensure_vllm_server(resource_config, model_config)
                                # Update the base URL for agent creation
                                self.vllm_base_url = base_url

                            # Get samples for this benchmark
                            samples = self._get_samples(benchmark)

                            # Start energy monitor to track GPUs
                            collector = EnergyMonitorCollector()
                            with collector.start():
                                self._telemetry_session = TelemetrySession(collector, buffer_seconds=600.0)
                                self._run_combination(
                                    writer=writer,
                                    benchmark=benchmark,
                                    model=model,
                                    agent=agent,
                                    gpu_type=gpu_type,
                                    resource_config=resource_config,
                                    samples=samples,
                                    model_config=model_config,
                                    backend=backend,
                                )

                        # Mark as completed
                        progress.mark_completed(
                            gpu_type.value,
                            resource_config.value,
                            agent.value,
                            model.value,
                            benchmark.value,
                            len(samples),
                        )
                    except Exception as e:
                        logger.error(f"Failed {combo_desc}: {e}")
                        progress.mark_failed(
                            gpu_type.value,
                            resource_config.value,
                            agent.value,
                            model.value,
                            benchmark.value,
                            str(e),
                        )

                # Write summary and metadata
                metadata = GridMetadata(
                    gpu_types=[g.value for g in self.config.gpu_types],
                    resource_configs=[r.value for r in self.config.resource_configs],
                    benchmarks=[b.value for b in self.config.benchmarks],
                    models=[m.value for m in self.config.models],
                    agents=[a.value for a in self.config.agents],
                    # Legacy field for backwards compatibility
                    hardware_configs=[
                        f"{g.value}/{r.value}"
                        for g in self.config.gpu_types
                        for r in self.config.resource_configs
                    ],
                    queries_per_benchmark=self.config.queries_per_benchmark,
                    seed=self.config.seed,
                    timestamp=datetime.now().isoformat(),
                    total_combinations=self.config.total_combinations(),
                    total_queries=self.config.total_queries(),
                )
                writer.finalize(metadata)

                # Log progress summary
                summary = progress.get_summary()
                logger.info(
                    f"Progress: {summary['completed']} completed, "
                    f"{summary['skipped']} skipped, {summary['failed']} failed"
                )
        finally:
            # Cleanup vLLM server
            if self._vllm_manager:
                logger.info("Stopping vLLM server")
                self._vllm_manager.stop_all()
                self._vllm_manager = None

        logger.info(f"Grid evaluation complete. Results written to {output_dir}")

    def _load_datasets(self) -> None:
        """Load datasets with optional sampling.

        If use_full_datasets is True, loads all samples.
        Otherwise, applies queries_per_benchmark limit.
        """
        random.seed(self.config.seed)

        # Load HLE dataset
        if BenchmarkType.HLE in self.config.benchmarks:
            logger.info("Loading HLE dataset...")
            from evals.benchmarks.hle.dataset import load_hle_dataset

            # Apply text_only filter from config
            all_hle = load_hle_dataset(
                split="test",  # HLE dataset uses test split
                text_only=self.config.hle_text_only,
            )
            logger.info(
                f"Loaded {len(all_hle)} HLE samples "
                f"(text_only={self.config.hle_text_only})"
            )

            # Apply sampling if not using full datasets
            if self.config.use_full_datasets:
                self._hle_samples = all_hle
            elif len(all_hle) > self.config.queries_per_benchmark:
                self._hle_samples = random.sample(
                    all_hle, self.config.queries_per_benchmark
                )
            else:
                self._hle_samples = all_hle
            logger.info(f"Using {len(self._hle_samples)} HLE samples for evaluation")

        # Load GAIA dataset
        if BenchmarkType.GAIA in self.config.benchmarks:
            logger.info("Loading GAIA dataset...")
            from evals.benchmarks.gaia.dataset import load_gaia_samples

            # Determine subset based on gaia_level filter
            if self.config.gaia_level is not None:
                subset = f"2023_level{self.config.gaia_level}"
                logger.info(f"Filtering GAIA to level {self.config.gaia_level}")
            else:
                subset = "2023_all"

            all_gaia = list(
                load_gaia_samples(
                    subset=subset,
                    split="validation",
                    shuffle=True,
                    seed=self.config.seed,
                )
            )
            logger.info(f"Loaded {len(all_gaia)} GAIA samples")

            # Apply sampling if not using full datasets
            if self.config.use_full_datasets:
                self._gaia_samples = all_gaia
            elif len(all_gaia) > self.config.queries_per_benchmark:
                self._gaia_samples = all_gaia[: self.config.queries_per_benchmark]
            else:
                self._gaia_samples = all_gaia
            logger.info(f"Using {len(self._gaia_samples)} GAIA samples for evaluation")

        # Load SWEBENCH dataset
        if BenchmarkType.SWEBENCH in self.config.benchmarks:
            logger.info("Loading SWEBENCH dataset...")
            from evals.benchmarks.swebench.dataset import load_swebench_samples

            all_swebench = list(load_swebench_samples(shuffle=True, seed=self.config.seed))
            logger.info(f"Loaded {len(all_swebench)} SWEBENCH samples")

            if self.config.use_full_datasets:
                self._swebench_samples = all_swebench
            elif len(all_swebench) > self.config.queries_per_benchmark:
                self._swebench_samples = all_swebench[: self.config.queries_per_benchmark]
            else:
                self._swebench_samples = all_swebench
            logger.info(f"Using {len(self._swebench_samples)} SWEBENCH samples for evaluation")

        # Load APEX dataset
        if BenchmarkType.APEX in self.config.benchmarks:
            logger.info("Loading APEX dataset...")
            from evals.benchmarks.apex.dataset import load_apex_samples

            all_apex = list(load_apex_samples(shuffle=True, seed=self.config.seed))
            logger.info(f"Loaded {len(all_apex)} APEX samples")

            if self.config.use_full_datasets:
                self._apex_samples = all_apex
            elif len(all_apex) > self.config.queries_per_benchmark:
                self._apex_samples = all_apex[: self.config.queries_per_benchmark]
            else:
                self._apex_samples = all_apex
            logger.info(f"Using {len(self._apex_samples)} APEX samples for evaluation")

        # Load BROWSECOMP dataset
        if BenchmarkType.BROWSECOMP in self.config.benchmarks:
            logger.info("Loading BROWSECOMP dataset...")
            from evals.benchmarks.browsecomp.dataset import load_browsecomp_samples

            all_browsecomp = load_browsecomp_samples(shuffle=True, seed=self.config.seed)
            logger.info(f"Loaded {len(all_browsecomp)} BROWSECOMP samples")

            if self.config.use_full_datasets:
                self._browsecomp_samples = all_browsecomp
            elif len(all_browsecomp) > self.config.queries_per_benchmark:
                self._browsecomp_samples = all_browsecomp[: self.config.queries_per_benchmark]
            else:
                self._browsecomp_samples = all_browsecomp
            logger.info(f"Using {len(self._browsecomp_samples)} BROWSECOMP samples for evaluation")

        # Load DEEPRESEARCH dataset
        if BenchmarkType.DEEPRESEARCH in self.config.benchmarks:
            logger.info("Loading DEEPRESEARCH dataset...")
            from evals.benchmarks.deepresearch.dataset import load_deepresearch_samples

            all_deepresearch = list(load_deepresearch_samples(shuffle=True, seed=self.config.seed))
            logger.info(f"Loaded {len(all_deepresearch)} DEEPRESEARCH samples")

            if self.config.use_full_datasets:
                self._deepresearch_samples = all_deepresearch
            elif len(all_deepresearch) > self.config.queries_per_benchmark:
                self._deepresearch_samples = all_deepresearch[: self.config.queries_per_benchmark]
            else:
                self._deepresearch_samples = all_deepresearch
            logger.info(f"Using {len(self._deepresearch_samples)} DEEPRESEARCH samples for evaluation")

        # Load SIMPLEQA dataset
        if BenchmarkType.SIMPLEQA in self.config.benchmarks:
            logger.info("Loading SIMPLEQA dataset...")
            from evals.benchmarks.simpleqa.dataset import load_simpleqa_samples

            all_simpleqa = list(load_simpleqa_samples(shuffle=True, seed=self.config.seed))
            logger.info(f"Loaded {len(all_simpleqa)} SIMPLEQA samples")

            if self.config.use_full_datasets:
                self._simpleqa_samples = all_simpleqa
            elif len(all_simpleqa) > self.config.queries_per_benchmark:
                self._simpleqa_samples = all_simpleqa[: self.config.queries_per_benchmark]
            else:
                self._simpleqa_samples = all_simpleqa
            logger.info(f"Using {len(self._simpleqa_samples)} SIMPLEQA samples for evaluation")

        # Load SWEFFICIENCY dataset
        if BenchmarkType.SWEFFICIENCY in self.config.benchmarks:
            logger.info("Loading SWEFFICIENCY dataset...")
            from evals.benchmarks.swefficiency.dataset import load_swefficiency_samples

            all_swefficiency = list(load_swefficiency_samples(shuffle=True, seed=self.config.seed))
            logger.info(f"Loaded {len(all_swefficiency)} SWEFFICIENCY samples")

            if self.config.use_full_datasets:
                self._swefficiency_samples = all_swefficiency
            elif len(all_swefficiency) > self.config.queries_per_benchmark:
                self._swefficiency_samples = all_swefficiency[: self.config.queries_per_benchmark]
            else:
                self._swefficiency_samples = all_swefficiency
            logger.info(f"Using {len(self._swefficiency_samples)} SWEFFICIENCY samples for evaluation")

    def _get_samples(self, benchmark: BenchmarkType) -> List[Any]:
        """Get samples for a benchmark."""
        if benchmark == BenchmarkType.HLE:
            return self._hle_samples or []
        elif benchmark == BenchmarkType.GAIA:
            return self._gaia_samples or []
        elif benchmark == BenchmarkType.SWEBENCH:
            return self._swebench_samples or []
        elif benchmark == BenchmarkType.APEX:
            return self._apex_samples or []
        elif benchmark == BenchmarkType.BROWSECOMP:
            return self._browsecomp_samples or []
        elif benchmark == BenchmarkType.DEEPRESEARCH:
            return self._deepresearch_samples or []
        elif benchmark == BenchmarkType.SIMPLEQA:
            return self._simpleqa_samples or []
        elif benchmark == BenchmarkType.SWEFFICIENCY:
            return self._swefficiency_samples or []
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    @staticmethod
    def _chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _run_combination(
        self,
        writer: JSONLWriter,
        benchmark: BenchmarkType,
        model: ModelType,
        agent: AgentType,
        gpu_type: GpuType,
        resource_config: ResourceConfig,
        samples: List[Any],
        model_config: Dict[str, Any],
        backend: str = "vllm",
    ) -> None:
        """Run evaluation for a single configuration combination.

        Args:
            writer: JSONL output writer
            benchmark: Benchmark type
            model: Model type
            agent: Agent type
            gpu_type: GPU hardware type
            resource_config: Resource configuration
            samples: Benchmark samples to evaluate
            model_config: Model configuration dict
            backend: Inference backend ("vllm" or "ollama")
        """
        if self.config.batch_mode:
            batch_size = model_config.get("max_batch_size", 1)
            if batch_size <= 1:
                self._run_combination_sequential(
                    writer=writer,
                    benchmark=benchmark,
                    model=model,
                    agent=agent,
                    gpu_type=gpu_type,
                    resource_config=resource_config,
                    samples=samples,
                    model_config=model_config,
                    backend=backend,
                )
            else:
                self._run_combination_batched(
                    writer=writer,
                    benchmark=benchmark,
                    model=model,
                    agent=agent,
                    gpu_type=gpu_type,
                    resource_config=resource_config,
                    samples=samples,
                    model_config=model_config,
                    backend=backend,
                    batch_size=batch_size,
                )
        elif self.config.num_workers > 1:
            self._run_combination_parallel(
                writer=writer,
                benchmark=benchmark,
                model=model,
                agent=agent,
                gpu_type=gpu_type,
                resource_config=resource_config,
                samples=samples,
                model_config=model_config,
                backend=backend,
                num_workers=self.config.num_workers,
            )
        else:
            self._run_combination_sequential(
                writer=writer,
                benchmark=benchmark,
                model=model,
                agent=agent,
                gpu_type=gpu_type,
                resource_config=resource_config,
                samples=samples,
                model_config=model_config,
                backend=backend,
            )

    def _run_combination_sequential(
        self,
        writer: JSONLWriter,
        benchmark: BenchmarkType,
        model: ModelType,
        agent: AgentType,
        gpu_type: GpuType,
        resource_config: ResourceConfig,
        samples: List[Any],
        model_config: Dict[str, Any],
        backend: str = "vllm",
    ) -> None:
        """Run evaluation sequentially with per-query energy attribution."""
        from ipw.telemetry.correlation import (
            _safe_delta,
            compute_analysis,
            correlate_energy_to_events,
        )
        from ipw.telemetry.events import EventRecorder

        # Create agent instance
        agent_instance, recorder = self._create_agent(
            model, agent, benchmark=benchmark, backend=backend
        )

        for idx, sample in enumerate(samples):
            logger.info(
                f"  Query {idx + 1}/{len(samples)}: {self._get_sample_id(sample, benchmark)}"
            )

            # Clear recorder for new query
            recorder.clear()

            # Get query text and ground truth
            query_text = self._get_query_text(sample, benchmark)
            ground_truth = self._get_ground_truth(sample, benchmark)

            # Run evaluation with telemetry collection
            error = None
            response = ""
            run_result = None

            # Use telemetry session to collect energy samples during query execution
            with self._telemetry_session:
                start_time = time.time()
                try:
                    run_result = agent_instance.run(query_text)
                    # Convert response to string, preserving RunResult separately
                    if hasattr(run_result, "content"):
                        response = run_result.content
                    elif isinstance(run_result, dict):
                        response = run_result.get("content", str(run_result))
                    else:
                        response = run_result
                    if isinstance(response, list):
                        response = "\n".join(str(item) for item in response)
                    response = str(response)
                except Exception as e:
                    logger.error(f"Error running query: {e}")
                    error = str(e)

                end_time = time.time()
                # Get energy samples collected during this query
                energy_samples = list(self._telemetry_session.window(start_time, end_time))

            latency = end_time - start_time

            # --- Direct energy computation from window samples (Issue 1 fix) ---
            # This is the primary path: compute energy directly from the first/last
            # samples in the window, independent of event correlation.
            total_joules = 0.0  # GPU energy
            cpu_joules = 0.0   # CPU energy (RAPL)
            max_power = 0.0

            if energy_samples:
                first_sample = energy_samples[0]
                last_sample = energy_samples[-1]
                total_joules = _safe_delta(
                    getattr(last_sample.reading, "energy_joules", None),
                    getattr(first_sample.reading, "energy_joules", None),
                )
                cpu_joules = _safe_delta(
                    getattr(last_sample.reading, "cpu_energy_joules", None),
                    getattr(first_sample.reading, "cpu_energy_joules", None),
                )
                # Compute max and avg power from all samples in window
                power_readings = [
                    s.reading.power_watts
                    for s in energy_samples
                    if s.reading.power_watts is not None
                ]
                if power_readings:
                    max_power = max(power_readings)

                logger.debug(
                    f"  Energy window: {len(energy_samples)} samples, "
                    f"{total_joules:.2f}J, max {max_power:.1f}W "
                    f"(ts range: {first_sample.timestamp:.3f}-{last_sample.timestamp:.3f}, "
                    f"query range: {start_time:.3f}-{end_time:.3f})"
                )
            else:
                # Diagnostic: log why we have no samples
                ts_info = ""
                if self._telemetry_session is not None:
                    ts_info = (
                        f"telemetry_ready={self._telemetry_session.is_ready}, "
                        f"buffered_samples={self._telemetry_session.sample_count}, "
                        f"last_ts={self._telemetry_session.last_sample_timestamp}"
                    )
                logger.warning(
                    f"  No energy samples in window [{start_time:.3f}, {end_time:.3f}] "
                    f"({ts_info})"
                )

            # Event correlation for per-action breakdowns (enrichment, not primary)
            events = recorder.get_events()
            logger.info(f"  Correlation: {len(energy_samples)} samples, {len(events)} events")
            if events:
                logger.info(f"  Event types: {[e.event_type for e in events]}")
            breakdowns = correlate_energy_to_events(energy_samples, events)
            logger.info(f"  Breakdowns: {len(breakdowns)} items")
            analysis = compute_analysis(breakdowns)

            # Serialize per-action breakdowns
            action_breakdowns = [asdict(b) for b in breakdowns] if breakdowns else None
            energy_by_action = analysis.get("energy_by_action") or None

            # --- Extract turns/tools from RunResult when available (Issue 2 fix) ---
            if run_result is not None and hasattr(run_result, "num_turns"):
                turns = run_result.num_turns
                tools_used = (
                    dict(Counter(run_result.tool_names_used))
                    if hasattr(run_result, "tool_names_used")
                    else {}
                )
                models_called = {
                    getattr(run_result, "model_id", model_config["model_id"]): turns
                }
            else:
                # Fallback to analysis (for agents that don't return RunResult)
                tools_used = self._extract_tool_counts(analysis)
                turns = analysis.get("action_counts", {}).get("lm_inference", 0)
                models_called = analysis.get("model_counts", {})

            # Score correctness using LLM judge or exact match
            sample_id = self._get_sample_id(sample, benchmark)
            is_correct, grading_meta = self._score_response(
                response=response,
                ground_truth=ground_truth,
                benchmark=benchmark,
                question=query_text,
                sample=sample,
                query_id=sample_id,
            )
            # Write grading audit record
            self._write_grading_record(grading_meta, writer.output_dir)

            # Combined hardware string for backwards compatibility
            hardware = f"{gpu_type.value}/{resource_config.value}"

            # Write result
            result = QueryResult(
                query_id=sample_id,
                benchmark=benchmark.value,
                model=model.value,
                agent=agent.value,
                gpu_type=gpu_type.value,
                resource_config=resource_config.value,
                hardware=hardware,
                avg_joules=total_joules,
                gpu_joules=total_joules,
                cpu_joules=cpu_joules,
                max_power_watts=max_power,
                latency_seconds=latency,
                tools_used=tools_used,
                turns=turns,
                models_called=models_called,
                is_correct=is_correct,
                response=response,
                ground_truth=ground_truth,
                error=error,
                grade=grading_meta.get("grade"),
                total_params_b=model_config.get("total_params_b"),
                active_params_b=model_config.get("active_params_b"),
                action_breakdowns=action_breakdowns,
                energy_by_action=energy_by_action,
            )
            writer.write_query_result(result)

    def _run_combination_batched(
        self,
        writer: JSONLWriter,
        benchmark: BenchmarkType,
        model: ModelType,
        agent: AgentType,
        gpu_type: GpuType,
        resource_config: ResourceConfig,
        samples: List[Any],
        model_config: Dict[str, Any],
        backend: str = "vllm",
        batch_size: int = 1,
    ) -> None:
        """Run evaluation in batches with amortized energy attribution.

        Each batch of N queries runs concurrently via ThreadPoolExecutor.
        Energy is measured for the entire batch, then amortized (total / N).
        Per-query latency, accuracy, turns, and tool calls are tracked individually.
        """
        from ipw.telemetry.correlation import _safe_delta

        total_samples = len(samples)
        logger.info(
            f"  Running {total_samples} queries in batches of {batch_size} "
            f"(batch mode, amortized energy)"
        )

        hardware = f"{gpu_type.value}/{resource_config.value}"
        writer_lock = Lock()
        global_completed = [0]

        for batch_idx, batch in enumerate(self._chunks(samples, batch_size)):
            actual_batch_size = len(batch)
            logger.info(
                f"  Batch {batch_idx + 1} "
                f"({global_completed[0] + 1}-{global_completed[0] + actual_batch_size}"
                f"/{total_samples}): {actual_batch_size} queries"
            )

            # Each slot holds (result_kwargs, error) or None if the worker failed
            batch_results: List[Optional[Dict[str, Any]]] = [None] * actual_batch_size

            def process_batch_item(slot: int, sample: Any) -> None:
                """Process a single sample within a batch."""
                agent_instance, recorder = self._create_agent(
                    model, agent, benchmark=benchmark, backend=backend
                )

                sample_id = self._get_sample_id(sample, benchmark)
                query_text = self._get_query_text(sample, benchmark)
                ground_truth = self._get_ground_truth(sample, benchmark)

                error = None
                response = ""
                tools_used: Dict[str, int] = {}
                turns = 0
                models_called: Dict[str, int] = {}

                start_time = time.time()
                try:
                    raw_response = agent_instance.run(query_text)

                    # Extract tool usage from TextToolAgent RunResult
                    if hasattr(raw_response, "tool_names_used"):
                        tools_used = dict(Counter(raw_response.tool_names_used))
                        turns = raw_response.num_turns

                    # Derive models_called
                    models_called = {model_config["model_id"]: turns}
                    cloud_prefixes = ("openai:", "anthropic:", "openrouter:", "gemini:")
                    for tool_name, count in tools_used.items():
                        if any(tool_name.startswith(p) for p in cloud_prefixes):
                            models_called[tool_name] = count

                    # Convert response to string
                    if hasattr(raw_response, "content"):
                        response = raw_response.content
                    elif isinstance(raw_response, dict):
                        response = raw_response.get("content", str(raw_response))
                    else:
                        response = raw_response
                    if isinstance(response, list):
                        response = "\n".join(str(item) for item in response)
                    response = str(response)
                except Exception as e:
                    logger.error(f"Error running query {sample_id}: {e}")
                    error = str(e)

                end_time = time.time()
                latency = end_time - start_time

                # Score correctness
                is_correct, grading_meta = self._score_response(
                    response=response,
                    ground_truth=ground_truth,
                    benchmark=benchmark,
                    question=query_text,
                    sample=sample,
                    query_id=sample_id,
                )
                self._write_grading_record(grading_meta, writer.output_dir)

                batch_results[slot] = {
                    "query_id": sample_id,
                    "benchmark": benchmark.value,
                    "model": model.value,
                    "agent": agent.value,
                    "gpu_type": gpu_type.value,
                    "resource_config": resource_config.value,
                    "hardware": hardware,
                    "latency_seconds": latency,
                    "tools_used": tools_used,
                    "turns": turns,
                    "models_called": models_called,
                    "is_correct": is_correct,
                    "response": response,
                    "ground_truth": ground_truth,
                    "error": error,
                    "grade": grading_meta.get("grade"),
                    "total_params_b": model_config.get("total_params_b"),
                    "active_params_b": model_config.get("active_params_b"),
                }

            # Run the batch with telemetry collection
            with self._telemetry_session:
                batch_start = time.time()

                with ThreadPoolExecutor(max_workers=actual_batch_size) as executor:
                    futures = {
                        executor.submit(process_batch_item, slot, sample): slot
                        for slot, sample in enumerate(batch)
                    }
                    for future in as_completed(futures):
                        slot = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Worker failed for batch slot {slot}: {e}")

                batch_end = time.time()
                energy_samples = list(
                    self._telemetry_session.window(batch_start, batch_end)
                )

            # Compute amortized energy
            completed_results = [r for r in batch_results if r is not None]
            completed_count = len(completed_results)

            total_joules = 0.0
            cpu_joules = 0.0
            max_power = 0.0

            if energy_samples:
                first_sample = energy_samples[0]
                last_sample = energy_samples[-1]
                total_joules = _safe_delta(
                    getattr(last_sample.reading, "energy_joules", None),
                    getattr(first_sample.reading, "energy_joules", None),
                )
                cpu_joules = _safe_delta(
                    getattr(last_sample.reading, "cpu_energy_joules", None),
                    getattr(first_sample.reading, "cpu_energy_joules", None),
                )
                power_readings = [
                    s.reading.power_watts
                    for s in energy_samples
                    if s.reading.power_watts is not None
                ]
                if power_readings:
                    max_power = max(power_readings)

                logger.info(
                    f"  Batch energy: {total_joules:.2f}J GPU, {cpu_joules:.2f}J CPU, "
                    f"max {max_power:.1f}W, {len(energy_samples)} samples"
                )
            else:
                logger.warning(
                    f"  No energy samples for batch "
                    f"[{batch_start:.3f}, {batch_end:.3f}]"
                )

            # Amortize energy across completed queries
            if completed_count > 0:
                amortized_gpu = total_joules / completed_count
                amortized_cpu = cpu_joules / completed_count
            else:
                amortized_gpu = 0.0
                amortized_cpu = 0.0

            # Write results for all completed queries in this batch
            for r in completed_results:
                result = QueryResult(
                    query_id=r["query_id"],
                    benchmark=r["benchmark"],
                    model=r["model"],
                    agent=r["agent"],
                    gpu_type=r["gpu_type"],
                    resource_config=r["resource_config"],
                    hardware=r["hardware"],
                    avg_joules=amortized_gpu,
                    gpu_joules=amortized_gpu,
                    cpu_joules=amortized_cpu,
                    max_power_watts=max_power,
                    latency_seconds=r["latency_seconds"],
                    tools_used=r["tools_used"],
                    turns=r["turns"],
                    models_called=r["models_called"],
                    is_correct=r["is_correct"],
                    response=r["response"],
                    ground_truth=r["ground_truth"],
                    error=r["error"],
                    grade=r["grade"],
                    total_params_b=r["total_params_b"],
                    active_params_b=r["active_params_b"],
                    action_breakdowns=None,
                    energy_by_action=None,
                    batch_size=batch_size,
                    concurrency=actual_batch_size,
                    energy_amortized=True,
                )
                writer.write_query_result(result)

            global_completed[0] += actual_batch_size
            logger.info(
                f"  Batch {batch_idx + 1} done: {completed_count}/{actual_batch_size} succeeded, "
                f"amortized energy: {amortized_gpu:.2f}J/query"
            )

    def _run_combination_parallel(
        self,
        writer: JSONLWriter,
        benchmark: BenchmarkType,
        model: ModelType,
        agent: AgentType,
        gpu_type: GpuType,
        resource_config: ResourceConfig,
        samples: List[Any],
        model_config: Dict[str, Any],
        backend: str = "vllm",
        num_workers: int = 4,
    ) -> None:
        """Run evaluation in parallel using ThreadPoolExecutor.

        Note: Per-query energy attribution is disabled in parallel mode.
        Energy metrics will be aggregated across all concurrent queries.
        """
        logger.info(f"  Running {len(samples)} queries with {num_workers} parallel workers")

        # Thread-safe writer lock
        writer_lock = Lock()
        completed_count = [0]  # Mutable counter for progress tracking

        def process_sample(idx_sample: Tuple[int, Any]) -> None:
            """Process a single sample in a worker thread."""
            idx, sample = idx_sample

            # Each thread needs its own agent instance for thread safety
            agent_instance, recorder = self._create_agent(
                model, agent, benchmark=benchmark, backend=backend
            )

            sample_id = self._get_sample_id(sample, benchmark)
            query_text = self._get_query_text(sample, benchmark)
            ground_truth = self._get_ground_truth(sample, benchmark)

            # Run evaluation (no per-query energy in parallel mode)
            error = None
            response = ""
            start_time = time.time()

            tools_used = {}
            turns = 0
            models_called = {}

            try:
                raw_response = agent_instance.run(query_text)

                # Extract tool usage from TextToolAgent RunResult
                if hasattr(raw_response, "tool_names_used"):
                    from collections import Counter
                    tools_used = dict(Counter(raw_response.tool_names_used))
                    turns = raw_response.num_turns

                # Derive models_called: the local model + any cloud LLM delegations
                models_called = {model_config["model_id"]: turns}
                cloud_prefixes = ("openai:", "anthropic:", "openrouter:", "gemini:")
                for tool_name, count in tools_used.items():
                    if any(tool_name.startswith(p) for p in cloud_prefixes):
                        models_called[tool_name] = count

                # Convert response to string
                if hasattr(raw_response, "content"):
                    response = raw_response.content
                elif isinstance(raw_response, dict):
                    response = raw_response.get("content", str(raw_response))
                else:
                    response = raw_response
                if isinstance(response, list):
                    response = "\n".join(str(item) for item in response)
                response = str(response)
            except Exception as e:
                logger.error(f"Error running query {sample_id}: {e}")
                error = str(e)

            end_time = time.time()
            latency = end_time - start_time

            # Score correctness using LLM judge or exact match
            is_correct, grading_meta = self._score_response(
                response=response,
                ground_truth=ground_truth,
                benchmark=benchmark,
                question=query_text,
                sample=sample,
                query_id=sample_id,
            )
            # Write grading audit record (thread-safe via append mode)
            self._write_grading_record(grading_meta, writer.output_dir)

            # Combined hardware string for backwards compatibility
            hardware = f"{gpu_type.value}/{resource_config.value}"

            # Create result (no per-query energy metrics in parallel mode)
            result = QueryResult(
                query_id=sample_id,
                benchmark=benchmark.value,
                model=model.value,
                agent=agent.value,
                gpu_type=gpu_type.value,
                resource_config=resource_config.value,
                hardware=hardware,
                avg_joules=0.0,  # Not available in parallel mode
                gpu_joules=0.0,  # Not available in parallel mode
                cpu_joules=0.0,  # Not available in parallel mode
                max_power_watts=0.0,  # Not available in parallel mode
                latency_seconds=latency,
                tools_used=tools_used,
                turns=turns,
                models_called=models_called,
                is_correct=is_correct,
                response=response,
                ground_truth=ground_truth,
                error=error,
                grade=grading_meta.get("grade"),
                total_params_b=model_config.get("total_params_b"),
                active_params_b=model_config.get("active_params_b"),
                action_breakdowns=None,
                energy_by_action=None,
            )

            # Thread-safe write
            with writer_lock:
                writer.write_query_result(result)
                completed_count[0] += 1
                if completed_count[0] % 10 == 0 or completed_count[0] == len(samples):
                    logger.info(f"  Progress: {completed_count[0]}/{len(samples)} queries completed")

        # Submit all tasks to thread pool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, (idx, sample)): idx
                for idx, sample in enumerate(samples)
            }

            # Wait for all to complete and handle exceptions
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker failed for sample index {idx}: {e}")

    def _create_agent(
        self,
        model: ModelType,
        agent_type: AgentType,
        benchmark: Optional[BenchmarkType] = None,
        backend: str = "vllm",
    ) -> Tuple[Any, "EventRecorder"]:
        """Create agent instance with event recorder and MCP tools.

        Args:
            model: Model type
            agent_type: Agent type
            benchmark: Optional benchmark type to enable benchmark-specific tools
            backend: Inference backend ("vllm" or "ollama")

        Returns:
            Tuple of (agent_instance, event_recorder)
        """
        from ipw.telemetry.events import EventRecorder

        recorder = EventRecorder()
        model_config = MODEL_REGISTRY[model]

        # Get MCP tools as dict for TextToolAgent
        mcp_tools = self._get_mcp_tools_dict(benchmark=benchmark)

        # Benchmark-specific max_turns: BrowseComp needs more turns for deep web search
        if benchmark == BenchmarkType.BROWSECOMP:
            max_turns = 30
        else:
            max_turns = 10

        # Context window management: leave room for response + buffer
        max_model_len = model_config.get("max_model_len", 32768)
        max_context_tokens = max_model_len - 4096 - 512  # response + buffer

        # Create agent with appropriate model type
        if agent_type == AgentType.REACT:
            # Use TextToolAgent for vLLM backend to avoid native tool calling issues
            # vLLM's --enable-auto-tool-choice requires model-specific parsers and
            # gives 0% accuracy on many models. TextToolAgent uses plain text
            # THOUGHT/TOOL/INPUT format with regex parsing instead.
            if backend == "vllm":
                from grid_eval.tool_agent import TextToolAgent

                agent = TextToolAgent(
                    model_id=model_config["model_id"],
                    vllm_base_url=self.vllm_base_url,
                    mcp_tools=mcp_tools,
                    event_recorder=recorder,
                    max_turns=max_turns,
                    temperature=0.0,
                    max_context_tokens=max_context_tokens,
                )
            else:
                # For non-vLLM backends (e.g., Ollama), use Agno React agent
                model_instance = self._create_agno_model(model_config, backend=backend)
                tools = self._get_agent_tools(benchmark=benchmark)
                from agents.agents.react import React

                agent = React(
                    model=model_instance,
                    tools=tools,
                    event_recorder=recorder,
                )
        elif agent_type == AgentType.OPENHANDS:
            model_instance = self._create_openhands_llm(model_config, backend=backend)
            from agents.agents.openhands import OpenHands

            agent = OpenHands(
                model=model_instance,
                mcp_tools=mcp_tools,
                event_recorder=recorder,
            )
        elif agent_type == AgentType.ORCHESTRATOR:
            # With a trained checkpoint, use full Orchestrator with learned policy.
            # Without one, use TextToolAgent so the LLM selects tools via text
            # generation (same approach the trajectory generator uses for training
            # data collection) instead of the broken HeuristicPolicy.
            checkpoint_path = None  # TODO: support passing checkpoint via config
            if checkpoint_path:
                from agents.agents.orchestrator import Orchestrator

                agent = Orchestrator(
                    checkpoint_path=checkpoint_path,
                    max_turns=max_turns,
                    ollama_base_url=self._ollama_base_url,
                    openai_api_key=None,  # Uses env var
                    event_recorder=recorder,
                )
            else:
                from grid_eval.tool_agent import TextToolAgent

                agent = TextToolAgent(
                    model_id=model_config["model_id"],
                    vllm_base_url=self.vllm_base_url,
                    mcp_tools=mcp_tools,
                    event_recorder=recorder,
                    max_turns=max_turns,
                    temperature=0.7,
                    max_context_tokens=max_context_tokens,
                )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent, recorder

    def _get_agent_tools(self, benchmark: Optional[BenchmarkType] = None) -> List[Callable]:
        """Get MCP tools for agents from the tool registry.

        Args:
            benchmark: Optional benchmark type (unused, kept for API compatibility)

        Returns:
            List of callable tool functions for the agent.
        """
        try:
            from agents.mcp.tool_registry import get_registry

            registry = get_registry(
                vllm_base_url=self.vllm_base_url,
                telemetry_collector=None,  # Telemetry handled separately
            )

            tools = []

            # Add utility tools
            for tool_name in AGENT_TOOLS["utility"]:
                tool_instance = registry.get_tool_instance(tool_name)
                if tool_instance:
                    # Wrap MCP server as callable for React agent
                    tools.append(self._wrap_mcp_tool(tool_instance, tool_name))
                else:
                    logger.warning(f"Tool '{tool_name}' not available")

            # Add cloud LLM tools only if explicitly enabled (avoids contamination)
            if self.config.include_cloud_tools:
                for tool_name in AGENT_TOOLS["cloud_llms"]:
                    tool_instance = registry.get_tool_instance(tool_name)
                    if tool_instance:
                        tools.append(self._wrap_mcp_tool(tool_instance, tool_name))
                    else:
                        logger.debug(f"Cloud tool '{tool_name}' not available (API key may not be set)")
            else:
                logger.info("Cloud LLM tools excluded (use --include-cloud-tools to enable)")

            logger.info(f"Configured {len(tools)} tools for agent")
            return tools

        except ImportError as e:
            logger.warning(f"Could not import tool registry: {e}")
            return []

    def _get_mcp_tools_dict(self, benchmark: Optional[BenchmarkType] = None) -> Dict[str, Any]:
        """Get MCP tools as a dictionary for TextToolAgent.

        Args:
            benchmark: Optional benchmark type (unused, kept for API compatibility)

        Returns:
            Dict mapping tool name to MCP server instance.
        """
        try:
            from agents.mcp.tool_registry import get_registry

            registry = get_registry(
                vllm_base_url=self.vllm_base_url,
                telemetry_collector=None,  # Telemetry handled separately
            )

            tools: Dict[str, Any] = {}

            # Add utility tools
            for tool_name in AGENT_TOOLS["utility"]:
                tool_instance = registry.get_tool_instance(tool_name)
                if tool_instance:
                    tools[tool_name] = tool_instance
                else:
                    logger.warning(f"Tool '{tool_name}' not available")

            # Add cloud LLM tools only if explicitly enabled (avoids contamination)
            if self.config.include_cloud_tools:
                for tool_name in AGENT_TOOLS["cloud_llms"]:
                    tool_instance = registry.get_tool_instance(tool_name)
                    if tool_instance:
                        tools[tool_name] = tool_instance
                    else:
                        logger.debug(f"Cloud tool '{tool_name}' not available (API key may not be set)")
            else:
                logger.info("Cloud LLM tools excluded (use --include-cloud-tools to enable)")

            logger.info(f"Configured {len(tools)} MCP tools for TextToolAgent")
            return tools

        except ImportError as e:
            logger.warning(f"Could not import tool registry: {e}")
            return {}

    def _wrap_mcp_tool(self, mcp_server: Any, tool_name: str) -> Callable:
        """Wrap an MCP server as a callable function for use with agents.

        Args:
            mcp_server: MCP server instance with execute() method
            tool_name: Name of the tool for the function

        Returns:
            Callable function that invokes the MCP server
        """
        def tool_fn(prompt: str | None = None, **kwargs: Any) -> str:
            """Execute the MCP tool with the given prompt."""
            # Coerce None to empty string for tools like 'think' that may be called without a prompt
            prompt = prompt or ""
            result = mcp_server.execute(prompt, **kwargs)
            return result.content if hasattr(result, 'content') else str(result)

        # Set function name and docstring for agent introspection
        tool_fn.__name__ = tool_name.replace(":", "_").replace("/", "_")
        spec = mcp_server._spec if hasattr(mcp_server, '_spec') else None
        if spec:
            tool_fn.__doc__ = spec.description
        else:
            tool_fn.__doc__ = f"Execute {tool_name} tool"

        return tool_fn

    def _create_agno_model(
        self, model_config: Dict[str, Any], backend: str = "vllm"
    ) -> Any:
        """Create agno model instance for React agent.

        Args:
            model_config: Model configuration dict
            backend: Inference backend ("vllm" or "ollama")

        Returns:
            agno model instance
        """
        try:
            from agno.models.openai import OpenAILike

            if backend == "ollama":
                # Use Ollama's OpenAI-compatible endpoint
                ollama_model = self._current_ollama_model
                if not ollama_model:
                    raise RuntimeError("Ollama model not initialized")

                # Ollama's OpenAI-compatible API is at /v1
                ollama_openai_url = f"{self._ollama_base_url}/v1"
                return OpenAILike(
                    id=ollama_model,
                    base_url=ollama_openai_url,
                    api_key="ollama",  # Ollama doesn't require a real key
                )
            elif model_config["type"] == "vllm":
                # self.vllm_base_url already includes /v1 from InferenceServerManager
                return OpenAILike(
                    id=model_config["model_id"],
                    base_url=self.vllm_base_url,
                )
            else:
                kwargs = {"id": model_config["model_id"]}
                if self.openai_base_url:
                    kwargs["base_url"] = self.openai_base_url
                return OpenAILike(**kwargs)
        except ImportError:
            raise ImportError(
                "agno package required for React agent. "
                "Install with: pip install agno"
            )

    def _create_openhands_llm(
        self, model_config: Dict[str, Any], backend: str = "vllm"
    ) -> Any:
        """Create OpenHands LLM instance.

        Note: litellm (used by OpenHands) requires model format `openai/<model_id>`
        with `base_url` pointing to the full API path including /v1.

        Args:
            model_config: Model configuration dict
            backend: Inference backend ("vllm" or "ollama")

        Returns:
            OpenHands LLM instance
        """
        try:
            from openhands.sdk import LLM

            if backend == "ollama":
                # Use Ollama's OpenAI-compatible endpoint
                ollama_model = self._current_ollama_model
                if not ollama_model:
                    raise RuntimeError("Ollama model not initialized")

                # Ollama's OpenAI-compatible API is at /v1
                ollama_openai_url = f"{self._ollama_base_url}/v1"
                return LLM(
                    model=f"openai/{ollama_model}",
                    base_url=ollama_openai_url,
                    api_key="ollama",  # Ollama doesn't require real key
                )
            elif model_config["type"] == "vllm":
                return LLM(
                    model=f"openai/{model_config['model_id']}",
                    base_url=self.vllm_base_url,
                    api_key="dummy",  # vLLM doesn't require real key
                )
            else:
                kwargs = {"model": model_config["model_id"]}
                if self.openai_base_url:
                    kwargs["base_url"] = self.openai_base_url
                return LLM(**kwargs)
        except ImportError:
            raise ImportError(
                "openhands package required for OpenHands agent. "
                "Install with: pip install openhands-ai"
            )

    def _get_sample_id(self, sample: Any, benchmark: BenchmarkType) -> str:
        """Extract sample ID from a benchmark sample."""
        if benchmark == BenchmarkType.HLE:
            return sample.task_id
        elif benchmark == BenchmarkType.GAIA:
            return sample.task_id
        elif benchmark == BenchmarkType.SWEBENCH:
            return sample.instance_id
        elif benchmark == BenchmarkType.APEX:
            return sample.task_id
        elif benchmark == BenchmarkType.BROWSECOMP:
            return sample.uid
        elif benchmark == BenchmarkType.DEEPRESEARCH:
            return sample.task_id
        elif benchmark == BenchmarkType.SIMPLEQA:
            return f"simpleqa_{sample.original_index}"
        elif benchmark == BenchmarkType.SWEFFICIENCY:
            return sample.instance_id
        return str(hash(str(sample)))

    def _get_query_text(self, sample: Any, benchmark: BenchmarkType) -> str:
        """Extract query text from a benchmark sample."""
        if benchmark == BenchmarkType.HLE:
            return sample.question
        elif benchmark == BenchmarkType.GAIA:
            return sample.get_prompt()
        elif benchmark == BenchmarkType.SWEBENCH:
            return sample.problem_statement
        elif benchmark == BenchmarkType.APEX:
            return sample.get_full_prompt()
        elif benchmark == BenchmarkType.BROWSECOMP:
            from evals.benchmarks.browsecomp.prompts import format_query
            return format_query(sample.question)
        elif benchmark == BenchmarkType.DEEPRESEARCH:
            return sample.get_prompt()
        elif benchmark == BenchmarkType.SIMPLEQA:
            return sample.get_prompt()
        elif benchmark == BenchmarkType.SWEFFICIENCY:
            return sample.get_prompt()
        return str(sample)

    def _get_ground_truth(self, sample: Any, benchmark: BenchmarkType) -> str:
        """Extract ground truth from a benchmark sample."""
        if benchmark == BenchmarkType.HLE:
            return sample.answer
        elif benchmark == BenchmarkType.GAIA:
            return sample.final_answer
        elif benchmark == BenchmarkType.SWEBENCH:
            return sample.patch  # Ground truth patch
        elif benchmark == BenchmarkType.APEX:
            return str(sample.rubric)  # Rubric for grading
        elif benchmark == BenchmarkType.BROWSECOMP:
            return sample.answer
        elif benchmark == BenchmarkType.DEEPRESEARCH:
            return ""  # No single ground truth for research reports
        elif benchmark == BenchmarkType.SIMPLEQA:
            return sample.answer
        elif benchmark == BenchmarkType.SWEFFICIENCY:
            return sample.patch  # Ground truth optimization
        return ""

    def _score_response(
        self,
        response: str,
        ground_truth: str,
        benchmark: BenchmarkType,
        question: str = "",
        sample: Any = None,
        query_id: str = "",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Score response against ground truth using appropriate method.

        Uses the unified scorer interface from grid_eval.scorers.

        Args:
            response: Model's response
            ground_truth: Expected correct answer
            benchmark: Benchmark type
            question: Original question (for LLM judge context)
            sample: Original sample (for benchmark-specific metadata)
            query_id: Query identifier for grading audit trail

        Returns:
            Tuple of (is_correct, grading_metadata)
        """
        grading_meta: Dict[str, Any] = {
            "query_id": query_id,
            "benchmark": benchmark.value,
            "response_preview": (response or "")[:200],
            "ground_truth": ground_truth,
        }

        if not response:
            grading_meta.update(match_type="empty_response", is_correct=False, grade="NOT_ATTEMPTED")
            return False, grading_meta

        # Check if exact match mode is enabled (bypass LLM judge)
        if self.config.use_exact_match:
            pred = response.strip().lower()
            truth = ground_truth.strip().lower()
            is_correct = pred == truth
            grading_meta.update(
                match_type="exact_config",
                is_correct=is_correct,
                grade="CORRECT" if is_correct else "INCORRECT",
            )
            return is_correct, grading_meta

        # SWE-bench/SWEfficiency: Docker-based evaluation (not implemented)
        if benchmark in (BenchmarkType.SWEBENCH, BenchmarkType.SWEFFICIENCY):
            logger.warning(f"SWE-bench evaluation not yet implemented, returning False")
            grading_meta.update(match_type="not_implemented", is_correct=False, grade="ERROR")
            return False, grading_meta

        # Use unified scorer interface
        try:
            from evals.benchmarks.scorer import get_scorer, ScoreResult

            # Get additional kwargs based on benchmark
            kwargs = {}
            if benchmark == BenchmarkType.APEX and sample is not None:
                if hasattr(sample, "rubric"):
                    kwargs["rubric"] = sample.rubric
            elif benchmark == BenchmarkType.DEEPRESEARCH and sample is not None:
                if hasattr(sample, "domain"):
                    kwargs["domain"] = sample.domain

            # Get the appropriate scorer for this benchmark
            scorer = get_scorer(
                benchmark=benchmark,
                model=self.grader_model,
                api_key=self.grader_api_key,
            )

            # Score using the unified interface
            result = scorer.score_sync(
                question=question,
                response=response,
                ground_truth=ground_truth,
                **kwargs,
            )

            # Populate grading metadata from ScorerResult
            grading_meta.update(
                match_type=result.metadata.get("match_type", "unknown"),
                is_correct=result.is_correct,
                grade=result.grade.value if result.grade else None,
                explanation=result.explanation,
                llm_response=result.raw_response,
                error=result.error,
            )

            # Handle errors by falling back to exact match
            if result.grade == ScoreResult.ERROR:
                logger.warning(f"Scorer error for {benchmark.value}: {result.error}")
                is_correct = response.strip().lower() == ground_truth.strip().lower()
                grading_meta.update(
                    match_type="exact_fallback_after_error",
                    is_correct=is_correct,
                    grade="CORRECT" if is_correct else "INCORRECT",
                )
                return is_correct, grading_meta

            return result.is_correct, grading_meta

        except Exception as e:
            logger.warning(f"Scoring failed for {benchmark.value}: {e}, falling back to exact match")
            is_correct = response.strip().lower() == ground_truth.strip().lower()
            grading_meta.update(
                match_type="exact_fallback_after_exception",
                is_correct=is_correct,
                grade="CORRECT" if is_correct else "INCORRECT",
                error=str(e),
            )
            return is_correct, grading_meta

    def _write_grading_record(
        self, grading_meta: Dict[str, Any], output_dir: Path
    ) -> None:
        """Write a grading record to the grading audit JSONL file.

        Args:
            grading_meta: Grading metadata dict
            output_dir: Output directory (grading/ subdir will be created)
        """
        grading_dir = output_dir / "grading"
        grading_dir.mkdir(parents=True, exist_ok=True)
        grading_file = grading_dir / "grading.jsonl"
        try:
            with open(grading_file, "a") as f:
                f.write(json.dumps(grading_meta, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write grading record: {e}")

    def _extract_tool_counts(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Extract tool usage counts from analysis."""
        action_counts = analysis.get("action_counts", {})
        # Filter to just tool calls
        return {
            k: v
            for k, v in action_counts.items()
            if k.startswith("tool_call") or k not in ["lm_inference", "idle"]
        }


__all__ = ["GridEvalRunner"]
