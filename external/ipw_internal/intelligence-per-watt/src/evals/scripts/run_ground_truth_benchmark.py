#!/usr/bin/env python3
"""Ground-truth vLLM benchmark with streaming TTFT/TBT measurement.

Runs real vLLM serving benchmarks across multiple models, configs, workloads,
and QPS levels.  Uses streaming responses to measure TTFT (time to first token)
and TBT (inter-token intervals).  Results are cached per-run so reruns skip
already-completed combinations.

Usage:
    python -m evals.scripts.run_ground_truth_benchmark \
        --output-dir data/ground_truth \
        --models qwen3-0.6b,qwen3-1.7b,qwen3-4b,qwen3-8b \
        --qps-levels 2,10,20 \
        --duration-s 60
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ground_truth_benchmark")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "1gpu-fp16": {"num_gpus": 1, "tensor_parallel": 1, "precision": "fp16"},
    "1gpu-bf16": {"num_gpus": 1, "tensor_parallel": 1, "precision": "bf16"},
    "4gpu-fp16": {"num_gpus": 4, "tensor_parallel": 4, "precision": "fp16"},
    "4gpu-bf16": {"num_gpus": 4, "tensor_parallel": 4, "precision": "bf16"},
    "8gpu-fp16": {"num_gpus": 8, "tensor_parallel": 8, "precision": "fp16"},
    "8gpu-bf16": {"num_gpus": 8, "tensor_parallel": 8, "precision": "bf16"},
}

WORKLOAD_PROFILES: Dict[str, Dict[str, int]] = {
    "chat": {"avg_input_tokens": 1024, "avg_output_tokens": 1024},
    "reasoning": {"avg_input_tokens": 1024, "avg_output_tokens": 2048},
    "rag": {"avg_input_tokens": 2048, "avg_output_tokens": 512},
    "agentic": {"avg_input_tokens": 1024, "avg_output_tokens": 1024},
}

DEFAULT_MODELS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
DEFAULT_QPS_LEVELS = [2, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_specs() -> Dict[str, Any]:
    """Return model specs from the inference_search registry."""
    from inference_search.cli import _EXAMPLE_MODELS
    return _EXAMPLE_MODELS


def _percentiles(values: List[float]) -> Dict[str, float]:
    """Compute p50, p90, p95, p99 percentiles."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _output_path(
    output_dir: Path, model_key: str, config_id: str, workload: str, qps: float,
) -> Path:
    """Build the per-run output path."""
    return output_dir / model_key / config_id / f"{workload}_{int(qps)}.json"


# ---------------------------------------------------------------------------
# Energy collection
# ---------------------------------------------------------------------------


class EnergyCollector:
    """Collects GPU energy/power during a benchmark window.

    Uses gRPC TelemetrySession (via energy-monitor daemon) when available,
    falls back to nvidia-smi subprocess polling.
    """

    def __init__(self, gpu_indices: Optional[str] = None) -> None:
        self._method: Optional[str] = None  # "telemetry" or "nvidia_smi"
        self._session: Any = None
        self._collector_ctx: Any = None
        self._monitor_ctx: Any = None
        self._power_samples: List[Dict[str, float]] = []
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._start_time: float = 0.0
        # Physical GPU indices to scope power measurement (e.g. "4" or "4,5,6,7").
        # When set, nvidia-smi is used with --id= for per-config GPU scoping.
        self._gpu_indices: Optional[str] = gpu_indices

    def start(self) -> "EnergyCollector":
        """Start energy collection. Returns self for chaining."""
        self._start_time = time.time()

        # When gpu_indices is set, the telemetry daemon can't scope to
        # specific GPUs, so go straight to nvidia-smi with --id= filtering.
        if self._gpu_indices is None:
            # Try telemetry daemon first (aggregate across all GPUs)
            try:
                from ipw.telemetry import EnergyMonitorCollector, wait_for_ready
                from ipw.execution.telemetry_session import TelemetrySession

                if wait_for_ready(timeout=2.0):
                    collector = EnergyMonitorCollector()
                    self._session = TelemetrySession(
                        collector,
                        buffer_seconds=7200.0,
                        max_samples=500_000,
                    )
                    self._session.__enter__()
                    self._method = "telemetry"
                    logger.info("Energy collection started via telemetry daemon")
                    return self
            except (ImportError, RuntimeError, OSError) as exc:
                logger.debug("Telemetry daemon unavailable: %s", exc)

        # Fallback (or primary when gpu_indices is set): nvidia-smi polling
        if self._try_nvidia_smi():
            self._method = "nvidia_smi"
            self._power_samples = []
            self._stop_event = threading.Event()
            self._polling_thread = threading.Thread(
                target=self._poll_nvidia_smi, daemon=True
            )
            self._polling_thread.start()
            scope_info = f" (GPUs: {self._gpu_indices})" if self._gpu_indices else ""
            logger.info("Energy collection started via nvidia-smi polling%s", scope_info)
        else:
            logger.warning("No energy collection method available")

        return self

    def stop(self) -> Dict[str, Any]:
        """Stop collection and return energy/power summary."""
        duration_s = time.time() - self._start_time

        if self._method == "telemetry":
            return self._stop_telemetry(duration_s)
        elif self._method == "nvidia_smi":
            return self._stop_nvidia_smi(duration_s)

        return self._empty_result()

    def _stop_telemetry(self, duration_s: float) -> Dict[str, Any]:
        """Extract energy data from TelemetrySession."""
        try:
            samples = list(self._session.readings())
        finally:
            self._session.__exit__(None, None, None)

        if len(samples) < 2:
            return self._empty_result()

        # Energy is a cumulative counter — compute delta
        total_energy_j: Optional[float] = None
        first = samples[0].reading
        last = samples[-1].reading
        if (
            first.energy_joules is not None
            and last.energy_joules is not None
        ):
            delta = last.energy_joules - first.energy_joules
            if delta >= 0:
                total_energy_j = delta

        # Power is instantaneous — average across samples
        power_readings = [
            s.reading.power_watts
            for s in samples
            if s.reading.power_watts is not None
        ]
        avg_power_w = (
            sum(power_readings) / len(power_readings)
            if power_readings
            else None
        )

        # Build power_samples timeline
        power_timeline = [
            {
                "timestamp_s": s.timestamp - self._start_time,
                "power_w": s.reading.power_watts,
            }
            for s in samples
            if s.reading.power_watts is not None
        ]

        # Fallback: approximate energy from average power
        if total_energy_j is None and avg_power_w is not None and duration_s > 0:
            total_energy_j = avg_power_w * duration_s

        return {
            "total_energy_j": total_energy_j,
            "avg_power_w": avg_power_w,
            "power_samples": power_timeline,
            "energy_method": "telemetry",
        }

    def _stop_nvidia_smi(self, duration_s: float) -> Dict[str, Any]:
        """Extract energy data from nvidia-smi polling samples."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._polling_thread is not None:
            self._polling_thread.join(timeout=5)

        if not self._power_samples:
            return self._empty_result()

        power_values = [s["power_w"] for s in self._power_samples]
        avg_power_w = sum(power_values) / len(power_values)
        total_energy_j = avg_power_w * duration_s

        return {
            "total_energy_j": total_energy_j,
            "avg_power_w": avg_power_w,
            "power_samples": self._power_samples,
            "energy_method": "nvidia_smi",
        }

    def _poll_nvidia_smi(self) -> None:
        """Background thread: poll nvidia-smi every 0.5s."""
        while self._stop_event is not None and not self._stop_event.is_set():
            power = self._query_nvidia_smi_power()
            if power is not None:
                self._power_samples.append(
                    {
                        "timestamp_s": time.time() - self._start_time,
                        "power_w": power,
                    }
                )
            self._stop_event.wait(0.5)

    def _query_nvidia_smi_power(self) -> Optional[float]:
        """Query total GPU power draw via nvidia-smi.

        When ``self._gpu_indices`` is set, uses ``--id=`` to scope the query
        to specific physical GPU indices, avoiding over-counting idle GPUs.
        """
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ]
            if self._gpu_indices:
                cmd.insert(1, f"--id={self._gpu_indices}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                values = [
                    float(v.strip())
                    for v in result.stdout.strip().split("\n")
                    if v.strip()
                ]
                return sum(values) if values else None
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    @staticmethod
    def _try_nvidia_smi() -> bool:
        """Check whether nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "total_energy_j": None,
            "avg_power_w": None,
            "power_samples": [],
            "energy_method": None,
        }


# ---------------------------------------------------------------------------
# Streaming request sender
# ---------------------------------------------------------------------------

def _send_streaming_request(
    vllm_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Send a single streaming request and measure TTFT, TBT, E2E.

    Returns per-request dict with ttft_s, tbt_values, e2e_s, tokens_out.
    """
    import requests as req_lib

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    token_times: List[float] = []
    tokens_out = 0

    try:
        with req_lib.post(
            f"{vllm_url}/v1/completions",
            json=payload,
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices and choices[0].get("text"):
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                        tokens_out += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "ttft_s": None,
            "tbt_values": [],
            "e2e_s": end_time - start_time,
            "tokens_out": 0,
            "error": str(e),
        }

    end_time = time.perf_counter()
    ttft_s = (first_token_time - start_time) if first_token_time is not None else None

    # Inter-token intervals
    tbt_values: List[float] = []
    for i in range(1, len(token_times)):
        tbt_values.append(token_times[i] - token_times[i - 1])

    return {
        "ttft_s": ttft_s,
        "tbt_values": tbt_values,
        "e2e_s": end_time - start_time,
        "tokens_out": tokens_out,
    }


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def _generate_prompts(
    workload: str,
    count: int,
    avg_input_tokens: int,
    rng: np.random.Generator,
) -> Tuple[List[str], List[int]]:
    """Generate prompts for a workload type.

    Returns (prompts, input_token_counts).
    Uses simple templates that approximate the target input length.
    Real WildChat/OpenThoughts data can replace these templates when available.
    """
    input_lens = rng.normal(avg_input_tokens, avg_input_tokens * 0.25, size=count)
    input_lens = np.clip(input_lens, 50, avg_input_tokens * 3).astype(int)

    prompts: List[str] = []
    if workload == "chat":
        templates = [
            "Explain the concept of {} in detail, covering its history, applications, and significance.",
            "Write a comprehensive overview of {}. Include examples and key insights.",
            "Discuss the role of {} in modern technology. Provide analysis and future outlook.",
        ]
        topics = [
            "machine learning", "distributed systems", "quantum computing",
            "neural networks", "cloud infrastructure", "data pipelines",
            "natural language processing", "computer vision", "reinforcement learning",
            "compiler optimization",
        ]
        for i, tlen in enumerate(input_lens):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            base = template.format(topic)
            # Pad to approximate target length (roughly 1 token per word)
            padding_words = max(0, int(tlen) - len(base.split()))
            prompt = base + " " + "context " * padding_words
            prompts.append(prompt.strip())
    elif workload == "reasoning":
        templates = [
            "Solve the following step by step, showing all work: What is the integral of {}?",
            "Prove that {} using a rigorous mathematical proof with all intermediate steps.",
            "Analyze the time complexity of {} and provide a detailed derivation.",
        ]
        problems = [
            "x^3 * sin(x) dx", "the harmonic series diverges",
            "merge sort is O(n log n)", "e^(i*pi) + 1 = 0",
            "every continuous function on [a,b] is integrable",
        ]
        for i, tlen in enumerate(input_lens):
            template = templates[i % len(templates)]
            problem = problems[i % len(problems)]
            base = template.format(problem)
            padding_words = max(0, int(tlen) - len(base.split()))
            prompt = base + " " + "detail " * padding_words
            prompts.append(prompt.strip())
    elif workload == "agentic":
        templates = [
            "You are an AI assistant with access to tools. The user asks: {}. "
            "Think step by step, use tools as needed, and provide a final answer.",
            "Given the task: {}, plan the steps needed, execute each one, "
            "and synthesize the results into a coherent response.",
        ]
        tasks = [
            "Find the current weather in San Francisco and suggest clothing",
            "Research the latest advances in battery technology and summarize",
            "Analyze this dataset for anomalies and generate a report",
            "Debug this code snippet and explain the fix",
            "Plan a 3-day itinerary for Tokyo including restaurants",
        ]
        for i, tlen in enumerate(input_lens):
            template = templates[i % len(templates)]
            task = tasks[i % len(tasks)]
            base = template.format(task)
            padding_words = max(0, int(tlen) - len(base.split()))
            prompt = base + " " + "context " * padding_words
            prompts.append(prompt.strip())
    else:
        # Generic fallback
        for tlen in input_lens:
            prompts.append("Explain the following in detail: " + "word " * max(1, int(tlen) - 10))

    return prompts, input_lens.tolist()


# ---------------------------------------------------------------------------
# Single QPS-level benchmark run
# ---------------------------------------------------------------------------

def run_benchmark_at_qps(
    vllm_url: str,
    model_name: str,
    workload: str,
    qps: float,
    duration_s: float,
    seed: int = 42,
    use_energy: bool = False,
    gpu_indices: Optional[str] = None,
) -> Dict[str, Any]:
    """Run benchmark at a single QPS level with streaming measurement.

    Returns aggregate result dict with TTFT/TBT/E2E percentiles and optional
    energy/power data.

    Args:
        gpu_indices: Physical GPU indices (e.g. "4" or "4,5,6,7") to scope
            nvidia-smi power measurement.  When set, only the specified GPUs
            are included in power/energy readings.
    """
    rng = np.random.default_rng(seed)
    profile = WORKLOAD_PROFILES[workload]
    total_requests = max(int(qps * duration_s), 1)

    # Generate arrival times (Poisson process)
    inter_arrival_times = rng.exponential(1.0 / qps, size=total_requests)
    arrival_times = np.cumsum(inter_arrival_times)

    # Generate prompts
    prompts, input_lens = _generate_prompts(
        workload, total_requests, profile["avg_input_tokens"], rng,
    )

    # Generate output token limits
    output_limits = rng.normal(
        profile["avg_output_tokens"],
        profile["avg_output_tokens"] * 0.3,
        size=total_requests,
    ).astype(int).clip(10, profile["avg_output_tokens"] * 3)

    # Start energy collection if requested
    energy_collector: Optional[EnergyCollector] = None
    if use_energy:
        energy_collector = EnergyCollector(gpu_indices=gpu_indices)
        energy_collector.start()

    all_ttft: List[float] = []
    all_tbt: List[float] = []
    all_e2e: List[float] = []
    completed = 0
    total_tokens_out = 0

    # Use ThreadPoolExecutor to dispatch requests concurrently at their
    # scheduled arrival times (Poisson process).  The serial loop only
    # sleeps until arrival time then submits; the actual HTTP streaming
    # runs in a worker thread so later requests are not blocked.
    max_workers = max(int(qps * 120) + 10, 32)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures: List[concurrent.futures.Future] = []

    start_wall = time.perf_counter()

    for i in range(total_requests):
        elapsed = time.perf_counter() - start_wall
        wait = arrival_times[i] - elapsed
        if wait > 0:
            time.sleep(wait)
        futures.append(
            executor.submit(
                _send_streaming_request,
                vllm_url,
                model_name,
                prompts[i],
                int(output_limits[i]),
            )
        )

    # Collect results as they complete
    for fut in concurrent.futures.as_completed(futures):
        result = fut.result()
        if result.get("error"):
            continue
        completed += 1
        total_tokens_out += result["tokens_out"]
        if result["ttft_s"] is not None:
            all_ttft.append(result["ttft_s"])
        all_tbt.extend(result["tbt_values"])
        all_e2e.append(result["e2e_s"])

    executor.shutdown(wait=True)

    wall_time = time.perf_counter() - start_wall
    logger.info(
        "  Completed %d/%d requests in %.1fs",
        completed, total_requests, wall_time,
    )

    # Collect energy data
    energy_data: Dict[str, Any] = {
        "total_energy_j": None,
        "avg_power_w": None,
        "energy_per_token_j": None,
        "energy_per_request_j": None,
        "power_samples": [],
        "energy_method": None,
    }
    if energy_collector is not None:
        raw = energy_collector.stop()
        energy_data["total_energy_j"] = raw.get("total_energy_j")
        energy_data["avg_power_w"] = raw.get("avg_power_w")
        energy_data["power_samples"] = raw.get("power_samples", [])
        energy_data["energy_method"] = raw.get("energy_method")
        if raw.get("total_energy_j") is not None:
            if total_tokens_out > 0:
                energy_data["energy_per_token_j"] = (
                    raw["total_energy_j"] / total_tokens_out
                )
            if completed > 0:
                energy_data["energy_per_request_j"] = (
                    raw["total_energy_j"] / completed
                )

    return {
        "model": model_name,
        "workload": workload,
        "qps_target": qps,
        "ttft": _percentiles(all_ttft),
        "tbt": _percentiles(all_tbt),
        "e2e": _percentiles(all_e2e),
        "throughput_tps": total_tokens_out / wall_time if wall_time > 0 else 0.0,
        "throughput_rps": completed / wall_time if wall_time > 0 else 0.0,
        "total_requests": total_requests,
        "completed_requests": completed,
        "duration_s": wall_time,
        **energy_data,
    }


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

_DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}

# Mock packages directory that shadows broken system tensorflow/h5py
_MOCK_PACKAGES_DIR = str(
    Path(__file__).resolve().parents[3] / "scripts" / "mock_packages"
)


def _start_vllm_server(
    model_id: str,
    config: Dict[str, Any],
    port: int = 8000,
    cuda_devices: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server."""
    dtype = _DTYPE_MAP.get(config["precision"], config["precision"])
    vllm_args = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--tensor-parallel-size", str(config["tensor_parallel"]),
        "--dtype", dtype,
        "--port", str(port),
        "--gpu-memory-utilization", "0.9",
    ]
    if config.get("engine_config", {}).get("quantization"):
        vllm_args += ["--quantization", config["engine_config"]["quantization"]]

    import os
    env = {**os.environ}
    # Prepend mock packages to PYTHONPATH so subprocesses also find the mocks
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _MOCK_PACKAGES_DIR + (":" + existing_pp if existing_pp else "")
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Redirect to files (not PIPE) to avoid subprocess deadlock when buffers fill
    stdout_dest = subprocess.DEVNULL
    stderr_dest = subprocess.DEVNULL
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model_id.replace("/", "_")
        stdout_dest = open(log_dir / f"vllm_{safe_model}_{port}.stdout.log", "w")
        stderr_dest = open(log_dir / f"vllm_{safe_model}_{port}.stderr.log", "w")

    logger.info("  Starting vLLM: %s", " ".join(vllm_args))
    return subprocess.Popen(vllm_args, stdout=stdout_dest, stderr=stderr_dest, env=env)


def _wait_for_server(url: str, timeout_s: int = 240) -> bool:
    """Poll server health endpoint until ready."""
    import urllib.request

    for _ in range(timeout_s // 2):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    """Terminate vLLM server process."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    model_keys: List[str],
    qps_levels: List[float],
    output_dir: Path,
    duration_s: float,
    port: int = 8000,
    seed: int = 42,
    cuda_devices: Optional[str] = None,
    use_energy: bool = False,
) -> List[Dict[str, Any]]:
    """Run ground-truth benchmarks for all model x config x workload x QPS combos.

    Skips runs whose output JSON already exists (cached).
    """
    model_specs = get_model_specs()
    vllm_url = f"http://localhost:{port}"

    all_results: List[Dict[str, Any]] = []
    workloads = list(WORKLOAD_PROFILES.keys())

    total = len(model_keys) * len(BENCHMARK_CONFIGS) * len(workloads) * len(qps_levels)
    run_idx = 0

    for model_key in model_keys:
        if model_key not in model_specs:
            logger.error("Unknown model key: %s", model_key)
            continue
        model_spec = model_specs[model_key]
        model_id = model_spec.model_id

        for config_id, config in BENCHMARK_CONFIGS.items():
            # Skip configs requiring more GPUs than available
            if cuda_devices:
                visible_gpus = [g.strip() for g in cuda_devices.split(",")]
                if config["num_gpus"] > len(visible_gpus):
                    logger.info(
                        "Skipping %s: needs %d GPUs, only %d available",
                        config_id, config["num_gpus"], len(visible_gpus),
                    )
                    run_idx += len(workloads) * len(qps_levels)
                    continue

            # Compute physical GPU indices for this config's num_gpus
            # so nvidia-smi power measurements only include active GPUs.
            config_gpu_indices: Optional[str] = None
            if cuda_devices:
                visible_gpus = [g.strip() for g in cuda_devices.split(",")]
                config_gpu_count = config["num_gpus"]
                config_gpu_indices = ",".join(visible_gpus[:config_gpu_count])

            # Check if all runs for this model+config are cached
            all_cached = True
            for wt in workloads:
                for qps in qps_levels:
                    out_path = _output_path(output_dir, model_key, config_id, wt, qps)
                    if not out_path.exists():
                        all_cached = False
                        break
                if not all_cached:
                    break

            if all_cached:
                logger.info(
                    "All runs cached for %s / %s, skipping server start",
                    model_key, config_id,
                )
                # Load cached results
                for wt in workloads:
                    for qps in qps_levels:
                        run_idx += 1
                        out_path = _output_path(output_dir, model_key, config_id, wt, qps)
                        with open(out_path) as f:
                            all_results.append(json.load(f))
                continue

            # Start vLLM server for this model+config
            logger.info("=== Starting server: %s / %s ===", model_key, config_id)
            server_proc = _start_vllm_server(
                model_id, config, port=port, cuda_devices=cuda_devices,
                log_dir=output_dir / "vllm_logs",
            )

            if not _wait_for_server(vllm_url):
                logger.error("Server failed to start for %s / %s", model_key, config_id)
                _stop_server(server_proc)
                run_idx += len(workloads) * len(qps_levels)
                continue

            logger.info("  Server ready")

            try:
                for wt in workloads:
                    for qps in qps_levels:
                        run_idx += 1
                        out_path = _output_path(output_dir, model_key, config_id, wt, qps)

                        # Skip if cached
                        if out_path.exists():
                            logger.info(
                                "[%d/%d] CACHED %s / %s / %s / qps=%g",
                                run_idx, total, model_key, config_id, wt, qps,
                            )
                            with open(out_path) as f:
                                all_results.append(json.load(f))
                            continue

                        logger.info(
                            "[%d/%d] Benchmarking %s / %s / %s / qps=%g",
                            run_idx, total, model_key, config_id, wt, qps,
                        )

                        result = run_benchmark_at_qps(
                            vllm_url=vllm_url,
                            model_name=model_id,
                            workload=wt,
                            qps=qps,
                            duration_s=duration_s,
                            seed=seed,
                            use_energy=use_energy,
                            gpu_indices=config_gpu_indices,
                        )
                        result["config_id"] = config_id
                        result["model_key"] = model_key

                        # Save per-run JSON
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, "w") as f:
                            json.dump(result, f, indent=2, default=str)

                        all_results.append(result)
                        energy_info = ""
                        if result.get("avg_power_w") is not None:
                            energy_info = f", power={result['avg_power_w']:.1f}W"
                        if result.get("energy_per_token_j") is not None:
                            energy_info += f", e/tok={result['energy_per_token_j']:.4f}J"
                        logger.info(
                            "  TTFT p50=%.3fs, TBT p50=%.4fs, E2E p50=%.3fs, "
                            "throughput=%.1f tps (%d/%d completed)%s",
                            result["ttft"]["p50"],
                            result["tbt"]["p50"],
                            result["e2e"]["p50"],
                            result["throughput_tps"],
                            result["completed_requests"],
                            result["total_requests"],
                            energy_info,
                        )
            finally:
                logger.info("  Stopping server for %s / %s", model_key, config_id)
                _stop_server(server_proc)

    logger.info("Benchmark complete: %d total runs", len(all_results))
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ground-truth vLLM benchmark with streaming TTFT/TBT measurement"
    )
    parser.add_argument(
        "--output-dir", default="data/ground_truth",
        help="Root output directory (default: data/ground_truth)",
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys (default: %(default)s)",
    )
    parser.add_argument(
        "--qps-levels", default=",".join(str(q) for q in DEFAULT_QPS_LEVELS),
        help="Comma-separated QPS levels (default: %(default)s)",
    )
    parser.add_argument(
        "--duration-s", type=float, default=60.0,
        help="Duration per benchmark run in seconds (default: 60)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--cuda-devices", default=None,
        help="CUDA_VISIBLE_DEVICES override (e.g., '0,1')",
    )
    parser.add_argument(
        "--use-energy", action="store_true", default=True,
        help="Collect GPU energy/power during benchmarks (default: True)",
    )
    parser.add_argument(
        "--no-energy", action="store_false", dest="use_energy",
        help="Disable energy/power collection",
    )

    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",")]
    qps_levels = [float(q.strip()) for q in args.qps_levels.split(",")]
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Ground Truth Benchmark")
    logger.info("=" * 60)
    logger.info("Models: %s", model_keys)
    logger.info("Configs: %s", list(BENCHMARK_CONFIGS.keys()))
    logger.info("Workloads: %s", list(WORKLOAD_PROFILES.keys()))
    logger.info("QPS levels: %s", qps_levels)
    logger.info("Duration: %.0fs per run", args.duration_s)
    logger.info("Output: %s", output_dir)
    logger.info("Energy collection: %s", "enabled" if args.use_energy else "disabled")
    logger.info(
        "Total runs: %d",
        len(model_keys) * len(BENCHMARK_CONFIGS) * len(WORKLOAD_PROFILES) * len(qps_levels),
    )

    start = time.time()
    results = run_all_benchmarks(
        model_keys=model_keys,
        qps_levels=qps_levels,
        output_dir=output_dir,
        duration_s=args.duration_s,
        port=args.port,
        seed=args.seed,
        cuda_devices=args.cuda_devices,
        use_energy=args.use_energy,
    )
    elapsed = time.time() - start

    # Print summary
    print()
    print("=" * 100)
    print("GROUND TRUTH BENCHMARK SUMMARY")
    print("=" * 100)
    header = (
        f"{'Model':<14} {'Config':<12} {'Workload':<10} {'QPS':>4} "
        f"{'TTFT p50':>9} {'TBT p50':>9} {'E2E p50':>9} {'TPS':>7} "
        f"{'Power W':>8} {'E/tok J':>8} {'Done':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x.get("model_key", ""), x.get("config_id", ""), x.get("workload", ""), x.get("qps_target", 0))):
        power_str = (
            f"{r['avg_power_w']:>7.1f}"
            if r.get("avg_power_w") is not None
            else "    n/a"
        )
        etok_str = (
            f"{r['energy_per_token_j']:>7.4f}"
            if r.get("energy_per_token_j") is not None
            else "    n/a"
        )
        print(
            f"{r.get('model_key', ''):.<14} {r.get('config_id', ''):.<12} "
            f"{r.get('workload', ''):.<10} {r.get('qps_target', 0):>4.0f} "
            f"{r['ttft']['p50']:>8.3f}s {r['tbt']['p50']:>8.4f}s "
            f"{r['e2e']['p50']:>8.3f}s {r['throughput_tps']:>6.0f} "
            f"{power_str} {etok_str} "
            f"{r['completed_requests']:>4}/{r['total_requests']}"
        )

    print()
    logger.info("Total elapsed: %.1fs", elapsed)
    logger.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
