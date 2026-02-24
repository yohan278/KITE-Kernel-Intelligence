#!/usr/bin/env python3
"""Run real vLLM serving benchmark for Phase 0 validation.

Sends requests from a synthetic Poisson trace at multiple QPS levels,
recording per-request latency metrics and cluster-level throughput/power.

Usage:
    python run_validation_benchmark.py \
        --model-id Qwen/Qwen3-8B \
        --gpu-type a100_80gb \
        --vllm-url http://localhost:8000 \
        --qps-levels 1,2,5,10,20 \
        --duration 60 \
        --output-dir data/validation
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validation_benchmark")


def _query_nvidia_smi() -> Optional[float]:
    """Read current GPU power draw via nvidia-smi (watts)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            values = [float(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
            return sum(values)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _send_request(
    vllm_url: str,
    model_name: str,
    input_tokens: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    """Send a single request to the vLLM server and record timing.

    Returns a dict with ttft_s, tbt_s, e2e_latency_s, tokens_in, tokens_out.
    """
    import requests

    # Generate a prompt with approximately input_tokens tokens
    prompt = "Explain the following concept in detail: " + "word " * max(1, input_tokens - 10)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_output_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_times: List[float] = []
    tokens_out = 0

    try:
        with requests.post(
            f"{vllm_url}/v1/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
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
            "tbt_s": None,
            "e2e_latency_s": end_time - start_time,
            "tokens_in": input_tokens,
            "tokens_out": 0,
            "error": str(e),
        }

    end_time = time.perf_counter()

    ttft_s = (first_token_time - start_time) if first_token_time is not None else None

    # Compute inter-token times
    tbt_values = []
    for i in range(1, len(token_times)):
        tbt_values.append(token_times[i] - token_times[i - 1])
    tbt_s = float(np.median(tbt_values)) if tbt_values else None

    return {
        "ttft_s": ttft_s,
        "tbt_s": tbt_s,
        "e2e_latency_s": end_time - start_time,
        "tokens_in": input_tokens,
        "tokens_out": tokens_out,
    }


def run_qps_level(
    vllm_url: str,
    model_name: str,
    qps: float,
    duration_s: float,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run benchmark at a single QPS level.

    Args:
        vllm_url: vLLM server URL.
        model_name: Model name for vLLM requests.
        qps: Target queries per second.
        duration_s: Duration in seconds.
        avg_input_tokens: Average input token count.
        avg_output_tokens: Average output token count.
        seed: Random seed.

    Returns:
        Dict with per-request results and aggregate metrics.
    """
    rng = np.random.default_rng(seed)
    total_requests = int(qps * duration_s)
    if total_requests < 1:
        total_requests = 1

    # Generate Poisson arrival times
    inter_arrival_times = rng.exponential(1.0 / qps, size=total_requests)
    arrival_times = np.cumsum(inter_arrival_times)

    # Generate per-request token counts
    input_tokens_list = np.maximum(
        10, rng.normal(avg_input_tokens, avg_input_tokens * 0.3, size=total_requests).astype(int)
    )
    output_tokens_list = np.maximum(
        10, rng.normal(avg_output_tokens, avg_output_tokens * 0.3, size=total_requests).astype(int)
    )

    logger.info("QPS=%.1f: sending %d requests over %.0fs", qps, total_requests, duration_s)

    results: List[Dict[str, Any]] = []
    power_samples: List[float] = []
    start_wall = time.perf_counter()

    for i in range(total_requests):
        # Wait until the scheduled arrival time
        elapsed = time.perf_counter() - start_wall
        wait = arrival_times[i] - elapsed
        if wait > 0:
            time.sleep(wait)

        # Sample GPU power
        power = _query_nvidia_smi()
        if power is not None:
            power_samples.append(power)

        result = _send_request(
            vllm_url=vllm_url,
            model_name=model_name,
            input_tokens=int(input_tokens_list[i]),
            max_output_tokens=int(output_tokens_list[i]),
        )
        result["request_index"] = i
        result["scheduled_arrival_s"] = float(arrival_times[i])
        results.append(result)

        if (i + 1) % 10 == 0 or i + 1 == total_requests:
            logger.info(
                "  QPS=%.1f: %d/%d requests completed", qps, i + 1, total_requests
            )

    total_wall_s = time.perf_counter() - start_wall

    # Aggregate metrics
    ttft_values = [r["ttft_s"] for r in results if r.get("ttft_s") is not None]
    tbt_values = [r["tbt_s"] for r in results if r.get("tbt_s") is not None]
    e2e_values = [r["e2e_latency_s"] for r in results if r.get("e2e_latency_s") is not None]
    total_tokens_out = sum(r.get("tokens_out", 0) for r in results)

    def _pcts(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        arr = np.array(vals)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    return {
        "qps_target": qps,
        "qps_achieved": len(results) / total_wall_s if total_wall_s > 0 else 0.0,
        "duration_s": total_wall_s,
        "total_requests": len(results),
        "total_tokens_out": total_tokens_out,
        "throughput_rps": len(results) / total_wall_s if total_wall_s > 0 else 0.0,
        "throughput_tps": total_tokens_out / total_wall_s if total_wall_s > 0 else 0.0,
        "ttft": _pcts(ttft_values),
        "tbt": _pcts(tbt_values),
        "e2e": _pcts(e2e_values),
        "avg_power_w": float(np.mean(power_samples)) if power_samples else 0.0,
        "per_request_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real vLLM serving benchmark for validation"
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-8B",
        help="Model ID served by vLLM (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--gpu-type", default="a100_80gb",
        help="GPU type identifier (default: a100_80gb)",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000",
        help="vLLM endpoint URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--qps-levels", default="1,2,5,10,20",
        help="Comma-separated QPS levels (default: 1,2,5,10,20)",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Duration per QPS level in seconds (default: 60)",
    )
    parser.add_argument(
        "--avg-input-tokens", type=int, default=500,
        help="Average input tokens per request (default: 500)",
    )
    parser.add_argument(
        "--avg-output-tokens", type=int, default=200,
        help="Average output tokens per request (default: 200)",
    )
    parser.add_argument(
        "--output-dir", default="data/validation",
        help="Output directory (default: data/validation)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    qps_levels = [float(q.strip()) for q in args.qps_levels.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Validation Benchmark ===")
    logger.info("Model: %s", args.model_id)
    logger.info("GPU: %s", args.gpu_type)
    logger.info("vLLM URL: %s", args.vllm_url)
    logger.info("QPS levels: %s", qps_levels)
    logger.info("Duration: %.0fs per level", args.duration)
    logger.info("Output: %s", output_dir)

    all_results: List[Dict[str, Any]] = []

    for qps in qps_levels:
        logger.info("\n--- Running QPS=%.1f ---", qps)
        result = run_qps_level(
            vllm_url=args.vllm_url,
            model_name=args.model_id,
            qps=qps,
            duration_s=args.duration,
            avg_input_tokens=args.avg_input_tokens,
            avg_output_tokens=args.avg_output_tokens,
            seed=args.seed,
        )
        all_results.append(result)

        logger.info(
            "QPS=%.1f complete: %d requests, throughput=%.1f RPS / %.1f TPS, "
            "TTFT p50=%.3fs, TBT p50=%.4fs, E2E p50=%.3fs",
            qps,
            result["total_requests"],
            result["throughput_rps"],
            result["throughput_tps"],
            result["ttft"]["p50"],
            result["tbt"]["p50"],
            result["e2e"]["p50"],
        )

    # Save results
    model_short = args.model_id.replace("/", "_").lower()
    output_path = output_dir / f"{model_short}_{args.gpu_type}_vllm_real.jsonl"
    with open(output_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result, default=str) + "\n")
    logger.info("\nResults saved to %s", output_path)

    # Summary
    logger.info("\n=== Summary ===")
    for result in all_results:
        logger.info(
            "QPS=%5.1f | RPS=%5.1f | TPS=%7.1f | TTFT p50=%.3fs p99=%.3fs | "
            "E2E p50=%.3fs p99=%.3fs | Power=%.0fW",
            result["qps_target"],
            result["throughput_rps"],
            result["throughput_tps"],
            result["ttft"]["p50"],
            result["ttft"]["p99"],
            result["e2e"]["p50"],
            result["e2e"]["p99"],
            result["avg_power_w"],
        )


if __name__ == "__main__":
    main()
