#!/usr/bin/env python3
"""Run EventDrivenSimulator at validation QPS levels.

Simulates inference at the same QPS levels as run_validation_benchmark.py
using the EventDrivenSimulator with VLLMScheduler and LUT-based timing.

Usage:
    python run_validation_simulation.py \
        --model-id Qwen/Qwen3-8B \
        --gpu-type a100_80gb \
        --lut-dir data/luts_v2/ \
        --qps-levels 1,2,5,10,20 \
        --duration 60 \
        --output-dir data/validation
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validation_simulation")


def run_simulation_at_qps(
    model_spec: Any,
    hardware_spec: Any,
    inference_spec: Any,
    lut_bundle: Any,
    qps: float,
    duration_s: float,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one simulation at a given QPS level.

    Args:
        model_spec: ModelSpec for the model.
        hardware_spec: HardwareSpec for the GPU.
        inference_spec: InferenceSpec for vLLM config.
        lut_bundle: LUTBundle for timing lookup.
        qps: Target queries per second.
        duration_s: Simulation duration in seconds.
        avg_input_tokens: Average input tokens per request.
        avg_output_tokens: Average output tokens per request.
        seed: Random seed.

    Returns:
        Dict with simulation metrics at this QPS level.
    """
    from inference_simulator.engine.simulator import EventDrivenSimulator
    from inference_simulator.scheduler.vllm import VLLMScheduler
    from inference_simulator.types import WorkloadSpec

    scheduler = VLLMScheduler(
        max_num_seqs=inference_spec.engine_config.get("max_num_seqs", 256),
        max_num_batched_tokens=inference_spec.engine_config.get(
            "max_num_batched_tokens", 8192
        ),
        enable_chunked_prefill=inference_spec.engine_config.get(
            "enable_chunked_prefill", True
        ),
    )

    simulator = EventDrivenSimulator(
        model_spec=model_spec,
        hardware_spec=hardware_spec,
        inference_spec=inference_spec,
        scheduler=scheduler,
        lut_bundle=lut_bundle,
    )

    workload_spec = WorkloadSpec(
        qps=qps,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        input_token_std=avg_input_tokens * 0.3,
        output_token_std=avg_output_tokens * 0.3,
    )

    logger.info("QPS=%.1f: running simulation for %.0fs", qps, duration_s)
    metrics = simulator.run(
        workload_spec=workload_spec,
        duration_s=duration_s,
        seed=seed,
    )

    return {
        "qps_target": qps,
        "total_requests": metrics.total_requests,
        "total_tokens_out": metrics.total_tokens_generated,
        "throughput_rps": metrics.throughput_rps,
        "throughput_tps": metrics.throughput_tps,
        "ttft": {
            "p50": metrics.ttft_p50,
            "p90": metrics.ttft_p90,
            "p99": metrics.ttft_p99,
        },
        "tbt": {
            "p50": metrics.tbt_p50,
            "p90": metrics.tbt_p90,
            "p99": metrics.tbt_p99,
        },
        "e2e": {
            "p50": metrics.e2e_p50,
            "p90": metrics.e2e_p90,
            "p99": metrics.e2e_p99,
        },
        "total_energy_j": metrics.total_energy_j,
        "avg_power_w": metrics.avg_power_w,
        "gpu_utilization": metrics.gpu_utilization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EventDrivenSimulator at validation QPS levels"
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-8B",
        help="Model ID (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--model-params-b", type=float, default=8.0,
        help="Model parameters in billions (default: 8.0)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=36,
        help="Number of transformer layers (default: 36)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=4096,
        help="Hidden dimension size (default: 4096)",
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=32,
        help="Number of attention heads (default: 32)",
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=8,
        help="Number of KV heads (default: 8)",
    )
    parser.add_argument(
        "--gpu-type", default="a100_80gb",
        help="GPU type identifier (default: a100_80gb)",
    )
    parser.add_argument(
        "--gpu-memory-gb", type=float, default=80.0,
        help="GPU memory in GB (default: 80.0)",
    )
    parser.add_argument(
        "--tdp-watts", type=float, default=400.0,
        help="GPU TDP in watts (default: 400.0)",
    )
    parser.add_argument(
        "--peak-fp16-tflops", type=float, default=312.0,
        help="Peak FP16 TFLOPS (default: 312.0)",
    )
    parser.add_argument(
        "--hbm-bandwidth-gb-s", type=float, default=2039.0,
        help="HBM bandwidth in GB/s (default: 2039.0)",
    )
    parser.add_argument(
        "--precision", default="bf16",
        help="Model precision (default: bf16)",
    )
    parser.add_argument(
        "--lut-dir", default="data/luts_v2",
        help="LUT bundle directory (default: data/luts_v2)",
    )
    parser.add_argument(
        "--qps-levels", default="1,2,5,10,20",
        help="Comma-separated QPS levels (default: 1,2,5,10,20)",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Simulation duration per QPS level in seconds (default: 60)",
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

    from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec
    from inference_simulator.types.lut_bundle import LUTBundle

    # Build specs
    model_spec = ModelSpec(
        model_id=args.model_id,
        total_params_billion=args.model_params_b,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
    )

    hardware_spec = HardwareSpec(
        name=args.gpu_type,
        memory_gb=args.gpu_memory_gb,
        tdp_watts=args.tdp_watts,
        peak_fp16_tflops=args.peak_fp16_tflops,
        hbm_bandwidth_gb_s=args.hbm_bandwidth_gb_s,
    )

    inference_spec = InferenceSpec.for_vllm(
        num_gpus=1,
        tensor_parallel=1,
        precision=args.precision,
    )

    # Load LUT bundle
    lut_dir = Path(args.lut_dir)
    lut_bundle = None
    if lut_dir.exists():
        token_ops = lut_dir / "gpu_token_ops.npz"
        prefill_lut = lut_dir / "gpu_attention_prefill.npz"
        decode_lut = lut_dir / "gpu_attention_decode.npz"
        if all(p.exists() for p in [token_ops, prefill_lut, decode_lut]):
            lut_bundle = LUTBundle(
                base_dir=lut_dir,
                model_id=args.model_id,
                hardware_id=args.gpu_type,
                quantization=args.precision,
                gpu_token_ops_lut=token_ops,
                gpu_attention_prefill_lut=prefill_lut,
                gpu_attention_decode_lut=decode_lut,
            )
            logger.info("Loaded LUT bundle from %s", lut_dir)
        else:
            logger.warning("LUT files not found in %s; using roofline fallback", lut_dir)
    else:
        logger.warning("LUT directory does not exist: %s; using roofline fallback", lut_dir)

    qps_levels = [float(q.strip()) for q in args.qps_levels.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Validation Simulation ===")
    logger.info("Model: %s (%.1fB params)", args.model_id, args.model_params_b)
    logger.info("GPU: %s", args.gpu_type)
    logger.info("QPS levels: %s", qps_levels)
    logger.info("Duration: %.0fs per level", args.duration)
    logger.info("LUT bundle: %s", "loaded" if lut_bundle else "roofline fallback")

    all_results: List[Dict[str, Any]] = []

    for qps in qps_levels:
        result = run_simulation_at_qps(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            lut_bundle=lut_bundle,
            qps=qps,
            duration_s=args.duration,
            avg_input_tokens=args.avg_input_tokens,
            avg_output_tokens=args.avg_output_tokens,
            seed=args.seed,
        )
        all_results.append(result)

        logger.info(
            "QPS=%.1f: %d requests, throughput=%.1f RPS / %.1f TPS, "
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
    output_path = output_dir / f"{model_short}_{args.gpu_type}_vllm_simulated.jsonl"
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
