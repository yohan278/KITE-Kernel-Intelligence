#!/usr/bin/env python3
"""Run Pipeline #2 with active vs synthetic profiles and compare results.

Usage:
    python run_active_pipeline.py --output-dir data/active_vs_synthetic --duration-s 60.0
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("active_pipeline")

# Workload type -> synthetic profile dataset key
WORKLOAD_TO_DATASET = {
    "chat": "wildchat",
    "reasoning": "openthoughts",
    "rag": "hotpotqa",
    "agentic": "agentdata",
}

# Default paths
DEFAULT_LUT_DIR = Path("data/luts")
DEFAULT_SYNTHETIC_DIR = Path("data/workload_profiles")
DEFAULT_ACTIVE_DIR = Path("data/active_characterization/workload_profiles")

MODEL_ID = "Qwen/Qwen3-8B"
HARDWARE_KEY = "a100_80gb"
PRECISION = "fp16"


def load_lut_bundle(lut_dir: Path):
    """Load a LUTBundle from a directory of .npz files."""
    from inference_simulator.types.lut_bundle import LUTBundle

    def _find(name: str) -> Optional[Path]:
        p = lut_dir / name
        return p if p.exists() else None

    bundle = LUTBundle(
        base_dir=lut_dir,
        model_id=MODEL_ID,
        hardware_id=HARDWARE_KEY,
        quantization=PRECISION,
        gpu_token_ops_lut=_find("gpu_token_ops.npz") or _find("token_ops.npz"),
        gpu_attention_prefill_lut=_find("gpu_attention_prefill.npz") or _find("attention_prefill.npz"),
        gpu_attention_decode_lut=_find("gpu_attention_decode.npz") or _find("attention_decode.npz"),
        gpu_moe_lut=_find("gpu_moe.npz") or _find("moe.npz"),
        network_lut=_find("network.npz"),
        energy_lut=_find("energy.npz"),
        tool_distributions=_find("tool_distributions.pkl"),
    )
    if not bundle.exists():
        raise FileNotFoundError(f"LUT bundle at {lut_dir} missing required files")
    return bundle


def run_simulation(
    lut_bundle,
    workload_type: str,
    workload_profile,
    duration_s: float,
):
    """Run EventDrivenSimulator with the given profile and return metrics."""
    from dataset_generator.cli import _load_model_spec
    from inference_simulator.engine.simulator import EventDrivenSimulator
    from inference_simulator.scheduler.vllm import VLLMScheduler
    from inference_simulator.types import (
        HardwareSpec,
        InferenceSpec,
        WorkloadSpec,
    )

    model_spec = _load_model_spec(MODEL_ID)
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    inference_spec = InferenceSpec(precision=PRECISION)

    scheduler = VLLMScheduler(
        max_num_seqs=inference_spec.max_batch_size,
        max_num_batched_tokens=inference_spec.max_batch_size * 2048,
    )
    sim = EventDrivenSimulator(
        model_spec=model_spec,
        hardware_spec=hw_spec,
        inference_spec=inference_spec,
        scheduler=scheduler,
        lut_bundle=lut_bundle,
    )

    # Build workload spec from type
    _FACTORIES = {
        "chat": WorkloadSpec.for_chat,
        "reasoning": WorkloadSpec.for_reasoning,
        "agentic": WorkloadSpec.for_agentic,
        "rag": WorkloadSpec.for_rag,
    }
    factory = _FACTORIES.get(workload_type, WorkloadSpec)
    workload = factory(qps=1.0) if callable(factory) else WorkloadSpec()

    metrics = sim.run(
        workload, duration_s=duration_s, seed=42,
        workload_profile=workload_profile,
    )
    return metrics


def format_value(val: float, fmt: str = ".3f") -> str:
    """Format a float value, returning '-' for zero."""
    if val == 0.0:
        return "-"
    return f"{val:{fmt}}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare active vs synthetic profiles through Pipeline #2"
    )
    parser.add_argument(
        "--output-dir", default="data/active_vs_synthetic",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--duration-s", type=float, default=60.0,
        help="Simulation duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--lut-dir", default=str(DEFAULT_LUT_DIR),
        help="Directory containing LUT .npz files",
    )
    parser.add_argument(
        "--synthetic-dir", default=str(DEFAULT_SYNTHETIC_DIR),
        help="Directory containing synthetic profile JSONs",
    )
    parser.add_argument(
        "--active-dir", default=str(DEFAULT_ACTIVE_DIR),
        help="Directory containing active profile JSONs",
    )
    parser.add_argument(
        "--workloads", default="chat,reasoning,rag,agentic",
        help="Comma-separated workload types to compare",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lut_dir = Path(args.lut_dir)
    synthetic_dir = Path(args.synthetic_dir)
    active_dir = Path(args.active_dir)
    workloads = [w.strip() for w in args.workloads.split(",")]

    logger.info("=== Active vs Synthetic Profile Comparison ===")
    logger.info("LUT dir: %s", lut_dir)
    logger.info("Synthetic dir: %s", synthetic_dir)
    logger.info("Active dir: %s", active_dir)
    logger.info("Workloads: %s", workloads)
    logger.info("Duration: %.1fs", args.duration_s)

    # Load LUT bundle once
    logger.info("\nLoading LUT bundle from %s...", lut_dir)
    lut_bundle = load_lut_bundle(lut_dir)
    logger.info("LUT bundle loaded successfully")

    from inference_simulator.types.workload_profile import WorkloadProfile

    results: List[Dict[str, Any]] = []

    for wt in workloads:
        dataset_key = WORKLOAD_TO_DATASET.get(wt, wt)

        # Load synthetic profile
        synthetic_path = synthetic_dir / f"{dataset_key}_profile.json"
        if not synthetic_path.exists():
            logger.warning("Synthetic profile not found: %s, skipping", synthetic_path)
            continue

        # Load active profile
        active_path = active_dir / f"{wt}_active_profile.json"
        if not active_path.exists():
            logger.warning("Active profile not found: %s, skipping", active_path)
            continue

        synthetic_profile = WorkloadProfile.load(synthetic_path)
        active_profile = WorkloadProfile.load(active_path)

        logger.info("\n--- %s ---", wt)
        logger.info("Synthetic: %d samples, input_mean=%.0f",
                     synthetic_profile.n_samples,
                     synthetic_profile.input_tokens_dist.mean if synthetic_profile.input_tokens_dist else 0)
        logger.info("Active: %d samples, input_mean=%.0f",
                     active_profile.n_samples,
                     active_profile.input_tokens_dist.mean if active_profile.input_tokens_dist else 0)

        # Run simulator with synthetic profile
        logger.info("Running simulation with synthetic profile...")
        synthetic_metrics = run_simulation(
            lut_bundle, wt, synthetic_profile, args.duration_s,
        )
        logger.info("  Synthetic: %d requests, %.1f tok/s",
                     synthetic_metrics.total_requests, synthetic_metrics.throughput_tps)

        # Run simulator with active profile
        logger.info("Running simulation with active profile...")
        active_metrics = run_simulation(
            lut_bundle, wt, active_profile, args.duration_s,
        )
        logger.info("  Active: %d requests, %.1f tok/s",
                     active_metrics.total_requests, active_metrics.throughput_tps)

        # Collect profile stats
        def _profile_stats(profile: WorkloadProfile) -> Dict[str, float]:
            return {
                "n_samples": profile.n_samples,
                "input_mean": profile.input_tokens_dist.mean if profile.input_tokens_dist else 0,
                "output_mean": (profile.answer_tokens_dist.mean if profile.answer_tokens_dist else 0),
            }

        results.append({
            "workload": wt,
            "synthetic": {
                "profile": _profile_stats(synthetic_profile),
                "metrics": asdict(synthetic_metrics),
            },
            "active": {
                "profile": _profile_stats(active_profile),
                "metrics": asdict(active_metrics),
            },
        })

    # Print comparison table
    print("\n=== Active vs Synthetic Profile Comparison ===\n")
    header = (
        f"{'Workload':<12}| {'Source':<10}| {'Requests':>8} | {'Throughput':>10} | "
        f"{'TTFT P50':>9} | {'E2E P50':>9} | {'Avg In':>7} | {'Avg Out':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        wt = row["workload"]
        for source in ["synthetic", "active"]:
            m = row[source]["metrics"]
            p = row[source]["profile"]
            print(
                f"{wt:<12}| {source:<10}| {m['total_requests']:>8} | "
                f"{format_value(m['throughput_tps'], '.0f'):>8} t/s | "
                f"{format_value(m['ttft_p50'], '.3f'):>8}s | "
                f"{format_value(m['e2e_p50'], '.3f'):>8}s | "
                f"{p['input_mean']:>7.0f} | {p['output_mean']:>7.0f}"
            )

    # Save JSON summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "model_id": MODEL_ID,
                "hardware_key": HARDWARE_KEY,
                "precision": PRECISION,
                "duration_s": args.duration_s,
                "lut_dir": str(lut_dir),
            },
            "results": results,
        }, f, indent=2, default=str)
    logger.info("\nSummary saved to %s", summary_path)


if __name__ == "__main__":
    main()
