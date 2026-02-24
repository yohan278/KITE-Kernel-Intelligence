#!/usr/bin/env python3
"""Full end-to-end pipeline: Profile → LUT → Simulate → Search.

Runs Pipelines #1, #1b, #2, #3 for Qwen3 models on A100-80GB.

Usage:
    python run_full_pipeline.py --output-dir data/full_pipeline --duration-s 60.0
    python run_full_pipeline.py --output-dir data/full_pipeline --skip-profiling
    python run_full_pipeline.py --output-dir data/full_pipeline --models qwen3-0.6b,qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("full_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARDWARE_KEY = "a100_80gb"
PRECISION = "fp16"

# Models to profile (keys into _EXAMPLE_MODELS in inference_search.cli)
ALL_MODEL_KEYS = [
    "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "qwen3-30b-a3b", "glm-4.7-flash",
]
# All models need fresh profiling for e2e_v5
MODELS_TO_PROFILE = list(ALL_MODEL_KEYS)

WORKLOAD_TYPES = ["chat", "reasoning", "rag", "agentic"]
PROFILERS = ["token_ops", "attention", "agentic", "sampling", "communication", "cpu_host"]

# Paths to existing workload profiles
SYNTHETIC_PROFILE_DIR = Path("data/workload_profiles")
ACTIVE_PROFILE_DIR = Path("data/active_characterization/workload_profiles")

# Synthetic profile filenames (dataset key -> filename)
SYNTHETIC_PROFILE_MAP = {
    "chat": "wildchat_profile.json",
    "reasoning": "openthoughts_profile.json",
    "rag": "hotpotqa_profile.json",
    "agentic": "agentdata_profile.json",
}
# Active profile filenames
ACTIVE_PROFILE_MAP = {
    "chat": "chat_active_profile.json",
    "reasoning": "reasoning_active_profile.json",
    "rag": "rag_active_profile.json",
    "agentic": "agentic_active_profile.json",
}

# 5 search profiles: (name, SLA constraints, optimization targets)
SEARCH_PROFILES = {
    "latency": {
        "constraints": [
            ("ttft_p95", 2.0, "max"),
            ("tbt_p95", 0.1, "max"),
            ("e2e_p95", 30.0, "max"),
        ],
        "targets": ["e2e_p50", "ttft_p50"],
    },
    "throughput": {
        "constraints": [
            ("throughput_tps", 100.0, "min"),
        ],
        "targets": ["throughput_tps"],
    },
    "ipj": {
        "constraints": [
            ("ttft_p95", 5.0, "max"),
        ],
        "targets": ["ipj"],
    },
    "ipw": {
        "constraints": [
            ("ttft_p95", 5.0, "max"),
        ],
        "targets": ["ipw"],
    },
    "cost": {
        "constraints": [
            ("ttft_p95", 5.0, "max"),
            ("throughput_tps", 50.0, "min"),
        ],
        "targets": ["cost_per_query_usd"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_specs() -> Dict[str, Any]:
    """Return model specs from the inference_search registry."""
    from inference_search.cli import _EXAMPLE_MODELS
    return _EXAMPLE_MODELS


def get_model_slug(model_id: str) -> str:
    """Convert model_id to filesystem-safe slug."""
    return model_id.replace("/", "_")


def load_lut_bundle(lut_dir: Path, model_id: str = ""):
    """Load a LUTBundle from a directory."""
    from inference_simulator.types.lut_bundle import LUTBundle

    def _find(name: str) -> Optional[Path]:
        p = lut_dir / name
        return p if p.exists() else None

    bundle = LUTBundle(
        base_dir=lut_dir,
        model_id=model_id,
        hardware_id=HARDWARE_KEY,
        quantization=PRECISION,
        gpu_token_ops_lut=_find("gpu_token_ops.npz") or _find("token_ops.npz"),
        gpu_attention_prefill_lut=(
            _find("gpu_attention_prefill.npz") or _find("attention_prefill.npz")
        ),
        gpu_attention_decode_lut=(
            _find("gpu_attention_decode.npz") or _find("attention_decode.npz")
        ),
        gpu_moe_lut=_find("gpu_moe.npz") or _find("moe.npz"),
        network_lut=_find("network.npz"),
        energy_lut=_find("energy.npz"),
        tool_distributions=_find("tool_distributions.pkl"),
    )
    if not bundle.exists():
        raise FileNotFoundError(f"LUT bundle at {lut_dir} missing required files")
    return bundle


# ---------------------------------------------------------------------------
# Stage 1: Operator Profiling
# ---------------------------------------------------------------------------

def run_profiling(
    model_keys: List[str],
    output_dir: Path,
    use_energy: bool = True,
) -> Dict[str, Path]:
    """Profile models and return {model_key: profiling_output_dir}.

    Uses ProfilingRunner with default SweepConfig. Profilers generate
    analytical measurements from ModelSpec dimensions (no live GPU needed).

    Args:
        model_keys: Models to profile.
        output_dir: Root output directory.
        use_energy: Collect energy/power measurements via NVML.
    """
    from dataset_generator.profiler.runner import ProfilingRunner
    from dataset_generator.profiler.sweep import SweepConfig
    from inference_simulator.types import HardwareSpec

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    profile_dirs: Dict[str, Path] = {}

    for key in model_keys:
        model_spec = model_specs[key]
        slug = get_model_slug(model_spec.model_id)
        logger.info("=== Stage 1: Profiling %s (energy=%s) ===", model_spec.model_id, use_energy)

        runner = ProfilingRunner(
            model_spec=model_spec,
            hardware_spec=hw_spec,
            sweep_config=SweepConfig(use_energy=use_energy),
            output_dir=output_dir / "profiles",
            precision=PRECISION,
        )
        result = runner.run(profilers=PROFILERS)

        profile_dirs[key] = runner.output_dir
        logger.info(
            "  %s: %d measurements → %s",
            key, len(result.measurements), runner.output_dir,
        )

    return profile_dirs


# ---------------------------------------------------------------------------
# Stage 1b: LUT Generation
# ---------------------------------------------------------------------------

def run_lut_generation(
    model_keys: List[str],
    profile_dirs: Dict[str, Path],
    output_dir: Path,
) -> Dict[str, Path]:
    """Train estimators and generate LUT bundles per model.

    Returns {model_key: lut_dir}.
    """
    from inference_simulator.estimator.lut_generator import LUTGenerator
    from inference_simulator.types import HardwareSpec

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    generator = LUTGenerator()
    lut_dirs: Dict[str, Path] = {}

    for key in model_keys:
        model_spec = model_specs[key]
        slug = get_model_slug(model_spec.model_id)
        lut_dir = output_dir / "luts" / slug

        if key not in profile_dirs:
            logger.warning("No profiling data for %s, skipping LUT generation", key)
            continue

        logger.info("=== Stage 1b: LUT generation for %s ===", model_spec.model_id)
        bundle = generator.generate_full_bundle(
            profiling_dir=profile_dirs[key],
            output_dir=lut_dir,
            model_spec=model_spec,
            hw_spec=hw_spec,
        )
        lut_dirs[key] = lut_dir
        logger.info("  %s: LUTs → %s (exists=%s)", key, lut_dir, bundle.exists())

    return lut_dirs


# ---------------------------------------------------------------------------
# Stage 3: Simulation
# ---------------------------------------------------------------------------

def run_simulations(
    model_keys: List[str],
    lut_dirs: Dict[str, Path],
    output_dir: Path,
    duration_s: float,
) -> List[Dict[str, Any]]:
    """Run EventDrivenSimulator for each model × workload × profile source."""
    from inference_simulator.engine.simulator import EventDrivenSimulator
    from inference_simulator.scheduler.vllm import VLLMScheduler
    from inference_simulator.types import HardwareSpec, InferenceSpec, WorkloadSpec
    from inference_simulator.types.workload_profile import WorkloadProfile

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    sim_dir = output_dir / "simulation_results"
    sim_dir.mkdir(parents=True, exist_ok=True)

    workload_factories = {
        "chat": WorkloadSpec.for_chat,
        "reasoning": WorkloadSpec.for_reasoning,
        "rag": WorkloadSpec.for_rag,
        "agentic": WorkloadSpec.for_agentic,
    }

    all_results: List[Dict[str, Any]] = []

    for key in model_keys:
        if key not in lut_dirs:
            logger.warning("No LUTs for %s, skipping simulation", key)
            continue

        model_spec = model_specs[key]
        lut_bundle = load_lut_bundle(lut_dirs[key], model_spec.model_id)
        inference_spec = InferenceSpec(precision=PRECISION)

        for wt in WORKLOAD_TYPES:
            workload = workload_factories[wt](qps=1.0)

            # Run with both synthetic and active profiles
            for source, profile_dir, profile_map in [
                ("synthetic", SYNTHETIC_PROFILE_DIR, SYNTHETIC_PROFILE_MAP),
                ("active", ACTIVE_PROFILE_DIR, ACTIVE_PROFILE_MAP),
            ]:
                profile_path = profile_dir / profile_map[wt]
                if not profile_path.exists():
                    logger.warning("Profile %s not found, skipping", profile_path)
                    continue

                profile = WorkloadProfile.load(profile_path)
                logger.info(
                    "  Simulating %s / %s / %s (%d samples)...",
                    key, wt, source, profile.n_samples,
                )

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

                metrics = sim.run(
                    workload, duration_s=duration_s, seed=42,
                    workload_profile=profile,
                )

                result_entry = {
                    "model": model_spec.model_id,
                    "model_key": key,
                    "workload": wt,
                    "source": source,
                    "metrics": asdict(metrics),
                }
                all_results.append(result_entry)

                # Save individual result
                slug = get_model_slug(model_spec.model_id)
                result_path = sim_dir / f"{slug}_{wt}_{source}.json"
                with open(result_path, "w") as f:
                    json.dump(result_entry, f, indent=2, default=str)

                logger.info(
                    "    %d requests, %.1f tok/s, TTFT_p50=%.3fs",
                    metrics.total_requests,
                    metrics.throughput_tps,
                    metrics.ttft_p50,
                )

    return all_results


# ---------------------------------------------------------------------------
# Stage 4: Configuration Search
# ---------------------------------------------------------------------------

def run_searches(
    model_keys: List[str],
    lut_dirs: Dict[str, Path],
    output_dir: Path,
    duration_s: float,
) -> List[Dict[str, Any]]:
    """Run Pipeline #3 search for each model × workload × SLA profile."""
    from inference_search.cli import run_search
    from inference_search.ml_oracle import MLBackedOracle
    from inference_search.types import SLAConstraint, SearchConfig
    from inference_simulator.types import HardwareSpec, InferenceSpec, WorkloadSpec

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    search_dir = output_dir / "search_results"
    search_dir.mkdir(parents=True, exist_ok=True)

    workload_factories = {
        "chat": WorkloadSpec.for_chat,
        "reasoning": WorkloadSpec.for_reasoning,
        "rag": WorkloadSpec.for_rag,
        "agentic": WorkloadSpec.for_agentic,
    }

    # 4 inference spec variants (1-GPU and 4-GPU)
    inference_specs = [
        InferenceSpec(num_gpus=1, tensor_parallel=1, precision="fp16", max_batch_size=64),
        InferenceSpec(num_gpus=1, tensor_parallel=1, precision="bf16", max_batch_size=64),
        InferenceSpec(num_gpus=4, tensor_parallel=4, precision="fp16", max_batch_size=64),
        InferenceSpec(num_gpus=4, tensor_parallel=4, precision="bf16", max_batch_size=64),
    ]

    # Build model spec list for search (only models with LUTs)
    search_model_specs = [
        model_specs[key] for key in model_keys if key in lut_dirs
    ]
    if not search_model_specs:
        logger.error("No models have LUTs; cannot run search")
        return []

    # For MLBackedOracle, we use the first available LUT dir. The oracle
    # uses roofline fallback per-model when model-specific LUTs don't exist,
    # but for this pipeline each model has its own LUTs. We create one oracle
    # per model for accuracy.
    all_search_results: List[Dict[str, Any]] = []

    for wt in WORKLOAD_TYPES:
        workload = workload_factories[wt](qps=1.0)

        for profile_name, profile_cfg in SEARCH_PROFILES.items():
            sla_constraints = [
                SLAConstraint(metric, threshold, direction)
                for metric, threshold, direction in profile_cfg["constraints"]
            ]

            logger.info(
                "=== Stage 4: Search %s / %s ===",
                wt, profile_name,
            )

            # Run search per model (each has its own oracle/LUT)
            for key in model_keys:
                if key not in lut_dirs:
                    continue

                model_spec = model_specs[key]
                oracle = MLBackedOracle(
                    lut_bundle_dir=lut_dirs[key],
                    accuracy_score=0.85,
                    price_per_hour_usd=3.50,
                )

                config = SearchConfig(
                    model_specs=[model_spec],
                    hardware_specs=[hw_spec],
                    inference_specs=inference_specs,
                    workload_spec=workload,
                    sla_constraints=sla_constraints,
                    optimization_targets=profile_cfg["targets"],
                    duration_s=duration_s,
                    search_method="exhaustive",
                    accuracy_score=0.85,
                    price_per_gpu_hour_usd=3.50,
                )

                result = run_search(config, oracle=oracle)

                slug = get_model_slug(model_spec.model_id)
                result_entry = {
                    "model": model_spec.model_id,
                    "model_key": key,
                    "workload": wt,
                    "profile": profile_name,
                    "elapsed_seconds": result.elapsed_seconds,
                    "total_simulations": result.total_simulations,
                    "num_configs": len(result.all_results),
                    "pareto_size": len(result.pareto_frontier),
                    "configs": [
                        {
                            "model_id": r.model_spec.model_id,
                            "num_gpus": r.inference_spec.num_gpus,
                            "max_batch_size": r.inference_spec.max_batch_size,
                            "max_qps": r.max_qps,
                            "metrics": r.metrics,
                            "sla_violations": r.sla_violations,
                        }
                        for r in result.all_results
                    ],
                    "pareto_frontier": [
                        {
                            "model_id": r.model_spec.model_id,
                            "num_gpus": r.inference_spec.num_gpus,
                            "max_batch_size": r.inference_spec.max_batch_size,
                            "max_qps": r.max_qps,
                            "metrics": r.metrics,
                        }
                        for r in result.pareto_frontier
                    ],
                }
                all_search_results.append(result_entry)

                result_path = search_dir / f"{slug}_{wt}_{profile_name}.json"
                with open(result_path, "w") as f:
                    json.dump(result_entry, f, indent=2, default=str)

                pareto_qps = [r.max_qps for r in result.pareto_frontier]
                logger.info(
                    "  %s / %s / %s: %d configs, %d Pareto, best QPS=%.1f",
                    key, wt, profile_name,
                    len(result.all_results),
                    len(result.pareto_frontier),
                    max(pareto_qps) if pareto_qps else 0.0,
                )

    return all_search_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    sim_results: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> None:
    """Print comparison tables."""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)

    header = (
        f"{'Model':<20} {'Workload':<12} {'Source':<10} "
        f"{'Reqs':>6} {'Tok/s':>8} {'TTFT_p50':>9} {'E2E_p50':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in sim_results:
        m = r["metrics"]
        print(
            f"{r['model_key']:<20} {r['workload']:<12} {r['source']:<10} "
            f"{m['total_requests']:>6} "
            f"{m['throughput_tps']:>8.0f} "
            f"{m['ttft_p50']:>8.3f}s "
            f"{m['e2e_p50']:>8.3f}s"
        )

    print("\n" + "=" * 80)
    print("SEARCH RESULTS (Pareto Frontier)")
    print("=" * 80)

    header = (
        f"{'Model':<16} {'Workload':<12} {'Profile':<12} "
        f"{'#Cfg':>5} {'Pareto':>6} {'Best QPS':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in search_results:
        pareto_qps = [p["max_qps"] for p in r["pareto_frontier"]]
        print(
            f"{r['model_key']:<16} {r['workload']:<12} {r['profile']:<12} "
            f"{r['num_configs']:>5} {r['pareto_size']:>6} "
            f"{max(pareto_qps) if pareto_qps else 0:>8.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full end-to-end pipeline: Profile → LUT → Simulate → Search"
    )
    parser.add_argument(
        "--output-dir", default="data/full_pipeline",
        help="Root output directory (default: data/full_pipeline)",
    )
    parser.add_argument(
        "--duration-s", type=float, default=60.0,
        help="Simulation duration per run in seconds (default: 60)",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model keys to run (default: all 4 Qwen3 sizes)",
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip Stage 1 profiling (reuse existing CSVs)",
    )
    parser.add_argument(
        "--skip-lut", action="store_true",
        help="Skip Stage 1b LUT generation (reuse existing LUTs)",
    )
    parser.add_argument(
        "--skip-simulation", action="store_true",
        help="Skip Stage 3 simulation",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip Stage 4 search",
    )
    parser.add_argument(
        "--existing-lut-dir", default=None,
        help="Use existing LUT dir for Qwen3-8B (default: data/luts)",
    )
    parser.add_argument(
        "--use-energy", action="store_true", default=True,
        help="Collect energy measurements during profiling (default: True)",
    )
    parser.add_argument(
        "--no-energy", action="store_true",
        help="Disable energy measurement during profiling",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse model selection
    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
    else:
        model_keys = list(ALL_MODEL_KEYS)

    models_to_profile = [k for k in model_keys if k in MODELS_TO_PROFILE]

    logger.info("=" * 60)
    logger.info("Full Pipeline Run")
    logger.info("=" * 60)
    logger.info("Models: %s", model_keys)
    logger.info("Models to profile: %s", models_to_profile)
    logger.info("Output: %s", output_dir)
    logger.info("Duration: %.1fs", args.duration_s)
    logger.info("")

    use_energy = args.use_energy and not args.no_energy
    start_time = time.time()

    # ----- Stage 1: Profiling -----
    profile_dirs: Dict[str, Path] = {}
    if not args.skip_profiling and models_to_profile:
        t0 = time.time()
        profile_dirs = run_profiling(models_to_profile, output_dir, use_energy=use_energy)
        logger.info("Stage 1 complete: %.1fs", time.time() - t0)
    elif args.skip_profiling:
        logger.info("Stage 1: SKIPPED (--skip-profiling)")
        # Try to find existing profile dirs
        model_specs = get_model_specs()
        for key in models_to_profile:
            slug = get_model_slug(model_specs[key].model_id)
            hw_slug = "nvidia_a100_80gb_sxm"
            candidate = output_dir / "profiles" / slug / hw_slug / PRECISION
            if candidate.exists():
                profile_dirs[key] = candidate
                logger.info("  Found existing profiles: %s", candidate)

    # ----- Stage 1b: LUT Generation -----
    lut_dirs: Dict[str, Path] = {}

    # Add existing Qwen3-8B LUTs if available
    if "qwen3-8b" in model_keys:
        existing_lut = Path(args.existing_lut_dir or "data/luts")
        if existing_lut.exists():
            lut_dirs["qwen3-8b"] = existing_lut
            logger.info("Using existing Qwen3-8B LUTs: %s", existing_lut)

    if not args.skip_lut and profile_dirs:
        t0 = time.time()
        new_lut_dirs = run_lut_generation(models_to_profile, profile_dirs, output_dir)
        lut_dirs.update(new_lut_dirs)
        logger.info("Stage 1b complete: %.1fs", time.time() - t0)
    elif args.skip_lut:
        logger.info("Stage 1b: SKIPPED (--skip-lut)")
        # Try to find existing LUT dirs
        model_specs = get_model_specs()
        for key in models_to_profile:
            slug = get_model_slug(model_specs[key].model_id)
            candidate = output_dir / "luts" / slug
            if candidate.exists():
                lut_dirs[key] = candidate
                logger.info("  Found existing LUTs: %s", candidate)

    logger.info("LUT dirs available: %s", {k: str(v) for k, v in lut_dirs.items()})

    # ----- Stage 3: Simulation -----
    sim_results: List[Dict[str, Any]] = []
    if not args.skip_simulation:
        t0 = time.time()
        sim_results = run_simulations(model_keys, lut_dirs, output_dir, args.duration_s)
        logger.info("Stage 3 complete: %.1fs (%d results)", time.time() - t0, len(sim_results))
    else:
        logger.info("Stage 3: SKIPPED (--skip-simulation)")

    # ----- Stage 4: Search -----
    search_results: List[Dict[str, Any]] = []
    if not args.skip_search:
        t0 = time.time()
        search_results = run_searches(model_keys, lut_dirs, output_dir, args.duration_s)
        logger.info("Stage 4 complete: %.1fs (%d results)", time.time() - t0, len(search_results))
    else:
        logger.info("Stage 4: SKIPPED (--skip-search)")

    # ----- Summary -----
    elapsed = time.time() - start_time

    if sim_results or search_results:
        print_summary(sim_results, search_results)

    summary = {
        "config": {
            "models": model_keys,
            "hardware": HARDWARE_KEY,
            "precision": PRECISION,
            "duration_s": args.duration_s,
            "output_dir": str(output_dir),
        },
        "timing": {
            "total_seconds": round(elapsed, 1),
        },
        "stages": {
            "profiling": {
                "models_profiled": list(profile_dirs.keys()),
                "dirs": {k: str(v) for k, v in profile_dirs.items()},
            },
            "lut_generation": {
                "models": list(lut_dirs.keys()),
                "dirs": {k: str(v) for k, v in lut_dirs.items()},
            },
            "simulation": {
                "num_results": len(sim_results),
            },
            "search": {
                "num_results": len(search_results),
                "profiles": list(SEARCH_PROFILES.keys()),
            },
        },
        "simulation_results": sim_results,
        "search_results": search_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("\nSummary saved to %s", summary_path)
    logger.info("Total elapsed: %.1fs", elapsed)


if __name__ == "__main__":
    main()
