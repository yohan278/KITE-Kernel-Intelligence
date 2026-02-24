#!/usr/bin/env python3
"""E2E Pipeline Validation: Profile → LUT → Simulate → Search.

Self-contained orchestrator for validating the simulator against real vLLM.
Profiles selected models across 6 inference configs, 4 workloads,
6 QPS levels, and 3 search SLA objectives. Holdout models are only used
for simulation (roofline fallback) and search, not profiling.

Usage:
    python -m evals.scripts.run_e2e_validation --output-dir data/e2e_v2
    python -m evals.scripts.run_e2e_validation --skip-profiling --skip-benchmark
    python -m evals.scripts.run_e2e_validation --models qwen3-0.6b,qwen3-4b
    python -m evals.scripts.run_e2e_validation --cuda-devices 0,1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("e2e_validation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("data/e2e_v2")
HARDWARE_KEY = "a100_80gb"

PROFILED_MODELS = [
    "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "qwen3-30b-a3b", "glm-4.7-flash",
]
HOLDOUT_MODELS = []
ALL_MODELS = PROFILED_MODELS + HOLDOUT_MODELS

WORKLOAD_TYPES = ["chat", "reasoning", "rag", "agentic"]
QPS_LEVELS = [1, 2, 5, 10, 20, 50]
PROFILERS = ["token_ops", "attention", "agentic", "sampling", "communication", "cpu_host"]

INFERENCE_CONFIGS = {
    "1gpu-fp16": {"num_gpus": 1, "tensor_parallel": 1, "precision": "fp16"},
    "1gpu-bf16": {"num_gpus": 1, "tensor_parallel": 1, "precision": "bf16"},
    "4gpu-fp16": {"num_gpus": 4, "tensor_parallel": 4, "precision": "fp16"},
    "4gpu-bf16": {"num_gpus": 4, "tensor_parallel": 4, "precision": "bf16"},
    "8gpu-fp16": {"num_gpus": 8, "tensor_parallel": 8, "precision": "fp16"},
    "8gpu-bf16": {"num_gpus": 8, "tensor_parallel": 8, "precision": "bf16"},
}

SEARCH_PROFILES = {
    "max-qps": {
        "constraints": [("ttft_p95", 5.0, "max"), ("tbt_p95", 0.2, "max")],
        "targets": ["throughput_tps"],
    },
    "min-latency": {
        "constraints": [
            ("ttft_p95", 1.0, "max"),
            ("tbt_p95", 0.05, "max"),
            ("e2e_p95", 10.0, "max"),
        ],
        "targets": ["e2e_p50", "ttft_p50"],
    },
    "min-energy": {
        "constraints": [
            ("ttft_p95", 5.0, "max"),
            ("throughput_tps", 10.0, "min"),
        ],
        "targets": ["ipj"],
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


def make_inference_spec(config_id: str):
    """Build an InferenceSpec from config dict."""
    from inference_simulator.types import InferenceSpec

    cfg = INFERENCE_CONFIGS[config_id]
    return InferenceSpec(
        num_gpus=cfg["num_gpus"],
        tensor_parallel=cfg["tensor_parallel"],
        precision=cfg["precision"],
        engine_config=cfg.get("engine_config", {}),
    )


def load_lut_bundle(
    lut_dir: Path,
    model_id: str = "",
    hardware_id: str = HARDWARE_KEY,
    quantization: str = "fp16",
):
    """Load a LUTBundle from a directory."""
    from inference_simulator.types.lut_bundle import LUTBundle

    def _find(name: str) -> Optional[Path]:
        p = lut_dir / name
        return p if p.exists() else None

    token_ops = _find("gpu_token_ops.npz") or _find("token_ops.npz")
    attn_prefill = _find("gpu_attention_prefill.npz") or _find("attention_prefill.npz")
    attn_decode = _find("gpu_attention_decode.npz") or _find("attention_decode.npz")

    # All three required LUTs must exist
    if not all([token_ops, attn_prefill, attn_decode]):
        raise FileNotFoundError(
            f"LUT bundle at {lut_dir} missing required .npz files "
            f"(token_ops={token_ops}, prefill={attn_prefill}, decode={attn_decode})"
        )

    bundle = LUTBundle(
        base_dir=lut_dir,
        model_id=model_id,
        hardware_id=hardware_id,
        quantization=quantization,
        gpu_token_ops_lut=token_ops,
        gpu_attention_prefill_lut=attn_prefill,
        gpu_attention_decode_lut=attn_decode,
        gpu_moe_lut=_find("gpu_moe.npz") or _find("moe.npz"),
        gpu_ssm_lut=_find("gpu_ssm.npz") or _find("ssm.npz"),
        network_lut=_find("network.npz"),
        cpu_ops_lut=_find("cpu_host.npz"),
        energy_lut=_find("energy.npz"),
        tool_distributions=_find("tool_distributions.pkl"),
    )
    return bundle


def get_workload(workload_type: str, qps: float):
    """Build a WorkloadSpec for a given workload type and QPS."""
    from inference_simulator.types import WorkloadSpec

    factories = {
        "chat": WorkloadSpec.for_chat,
        "reasoning": WorkloadSpec.for_reasoning,
        "rag": WorkloadSpec.for_rag,
        "agentic": WorkloadSpec.for_agentic,
    }
    return factories[workload_type](qps=qps)


def metrics_to_dict(metrics) -> Dict[str, Any]:
    """Convert SimulationMetrics to a JSON-serializable dict."""
    return asdict(metrics)


# ---------------------------------------------------------------------------
# Stage 1: Operator Profiling
# ---------------------------------------------------------------------------

def stage_profile(
    model_keys: List[str],
    output_dir: Path,
    profiling_mode: str = "unfused",
    use_energy: bool = True,
) -> Dict[str, Path]:
    """Profile models and return {model_key: profiling_output_dir}.

    Profiles all models passed in model_keys. Holdout-only filtering
    is handled by the caller (main) -- any model passed here gets profiled.
    MoE models automatically get the "moe" profiler added.

    Args:
        model_keys: Models to profile.
        output_dir: Root output directory.
        profiling_mode: "unfused", "fused", or "both".
        use_energy: Collect energy/power measurements via NVML.
    """
    from dataset_generator.profiler.runner import ProfilingRunner
    from dataset_generator.profiler.sweep import SweepConfig
    from inference_simulator.types import HardwareSpec
    from inference_simulator.types.model_spec import ArchitectureType

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    profile_dirs: Dict[str, Path] = {}

    sweep = SweepConfig(use_energy=use_energy, gpu_topologies=[1, 4, 8])

    for key in model_keys:
        model_spec = model_specs[key]
        logger.info("=== Stage 1: Profiling %s (mode=%s, energy=%s) ===",
                     model_spec.model_id, profiling_mode, use_energy)

        # Build profiler list: base profilers + conditional MoE
        profilers = list(PROFILERS)
        if model_spec.architecture_type == ArchitectureType.MOE_TRANSFORMER:
            profilers.append("moe")
            logger.info("  MoE architecture detected, adding 'moe' profiler")

        # When "both" mode: run unfused first, then fused in a subdirectory
        if profiling_mode == "both":
            # Unfused pass
            unfused_sweep = SweepConfig(
                use_energy=use_energy, gpu_topologies=[1, 4, 8],
            )
            unfused_runner = ProfilingRunner(
                model_spec=model_spec,
                hardware_spec=hw_spec,
                sweep_config=unfused_sweep,
                output_dir=output_dir / "profiles" / "unfused",
                precision="fp16",
            )
            unfused_result = unfused_runner.run(profilers=profilers)
            logger.info("  %s unfused: %d measurements", key, len(unfused_result.measurements))

            # Fused pass (add vllm_engine profiler)
            fused_sweep = SweepConfig(
                use_energy=use_energy, gpu_topologies=[1, 4, 8],
            )
            fused_runner = ProfilingRunner(
                model_spec=model_spec,
                hardware_spec=hw_spec,
                sweep_config=fused_sweep,
                output_dir=output_dir / "profiles" / "fused",
                precision="fp16",
            )
            fused_profilers = list(profilers) + ["vllm_engine"]
            fused_result = fused_runner.run(profilers=fused_profilers)
            logger.info("  %s fused: %d measurements", key, len(fused_result.measurements))

            # Use unfused profiles as the primary (compatible with LUT generation)
            profile_dirs[key] = unfused_runner.output_dir
        else:
            active_profilers = list(profilers)
            if profiling_mode == "fused":
                active_profilers.append("vllm_engine")

            runner = ProfilingRunner(
                model_spec=model_spec,
                hardware_spec=hw_spec,
                sweep_config=sweep,
                output_dir=output_dir / "profiles",
                precision="fp16",
            )
            result = runner.run(profilers=active_profilers)

            profile_dirs[key] = runner.output_dir
            logger.info(
                "  %s: %d measurements -> %s",
                key, len(result.measurements), runner.output_dir,
            )

    return profile_dirs


# ---------------------------------------------------------------------------
# Stage 1b: LUT Generation
# ---------------------------------------------------------------------------

def stage_lut_generation(
    model_keys: List[str],
    profile_dirs: Dict[str, Path],
    output_dir: Path,
    sim_config: Optional[object] = None,
) -> Dict[str, Path]:
    """Generate LUT bundles from profiling CSVs.

    Returns {model_key: lut_dir}. Skips models without profiling data.

    Args:
        model_keys: Models to generate LUTs for.
        profile_dirs: {model_key: profiling_csv_directory}.
        output_dir: Root output directory.
        sim_config: Optional SimulatorConfig for TP scaling and other overrides.
    """
    from inference_simulator.estimator.lut_generator import LUTGenerator
    from inference_simulator.types import HardwareSpec

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    generator = LUTGenerator()
    lut_dirs: Dict[str, Path] = {}

    for key in model_keys:
        if key not in profile_dirs:
            logger.warning("No profiling data for %s, skipping LUT generation", key)
            continue

        model_spec = model_specs[key]
        slug = get_model_slug(model_spec.model_id)
        lut_dir = output_dir / "luts" / slug

        logger.info("=== Stage 1b: LUT generation for %s ===", model_spec.model_id)
        bundle = generator.generate_full_bundle(
            profiling_dir=profile_dirs[key],
            output_dir=lut_dir,
            model_spec=model_spec,
            hw_spec=hw_spec,
        )
        lut_dirs[key] = lut_dir
        logger.info("  %s: LUTs -> %s (exists=%s)", key, lut_dir, bundle.exists())

    return lut_dirs


# ---------------------------------------------------------------------------
# Stage 2: Simulation
# ---------------------------------------------------------------------------

def stage_simulation(
    model_keys: List[str],
    lut_dirs: Dict[str, Path],
    profile_dirs: Dict[str, Path],
    output_dir: Path,
    duration_s: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run EventDrivenSimulator for each model x workload x config x QPS level.

    Profiled models use LUT bundles; holdout models use roofline fallback
    (lut_bundle=None).  When profiling CSVs are available, a
    PerOperatorPowerModel is fitted and passed to the simulator for
    load-dependent power estimation.
    """
    from inference_simulator.engine.simulator import EventDrivenSimulator
    from inference_simulator.scheduler.vllm import VLLMScheduler
    from inference_simulator.types import HardwareSpec

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    sim_dir = output_dir / "simulation"

    all_results: List[Dict[str, Any]] = []
    total_runs = len(model_keys) * len(WORKLOAD_TYPES) * len(INFERENCE_CONFIGS) * len(QPS_LEVELS)
    run_idx = 0

    for key in model_keys:
        model_spec = model_specs[key]
        is_holdout = key in HOLDOUT_MODELS

        # Load LUT bundle for profiled models, None for holdout (roofline fallback)
        lut_bundle = None
        if not is_holdout and key in lut_dirs:
            try:
                lut_bundle = load_lut_bundle(lut_dirs[key], model_spec.model_id)
                logger.info("  Loaded LUT bundle for %s", key)
            except FileNotFoundError as e:
                logger.warning("LUT bundle not found for %s, using roofline: %s", key, e)

        # Fit per-operator power model from profiled energy CSVs
        power_model = None
        if not is_holdout and key in profile_dirs:
            try:
                from inference_simulator.energy.power_model import PerOperatorPowerModel
                from inference_simulator.estimator.sklearn_base import load_csv_measurements
                from inference_simulator.types.operators import OperatorCategory

                csv_category_map = {
                    "linear": OperatorCategory.LINEAR,
                    "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
                    "attention_decode": OperatorCategory.ATTENTION_DECODE,
                    "embedding": OperatorCategory.EMBEDDING,
                    "normalization": OperatorCategory.NORMALIZATION,
                    "activation": OperatorCategory.ACTIVATION,
                }
                all_measurements = []
                for name, cat in csv_category_map.items():
                    csv_file = profile_dirs[key] / f"{name}.csv"
                    if csv_file.exists():
                        all_measurements.extend(load_csv_measurements(csv_file, cat))
                # Also load from combined CSVs (token_ops, attention, sampling, etc.)
                # Skip CSVs that don't have operator measurement schema
                _skip_csv_stems = {
                    "communication", "agentic", "cpu_host",
                    *csv_category_map,
                }
                for csv_file in profile_dirs[key].glob("*.csv"):
                    if csv_file.stem.lower() not in _skip_csv_stems:
                        all_measurements.extend(
                            load_csv_measurements(csv_file, OperatorCategory.LINEAR)
                        )

                has_power = any(
                    m.power_w is not None and m.power_w > 0
                    for m in all_measurements
                )
                if has_power:
                    pm = PerOperatorPowerModel()
                    pm.fit(all_measurements)
                    power_model = pm
                    logger.info(
                        "  Fitted power model for %s (%d measurements)",
                        key, len(all_measurements),
                    )
            except Exception as e:
                logger.warning("Failed to fit power model for %s: %s", key, e)

        for wt in WORKLOAD_TYPES:
            wt_dir = sim_dir / key / wt
            wt_dir.mkdir(parents=True, exist_ok=True)

            for config_id in INFERENCE_CONFIGS:
                # Skip configs requiring more GPUs than available
                _cfg = INFERENCE_CONFIGS[config_id]
                _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                _visible_gpus = [g for g in _visible.split(",") if g.strip()] if _visible else []
                if _visible_gpus and _cfg["num_gpus"] > len(_visible_gpus):
                    logger.info(
                        "Skipping sim %s: needs %d GPUs, only %d available",
                        config_id, _cfg["num_gpus"], len(_visible_gpus),
                    )
                    run_idx += len(QPS_LEVELS)
                    continue

                inference_spec = make_inference_spec(config_id)
                qps_results: List[Dict[str, Any]] = []

                for qps in QPS_LEVELS:
                    run_idx += 1
                    logger.info(
                        "[%d/%d] Simulating %s / %s / %s / qps=%d%s",
                        run_idx, total_runs, key, wt, config_id, qps,
                        " (roofline)" if lut_bundle is None else "",
                    )

                    workload = get_workload(wt, qps=float(qps))

                    scheduler = VLLMScheduler(
                        max_num_seqs=inference_spec.max_batch_size,
                        max_num_batched_tokens=inference_spec.max_batch_size * 2048,
                    )
                    # Load SimulatorConfig if available
                    sim_config = None
                    try:
                        from inference_simulator.engine.simulator_config import SimulatorConfig
                        config_path = output_dir / "simulator_config.json"
                        if config_path.exists():
                            sim_config = SimulatorConfig.from_json(config_path)
                    except Exception:
                        pass

                    sim = EventDrivenSimulator(
                        model_spec=model_spec,
                        hardware_spec=hw_spec,
                        inference_spec=inference_spec,
                        scheduler=scheduler,
                        lut_bundle=lut_bundle,
                        config=sim_config,
                        power_model=power_model,
                    )

                    metrics = sim.run(
                        workload_spec=workload,
                        duration_s=duration_s,
                        seed=seed,
                    )

                    entry = {
                        "model": model_spec.model_id,
                        "model_key": key,
                        "workload": wt,
                        "config_id": config_id,
                        "qps": qps,
                        "is_holdout": is_holdout,
                        "has_lut": lut_bundle is not None,
                        "metrics": metrics_to_dict(metrics),
                    }
                    qps_results.append(entry)
                    all_results.append(entry)

                # Write per-config JSONL file
                jsonl_path = wt_dir / f"{config_id}.jsonl"
                with open(jsonl_path, "w") as f:
                    for r in qps_results:
                        f.write(json.dumps(r, default=str) + "\n")

    logger.info(
        "Stage 2 complete: %d simulation runs across %d models",
        len(all_results), len(model_keys),
    )
    return all_results


# ---------------------------------------------------------------------------
# Stage 3: Search
# ---------------------------------------------------------------------------

def stage_search(
    model_keys: List[str],
    lut_dirs: Dict[str, Path],
    output_dir: Path,
    duration_s: float,
) -> List[Dict[str, Any]]:
    """Run Pipeline #3 search for each model x workload x SLA objective."""
    from inference_search.cli import run_search
    from inference_search.oracle import RooflineOracle
    from inference_search.types import SLAConstraint, SearchConfig
    from inference_simulator.types import HardwareSpec

    model_specs = get_model_specs()
    hw_spec = HardwareSpec.from_registry(HARDWARE_KEY)
    search_dir = output_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)

    # Build the 6 inference specs for the search space
    inference_specs = [make_inference_spec(cid) for cid in INFERENCE_CONFIGS]

    all_search_results: List[Dict[str, Any]] = []
    total_searches = len(model_keys) * len(WORKLOAD_TYPES) * len(SEARCH_PROFILES)
    search_idx = 0

    for key in model_keys:
        model_spec = model_specs[key]
        is_holdout = key in HOLDOUT_MODELS

        # Use RooflineOracle for search screening (fast analytical model).
        # Detailed LUT-backed simulation results are already in Stage 2.
        # RooflineOracle gives correct relative ordering for config search.
        oracle = RooflineOracle(
            accuracy_score=0.85,
            price_per_hour_usd=3.50,
        )

        for wt in WORKLOAD_TYPES:
            workload = get_workload(wt, qps=1.0)

            for profile_name, profile_cfg in SEARCH_PROFILES.items():
                search_idx += 1
                logger.info(
                    "[%d/%d] Search %s / %s / %s%s",
                    search_idx, total_searches, key, wt, profile_name,
                    " (roofline)" if is_holdout else "",
                )

                sla_constraints = [
                    SLAConstraint(metric, threshold, direction)
                    for metric, threshold, direction in profile_cfg["constraints"]
                ]

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

                result_entry = {
                    "model": model_spec.model_id,
                    "model_key": key,
                    "workload": wt,
                    "objective": profile_name,
                    "is_holdout": is_holdout,
                    "elapsed_seconds": result.elapsed_seconds,
                    "total_simulations": result.total_simulations,
                    "num_configs": len(result.all_results),
                    "pareto_size": len(result.pareto_frontier),
                    "configs": [
                        {
                            "model_id": r.model_spec.model_id,
                            "num_gpus": r.inference_spec.num_gpus,
                            "tensor_parallel": r.inference_spec.tensor_parallel,
                            "precision": r.inference_spec.precision,
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
                            "tensor_parallel": r.inference_spec.tensor_parallel,
                            "precision": r.inference_spec.precision,
                            "max_qps": r.max_qps,
                            "metrics": r.metrics,
                        }
                        for r in result.pareto_frontier
                    ],
                }
                all_search_results.append(result_entry)

                result_path = search_dir / f"{key}_{wt}_{profile_name}.json"
                with open(result_path, "w") as f:
                    json.dump(result_entry, f, indent=2, default=str)

                pareto_qps = [p["max_qps"] for p in result_entry["pareto_frontier"]]
                logger.info(
                    "  %d configs, %d Pareto, best QPS=%.1f",
                    len(result.all_results),
                    len(result.pareto_frontier),
                    max(pareto_qps) if pareto_qps else 0.0,
                )

    logger.info(
        "Stage 3 complete: %d search runs", len(all_search_results),
    )
    return all_search_results


# ---------------------------------------------------------------------------
# Stage 4: Real vLLM Benchmark (optional)
# ---------------------------------------------------------------------------

def stage_benchmark(
    model_keys: List[str],
    output_dir: Path,
    duration_s: float,
    seed: int = 42,
    use_energy: bool = True,
) -> List[Dict[str, Any]]:
    """Run real vLLM benchmark at QPS levels for each model x config.

    Uses ``run_benchmark_at_qps`` from ``run_ground_truth_benchmark`` for
    streaming TTFT/TBT measurement plus optional energy collection.
    Starts a vLLM server, runs Poisson-arrival benchmark, stops server.
    This stage is gated by --run-benchmark.
    """
    import subprocess

    from evals.scripts.run_ground_truth_benchmark import run_benchmark_at_qps

    model_specs = get_model_specs()
    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for key in model_keys:
        model_spec = model_specs[key]
        model_dir = val_dir / key
        model_dir.mkdir(parents=True, exist_ok=True)

        for config_id, cfg in INFERENCE_CONFIGS.items():
            # Skip configs requiring more GPUs than available
            import os as _os
            _visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _visible_gpus = [g for g in _visible.split(",") if g.strip()] if _visible else []
            if _visible_gpus and cfg["num_gpus"] > len(_visible_gpus):
                logger.info(
                    "Skipping %s: needs %d GPUs, only %d available",
                    config_id, cfg["num_gpus"], len(_visible_gpus),
                )
                continue

            logger.info(
                "=== Stage 4: Benchmark %s / %s ===", key, config_id,
            )

            # Build vLLM server launch args
            vllm_args = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_spec.model_id,
                "--tensor-parallel-size", str(cfg["tensor_parallel"]),
                "--dtype", cfg["precision"],
                "--port", "8000",
                "--gpu-memory-utilization", "0.9",
            ]
            if cfg.get("engine_config", {}).get("quantization"):
                vllm_args += ["--quantization", cfg["engine_config"]["quantization"]]

            logger.info("  Starting vLLM: %s", " ".join(vllm_args))

            # Start vLLM server
            server_proc = subprocess.Popen(
                vllm_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to be ready
            import urllib.request
            ready = False
            for attempt in range(120):
                try:
                    urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                    ready = True
                    break
                except Exception:
                    time.sleep(2)

            if not ready:
                logger.error("  vLLM server failed to start for %s/%s", key, config_id)
                server_proc.terminate()
                server_proc.wait()
                continue

            logger.info("  vLLM server ready, running benchmark...")

            # Determine workload type — default to "chat" for Stage 4
            workload = "chat"

            # Run benchmark at each QPS level
            config_results: List[Dict[str, Any]] = []
            for qps in QPS_LEVELS:
                logger.info("    QPS=%d ...", qps)
                try:
                    result = run_benchmark_at_qps(
                        vllm_url="http://localhost:8000",
                        model_name=model_spec.model_id,
                        workload=workload,
                        qps=float(qps),
                        duration_s=duration_s,
                        seed=seed,
                        use_energy=use_energy,
                    )
                    result["config_id"] = config_id
                    result["model_key"] = key
                    config_results.append(result)
                    all_results.append(result)

                    energy_info = ""
                    if result.get("avg_power_w") is not None:
                        energy_info = f", power={result['avg_power_w']:.1f}W"
                    if result.get("energy_per_token_j") is not None:
                        energy_info += f", e/tok={result['energy_per_token_j']:.4f}J"
                    logger.info(
                        "    TTFT p50=%.3fs, TBT p50=%.4fs, throughput=%.1f tps%s",
                        result.get("ttft", {}).get("p50", 0),
                        result.get("tbt", {}).get("p50", 0),
                        result.get("throughput_tps", 0),
                        energy_info,
                    )
                except Exception as e:
                    logger.error("    Benchmark failed at QPS=%d: %s", qps, e)

            # Save per-config results
            jsonl_path = model_dir / f"{config_id}_real.jsonl"
            with open(jsonl_path, "w") as f:
                for r in config_results:
                    f.write(json.dumps(r, default=str) + "\n")

            # Stop vLLM server
            server_proc.terminate()
            server_proc.wait()
            logger.info("  vLLM server stopped")

    return all_results


# ---------------------------------------------------------------------------
# Stage 5: Report Generation
# ---------------------------------------------------------------------------

def stage_report(
    output_dir: Path,
    sim_results: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    benchmark_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Generate validation reports and aggregate summary."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Per-model reports
    model_keys_seen = sorted(set(r["model_key"] for r in sim_results))
    for key in model_keys_seen:
        is_holdout = key in HOLDOUT_MODELS
        model_sim = [r for r in sim_results if r["model_key"] == key]
        model_search = [r for r in search_results if r["model_key"] == key]

        filename = f"holdout_{key.split('-')[1]}_validation.md" if is_holdout else f"{key}_validation.md"
        report_path = reports_dir / filename

        with open(report_path, "w") as f:
            f.write(f"# Validation Report: {key}\n\n")
            f.write(f"**Type**: {'Holdout (roofline only)' if is_holdout else 'Profiled (LUT-backed)'}\n\n")

            # Simulation summary table
            f.write("## Simulation Results\n\n")
            f.write("| Workload | Config | QPS | Throughput | TTFT P50 | E2E P50 | GPU Util | Power (W) | E/Tok (J) |\n")
            f.write("|----------|--------|-----|-----------|----------|---------|----------|-----------|----------|\n")
            for r in model_sim:
                m = r["metrics"]
                energy_j = m.get("total_energy_j", 0)
                total_toks = m.get("total_tokens_generated", 0)
                e_per_tok = energy_j / total_toks if total_toks > 0 else 0
                f.write(
                    f"| {r['workload']} | {r['config_id']} | {r['qps']} | "
                    f"{m['throughput_tps']:.1f} | {m['ttft_p50']:.4f} | "
                    f"{m['e2e_p50']:.4f} | {m['gpu_utilization']:.2f} | "
                    f"{m.get('avg_power_w', 0):.1f} | {e_per_tok:.4f} |\n"
                )

            # Search summary
            f.write("\n## Search Results\n\n")
            f.write("| Workload | Objective | Pareto Size | Best QPS |\n")
            f.write("|----------|-----------|-------------|----------|\n")
            for r in model_search:
                pareto_qps = [p["max_qps"] for p in r["pareto_frontier"]]
                f.write(
                    f"| {r['workload']} | {r['objective']} | "
                    f"{r['pareto_size']} | {max(pareto_qps) if pareto_qps else 0:.1f} |\n"
                )

            # Sim-vs-Real comparison section (if benchmark results available)
            if benchmark_results:
                model_bench = [
                    r for r in benchmark_results
                    if r.get("model_key") == key
                ]
                if model_bench:
                    f.write("\n## Sim vs Real Comparison\n\n")
                    f.write("| Workload | Config | QPS | Metric | Sim | Real | Error% |\n")
                    f.write("|----------|--------|-----|--------|-----|------|--------|\n")
                    for br in model_bench:
                        config_id_b = br.get("config_id", "")
                        qps_b = br.get("qps_target", 0)
                        workload_b = br.get("workload", "chat")
                        # Find matching sim result
                        matching_sim = [
                            s for s in model_sim
                            if s["config_id"] == config_id_b
                            and s["qps"] == int(qps_b)
                            and s["workload"] == workload_b
                        ]
                        if not matching_sim:
                            continue
                        sm = matching_sim[0]["metrics"]
                        # Compare key metrics
                        comparisons = [
                            ("TTFT P50", sm.get("ttft_p50", 0), br.get("ttft", {}).get("p50", 0)),
                            ("TBT P50", sm.get("tbt_p50", 0), br.get("tbt", {}).get("p50", 0)),
                            ("Throughput", sm.get("throughput_tps", 0), br.get("throughput_tps", 0)),
                        ]
                        # Add energy comparison if both sides have it
                        br_power = br.get("avg_power_w")
                        sm_power = sm.get("avg_power_w", 0)
                        if br_power is not None and sm_power > 0:
                            comparisons.append(("Avg Power", sm_power, br_power))
                        br_etok = br.get("energy_per_token_j")
                        sm_total_e = sm.get("total_energy_j", 0)
                        sm_total_t = sm.get("total_tokens_generated", 0)
                        sm_etok = sm_total_e / sm_total_t if sm_total_t > 0 else 0
                        if br_etok is not None and sm_etok > 0:
                            comparisons.append(("E/Token", sm_etok, br_etok))

                        for metric, sv, rv in comparisons:
                            err = ((sv - rv) / rv * 100) if rv != 0 else float("nan")
                            err_str = f"{err:+.1f}%" if not (err != err) else "N/A"
                            f.write(
                                f"| {workload_b} | {config_id_b} | {int(qps_b)} | "
                                f"{metric} | {sv:.4f} | {rv:.4f} | {err_str} |\n"
                            )

        logger.info("Report: %s", report_path)

    # Aggregate summary
    summary_md_path = reports_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# E2E Validation Summary\n\n")
        f.write(f"**Models**: {len(model_keys_seen)} ({len(PROFILED_MODELS)} profiled + {len(HOLDOUT_MODELS)} holdout)\n")
        f.write(f"**Simulation runs**: {len(sim_results)}\n")
        f.write(f"**Search runs**: {len(search_results)}\n\n")

        # Per-model throughput summary
        f.write("## Peak Throughput by Model (chat workload, QPS=50)\n\n")
        f.write("| Model | Config | Throughput (tok/s) | TTFT P95 (s) | Avg Power (W) | E/Token (J) |\n")
        f.write("|-------|--------|-------------------|-------------|---------------|-------------|\n")
        for key in model_keys_seen:
            peak_runs = [
                r for r in sim_results
                if r["model_key"] == key and r["workload"] == "chat" and r["qps"] == 50
            ]
            for r in peak_runs:
                m = r["metrics"]
                energy_j = m.get("total_energy_j", 0)
                total_toks = m.get("total_tokens_generated", 0)
                e_per_tok = energy_j / total_toks if total_toks > 0 else 0
                f.write(
                    f"| {key} | {r['config_id']} | "
                    f"{m['throughput_tps']:.1f} | {m['ttft_p95']:.4f} | "
                    f"{m.get('avg_power_w', 0):.1f} | {e_per_tok:.4f} |\n"
                )

        # Search highlights
        f.write("\n## Search Highlights\n\n")
        for profile_name in SEARCH_PROFILES:
            f.write(f"\n### {profile_name}\n\n")
            profile_results = [r for r in search_results if r["objective"] == profile_name]
            f.write("| Model | Workload | Pareto Size | Best QPS |\n")
            f.write("|-------|----------|-------------|----------|\n")
            for r in profile_results:
                pareto_qps = [p["max_qps"] for p in r["pareto_frontier"]]
                f.write(
                    f"| {r['model_key']} | {r['workload']} | "
                    f"{r['pareto_size']} | {max(pareto_qps) if pareto_qps else 0:.1f} |\n"
                )

    logger.info("Summary report: %s", summary_md_path)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def write_summary_json(
    output_dir: Path,
    model_keys: List[str],
    duration_s: float,
    profile_dirs: Dict[str, Path],
    lut_dirs: Dict[str, Path],
    sim_results: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    benchmark_results: List[Dict[str, Any]],
    elapsed: float,
) -> None:
    """Write machine-readable aggregate JSON."""
    summary = {
        "config": {
            "models": model_keys,
            "profiled_models": [k for k in model_keys if k in PROFILED_MODELS],
            "holdout_models": [k for k in model_keys if k in HOLDOUT_MODELS],
            "hardware": HARDWARE_KEY,
            "inference_configs": list(INFERENCE_CONFIGS.keys()),
            "workloads": WORKLOAD_TYPES,
            "qps_levels": QPS_LEVELS,
            "search_profiles": list(SEARCH_PROFILES.keys()),
            "duration_s": duration_s,
            "output_dir": str(output_dir),
        },
        "timing": {
            "total_seconds": round(elapsed, 1),
        },
        "counts": {
            "simulation_runs": len(sim_results),
            "search_runs": len(search_results),
            "benchmark_runs": len(benchmark_results),
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
        },
        "verification": {
            "lut_bundles_count": len(lut_dirs),
            "simulation_non_zero": sum(
                1 for r in sim_results
                if r["metrics"].get("throughput_tps", 0) > 0
            ),
            "simulation_total": len(sim_results),
            "search_with_pareto": sum(
                1 for r in search_results
                if r["pareto_size"] > 0
            ),
            "search_total": len(search_results),
            "holdout_completed": sum(
                1 for r in sim_results
                if r["is_holdout"]
            ),
        },
    }

    summary_path = output_dir / "e2e_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary JSON: %s", summary_path)


# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------

def print_console_summary(
    sim_results: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> None:
    """Print comparison tables to console."""
    print("\n" + "=" * 90)
    print("SIMULATION RESULTS (sample: QPS=10, chat workload)")
    print("=" * 90)

    header = (
        f"{'Model':<14} {'Config':<12} {'QPS':>4} "
        f"{'Tok/s':>8} {'TTFT_p50':>9} {'E2E_p50':>9} {'GPU%':>6} {'LUT':>4}"
    )
    print(header)
    print("-" * len(header))

    sample_results = [
        r for r in sim_results
        if r["workload"] == "chat" and r["qps"] == 10
    ]
    for r in sorted(sample_results, key=lambda x: (x["model_key"], x["config_id"])):
        m = r["metrics"]
        print(
            f"{r['model_key']:<14} {r['config_id']:<12} {r['qps']:>4} "
            f"{m['throughput_tps']:>8.0f} "
            f"{m['ttft_p50']:>8.3f}s "
            f"{m['e2e_p50']:>8.3f}s "
            f"{m['gpu_utilization']:>5.1%} "
            f"{'Y' if r['has_lut'] else 'N':>4}"
        )

    print("\n" + "=" * 90)
    print("SEARCH RESULTS (Pareto Frontier)")
    print("=" * 90)

    header = (
        f"{'Model':<14} {'Workload':<12} {'Objective':<14} "
        f"{'#Cfg':>5} {'Pareto':>6} {'Best QPS':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(search_results, key=lambda x: (x["model_key"], x["workload"], x["objective"])):
        pareto_qps = [p["max_qps"] for p in r["pareto_frontier"]]
        print(
            f"{r['model_key']:<14} {r['workload']:<12} {r['objective']:<14} "
            f"{r['num_configs']:>5} {r['pareto_size']:>6} "
            f"{max(pareto_qps) if pareto_qps else 0:>8.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2E Pipeline Validation: Profile -> LUT -> Simulate -> Search"
    )
    parser.add_argument(
        "--output-dir", default="data/e2e_v2",
        help="Root output directory (default: data/e2e_v2)",
    )
    parser.add_argument(
        "--duration-s", type=float, default=30.0,
        help="Simulation duration per run in seconds (default: 30)",
    )
    parser.add_argument(
        "--qps-levels", default=None,
        help="Comma-separated QPS levels (default: 1,2,5,10,20,50)",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model keys to profile and evaluate (default: all models)",
    )
    parser.add_argument(
        "--cuda-devices", default=None,
        help="CUDA_VISIBLE_DEVICES value (e.g. '0,1') for GPU selection",
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip Stage 1 profiling (reuse existing profiles)",
    )
    parser.add_argument(
        "--skip-lut", action="store_true",
        help="Skip Stage 1b LUT generation (reuse existing LUTs)",
    )
    parser.add_argument(
        "--skip-simulation", action="store_true",
        help="Skip Stage 2 simulation",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip Stage 3 search",
    )
    parser.add_argument(
        "--run-benchmark", action="store_true",
        help="Run Stage 4 real vLLM benchmarks (requires vLLM + GPUs)",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Explicitly skip Stage 4 benchmark (default behavior)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--profiling-mode", default="unfused",
        choices=["unfused", "fused", "both"],
        help="Profiling mode: unfused (per-operator), fused (vLLM), or both",
    )
    parser.add_argument(
        "--use-energy", action="store_true", default=True,
        help="Collect energy measurements during profiling (default: True)",
    )
    parser.add_argument(
        "--no-energy", action="store_true",
        help="Disable energy measurement during profiling",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to SimulatorConfig JSON (from micro-experiments)",
    )

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES early, before any GPU imports
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        logger.info("CUDA_VISIBLE_DEVICES set to %s", args.cuda_devices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse overrides
    global QPS_LEVELS
    if args.qps_levels:
        QPS_LEVELS = [int(q.strip()) for q in args.qps_levels.split(",")]

    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
    else:
        model_keys = list(ALL_MODELS)

    # Any model passed via --models gets profiled, unless it's in HOLDOUT_MODELS.
    # Holdout models are only used for simulation (roofline) and search.
    profiled_keys = [k for k in model_keys if k not in HOLDOUT_MODELS]
    holdout_keys = [k for k in model_keys if k in HOLDOUT_MODELS]

    logger.info("=" * 60)
    logger.info("E2E Pipeline Validation")
    logger.info("=" * 60)
    logger.info("Models: %s (profiled=%s, holdout=%s)", model_keys, profiled_keys, holdout_keys)
    logger.info("Configs: %s", list(INFERENCE_CONFIGS.keys()))
    logger.info("Workloads: %s", WORKLOAD_TYPES)
    logger.info("QPS levels: %s", QPS_LEVELS)
    logger.info("Search profiles: %s", list(SEARCH_PROFILES.keys()))
    logger.info("Duration: %.1fs, Seed: %d", args.duration_s, args.seed)
    logger.info("Output: %s", output_dir)
    logger.info(
        "Expected: %d sim runs, %d search runs",
        len(model_keys) * len(WORKLOAD_TYPES) * len(INFERENCE_CONFIGS) * len(QPS_LEVELS),
        len(model_keys) * len(WORKLOAD_TYPES) * len(SEARCH_PROFILES),
    )
    logger.info("")

    start_time = time.time()

    # Load SimulatorConfig if provided
    sim_config = None
    if args.config:
        try:
            from inference_simulator.engine.simulator_config import SimulatorConfig
            sim_config = SimulatorConfig.from_json(args.config)
            logger.info("Loaded SimulatorConfig from %s", args.config)
        except Exception as e:
            logger.warning("Failed to load SimulatorConfig: %s", e)

    use_energy = args.use_energy and not args.no_energy

    # ----- Stage 1: Profiling -----
    profile_dirs: Dict[str, Path] = {}
    if not args.skip_profiling and profiled_keys:
        t0 = time.time()
        profile_dirs = stage_profile(
            profiled_keys, output_dir,
            profiling_mode=args.profiling_mode,
            use_energy=use_energy,
        )
        logger.info("Stage 1 complete: %.1fs", time.time() - t0)
    elif args.skip_profiling:
        logger.info("Stage 1: SKIPPED (--skip-profiling)")
        model_specs = get_model_specs()
        for key in profiled_keys:
            slug = get_model_slug(model_specs[key].model_id)
            hw_slug = "nvidia_a100_80gb_sxm"
            candidate = output_dir / "profiles" / slug / hw_slug / "fp16"
            if candidate.exists():
                profile_dirs[key] = candidate
                logger.info("  Found existing profiles: %s", candidate)
            else:
                # Check data/full_pipeline as fallback
                fallback = Path("data/full_pipeline/profiles") / slug / hw_slug / "fp16"
                if fallback.exists():
                    profile_dirs[key] = fallback
                    logger.info("  Found fallback profiles: %s", fallback)

    # ----- Stage 1b: LUT Generation -----
    lut_dirs: Dict[str, Path] = {}
    if not args.skip_lut and profile_dirs:
        t0 = time.time()
        lut_dirs = stage_lut_generation(model_keys, profile_dirs, output_dir, sim_config=sim_config)
        logger.info("Stage 1b complete: %.1fs", time.time() - t0)
    elif args.skip_lut:
        logger.info("Stage 1b: SKIPPED (--skip-lut)")
        model_specs = get_model_specs()
        for key in profiled_keys:
            slug = get_model_slug(model_specs[key].model_id)
            candidate = output_dir / "luts" / slug
            if candidate.exists():
                lut_dirs[key] = candidate
                logger.info("  Found existing LUTs: %s", candidate)
            else:
                fallback = Path("data/full_pipeline/luts") / slug
                if fallback.exists():
                    lut_dirs[key] = fallback
                    logger.info("  Found fallback LUTs: %s", fallback)

    logger.info("LUT dirs available: %s", {k: str(v) for k, v in lut_dirs.items()})

    # ----- Stage 2: Simulation -----
    sim_results: List[Dict[str, Any]] = []
    if not args.skip_simulation:
        t0 = time.time()
        sim_results = stage_simulation(
            model_keys, lut_dirs, profile_dirs, output_dir, args.duration_s, seed=args.seed,
        )
        logger.info("Stage 2 complete: %.1fs (%d results)", time.time() - t0, len(sim_results))
    else:
        logger.info("Stage 2: SKIPPED (--skip-simulation)")
        # Load existing simulation results from disk
        sim_dir = output_dir / "simulation"
        if sim_dir.exists():
            for jsonl_path in sorted(sim_dir.rglob("*.jsonl")):
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            sim_results.append(json.loads(line))
            logger.info("  Loaded %d existing simulation results from disk", len(sim_results))

    # ----- Stage 3: Search -----
    search_results: List[Dict[str, Any]] = []
    if not args.skip_search:
        t0 = time.time()
        search_results = stage_search(
            model_keys, lut_dirs, output_dir, args.duration_s,
        )
        logger.info("Stage 3 complete: %.1fs (%d results)", time.time() - t0, len(search_results))
    else:
        logger.info("Stage 3: SKIPPED (--skip-search)")
        # Load existing search results from disk
        search_dir = output_dir / "search"
        if search_dir.exists():
            for json_path in sorted(search_dir.glob("*.json")):
                with open(json_path) as f:
                    search_results.append(json.load(f))
            logger.info("  Loaded %d existing search results from disk", len(search_results))

    # ----- Stage 4: Benchmark (optional) -----
    benchmark_results: List[Dict[str, Any]] = []
    if args.run_benchmark and not args.skip_benchmark:
        t0 = time.time()
        benchmark_results = stage_benchmark(
            model_keys, output_dir, args.duration_s, seed=args.seed,
            use_energy=use_energy,
        )
        logger.info("Stage 4 complete: %.1fs (%d results)", time.time() - t0, len(benchmark_results))
    else:
        logger.info("Stage 4: SKIPPED (use --run-benchmark to enable)")

    # ----- Stage 5: Reports -----
    elapsed = time.time() - start_time

    if sim_results or search_results:
        stage_report(output_dir, sim_results, search_results, benchmark_results)
        print_console_summary(sim_results, search_results)

    write_summary_json(
        output_dir, model_keys, args.duration_s,
        profile_dirs, lut_dirs,
        sim_results, search_results, benchmark_results,
        elapsed,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("E2E Validation Complete")
    logger.info("=" * 60)
    logger.info("Total elapsed: %.1fs", elapsed)
    logger.info("Simulation runs: %d", len(sim_results))
    logger.info("Search runs: %d", len(search_results))
    logger.info("Benchmark runs: %d", len(benchmark_results))
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
