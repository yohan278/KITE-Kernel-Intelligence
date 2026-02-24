#!/usr/bin/env python3
"""Golden Set Ablation: forward-selection of operator categories for minimal E2E error.

Determines the smallest subset of OperatorCategory groups that achieves
a target end-to-end simulation error vs ground truth.  Two main phases:

  Phase 1 -- Forward selection:
      Start from an empty set of categories.  At each step, greedily add the
      candidate category that reduces E2E percent error the most.  Stop when
      the error drops below the target threshold (default 10%).

  Phase 2 -- Cross-model generalization:
      Leave-one-out evaluation across model families to verify that the
      golden operator set transfers to unseen models.

Usage:
    python -m evals.scripts.run_golden_set_ablation \
        --profiling-dir data/e2e_v2/profiles \
        --gt-dir data/ground_truth \
        --output-dir data/ablation \
        --estimator-type random_forest \
        --error-target 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("golden_set_ablation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARDWARE_KEY = "a100_80gb"

# All ablation-candidate categories (order determines tie-breaking)
ALL_CATEGORIES = [
    "LINEAR",
    "NORMALIZATION",
    "ACTIVATION",
    "ATTENTION_PREFILL",
    "ATTENTION_DECODE",
    "KV_CACHE",
    "SAMPLING",
    "COMMUNICATION",
    "EMBEDDING",
]

# operator_name -> OperatorCategory mapping (mirrors lut_generator.py:252-280)
OPERATOR_NAME_TO_CATEGORY_STR: Dict[str, str] = {
    "linear_qkv": "LINEAR",
    "linear_o": "LINEAR",
    "mlp_up": "LINEAR",
    "mlp_gate": "LINEAR",
    "mlp_down": "LINEAR",
    "lm_head": "LINEAR",
    "embedding": "EMBEDDING",
    "rotary_embedding": "EMBEDDING",
    "layernorm": "NORMALIZATION",
    "rmsnorm": "NORMALIZATION",
    "gelu_activation": "ACTIVATION",
    "silu_activation": "ACTIVATION",
    "softmax": "ACTIVATION",
    "dropout": "ACTIVATION",
    "cross_entropy_loss": "ACTIVATION",
    "residual_add": "ACTIVATION",
    "attention_prefill": "ATTENTION_PREFILL",
    "attention_decode": "ATTENTION_DECODE",
    "sliding_window_attention": "ATTENTION_DECODE",
    "kv_cache_append": "KV_CACHE",
    "kv_cache_evict": "KV_CACHE",
    "mqa_gqa_expansion": "KV_CACHE",
}

# Simulation defaults
DEFAULT_CONFIG_ID = "1gpu-fp16"
DEFAULT_WORKLOAD = "chat"
DEFAULT_QPS = 10.0
DEFAULT_DURATION_S = 10.0

# Minimum measurements to attempt training
MIN_MEASUREMENTS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_enum(name: str):
    """Convert a category string name to an OperatorCategory enum."""
    from inference_simulator.types.operators import OperatorCategory
    return OperatorCategory[name]


def _build_operator_name_to_category():
    """Build the operator_name -> OperatorCategory dict."""
    from inference_simulator.types.operators import OperatorCategory
    return {
        op_name: OperatorCategory[cat_str]
        for op_name, cat_str in OPERATOR_NAME_TO_CATEGORY_STR.items()
    }


def _load_all_measurements(profiling_dir: Path) -> list:
    """Load all OperatorMeasurements from a profiling directory."""
    from inference_simulator.estimator.sklearn_base import (
        load_csv_measurements,
        load_csv_measurements_auto_category,
    )
    from inference_simulator.types.operators import OperatorCategory

    profiling_dir = Path(profiling_dir)
    op_name_to_cat = _build_operator_name_to_category()

    # Category-keyed individual CSVs
    csv_category_map = {
        "linear": OperatorCategory.LINEAR,
        "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
        "attention_decode": OperatorCategory.ATTENTION_DECODE,
        "embedding": OperatorCategory.EMBEDDING,
        "normalization": OperatorCategory.NORMALIZATION,
        "activation": OperatorCategory.ACTIVATION,
        "communication": OperatorCategory.COMMUNICATION,
        "kv_cache": OperatorCategory.KV_CACHE,
        "sampling": OperatorCategory.SAMPLING,
    }

    # Combined CSVs that contain mixed operator categories
    combined_csvs = {
        "token_ops", "attention", "agentic", "sampling",
        "communication", "moe", "ssm", "mtp", "cpu_host",
    }

    all_measurements = []

    for name, cat in csv_category_map.items():
        csv_file = profiling_dir / f"{name}.csv"
        if csv_file.exists():
            all_measurements.extend(load_csv_measurements(csv_file, cat))

    for csv_file in profiling_dir.glob("*.csv"):
        stem = csv_file.stem.lower()
        if stem in csv_category_map:
            continue  # already loaded above
        if stem in combined_csvs:
            all_measurements.extend(
                load_csv_measurements_auto_category(csv_file, op_name_to_cat)
            )

    return all_measurements


def _load_measurements_by_model(
    profiling_dir: Path,
) -> Dict[str, list]:
    """Load measurements grouped by model subdirectory.

    Expects profiling_dir to contain per-model subdirectories like:
        profiling_dir/Qwen_Qwen3-0.6B/nvidia_a100_80gb_sxm/fp16/*.csv
    """
    measurements_by_model: Dict[str, list] = {}

    # Try flat layout first (all CSVs directly in profiling_dir)
    flat_csvs = list(profiling_dir.glob("*.csv"))
    if flat_csvs:
        all_m = _load_all_measurements(profiling_dir)
        if all_m:
            measurements_by_model[profiling_dir.name] = all_m

    # Then try per-model subdirectory layout
    for model_dir in sorted(profiling_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # Look for CSVs in the model_dir or nested under hw/precision
        csv_dirs = []
        direct_csvs = list(model_dir.glob("*.csv"))
        if direct_csvs:
            csv_dirs.append(model_dir)
        for sub in model_dir.rglob("*.csv"):
            csv_dirs.append(sub.parent)
        csv_dirs = sorted(set(csv_dirs))

        for csv_dir in csv_dirs:
            m = _load_all_measurements(csv_dir)
            if m:
                key = model_dir.name
                measurements_by_model.setdefault(key, []).extend(m)

    return measurements_by_model


def _filter_measurements_by_categories(
    measurements: list,
    categories: Set[str],
) -> list:
    """Filter measurements to only those whose category is in the given set."""
    allowed = {_category_enum(c) for c in categories}
    return [m for m in measurements if m.category in allowed]


def _train_estimator(
    measurements: list,
    estimator_type: str,
):
    """Train an estimator on the given measurements.

    Returns a fitted SklearnEstimatorBase instance.
    """
    from inference_simulator.estimator.multi_output import _get_estimator_name_map

    name_map = _get_estimator_name_map()
    key = estimator_type.lower().replace("-", "_")
    if key not in name_map:
        raise ValueError(
            f"Unknown estimator type '{estimator_type}'. "
            f"Available: {sorted(name_map.keys())}"
        )
    est = name_map[key]()
    est.fit(measurements, val_fraction=0.2)
    return est


def _generate_luts(
    estimator,
    output_dir: Path,
    model_spec=None,
    hw_spec=None,
):
    """Generate LUT bundle from a trained estimator.

    Returns a LUTBundle.
    """
    from inference_simulator.estimator.lut_generator import LUTGenerator
    from inference_simulator.types.lut_bundle import LUTBundle

    output_dir.mkdir(parents=True, exist_ok=True)
    gen = LUTGenerator()

    token_ops_path = gen.generate_gpu_token_ops_lut(
        estimator, output_path=output_dir / "gpu_token_ops.npz"
    )
    prefill_path = gen.generate_attention_prefill_lut(
        estimator, output_path=output_dir / "attention_prefill.npz"
    )
    decode_path = gen.generate_attention_decode_lut(
        estimator, output_path=output_dir / "attention_decode.npz"
    )

    model_id = getattr(model_spec, "model_id", "ablation") if model_spec else "ablation"
    hardware_id = getattr(hw_spec, "name", HARDWARE_KEY) if hw_spec else HARDWARE_KEY

    return LUTBundle(
        base_dir=output_dir,
        model_id=model_id,
        hardware_id=hardware_id,
        quantization="fp16",
        gpu_token_ops_lut=token_ops_path,
        gpu_attention_prefill_lut=prefill_path,
        gpu_attention_decode_lut=decode_path,
    )


def _run_simulation(
    lut_bundle,
    model_spec,
    hw_spec,
    workload_type: str = DEFAULT_WORKLOAD,
    qps: float = DEFAULT_QPS,
    duration_s: float = DEFAULT_DURATION_S,
    config_id: str = DEFAULT_CONFIG_ID,
    seed: int = 42,
):
    """Run a single simulation and return SimulationMetrics."""
    from inference_simulator.engine.simulator import EventDrivenSimulator
    from inference_simulator.scheduler.vllm import VLLMScheduler
    from inference_simulator.types import InferenceSpec, WorkloadSpec

    inference_spec = InferenceSpec(
        num_gpus=1,
        tensor_parallel=1,
        precision="fp16",
    )

    workload_factories = {
        "chat": WorkloadSpec.for_chat,
        "reasoning": WorkloadSpec.for_reasoning,
        "rag": WorkloadSpec.for_rag,
        "agentic": WorkloadSpec.for_agentic,
    }
    workload = workload_factories.get(workload_type, WorkloadSpec.for_chat)(qps=qps)

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

    return sim.run(workload_spec=workload, duration_s=duration_s, seed=seed)


def _load_ground_truth(gt_dir: Path, config_id: str, workload_type: str, qps: float):
    """Load ground truth metrics from a JSON/JSONL file.

    Searches gt_dir for files matching the config pattern.
    Returns a dict of metric values or None if not found.
    """
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return None

    # Try several filename patterns
    candidates = [
        gt_dir / f"{config_id}_{workload_type}_qps{int(qps)}.json",
        gt_dir / f"{config_id}.jsonl",
        gt_dir / workload_type / f"{config_id}.jsonl",
    ]

    for path in candidates:
        if not path.exists():
            continue

        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("qps") == qps or entry.get("qps_target") == qps:
                        return entry.get("metrics", entry)
            # Return the last entry if no QPS match
            with open(path) as f:
                last = None
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
                return last.get("metrics", last) if last else None
        else:
            with open(path) as f:
                data = json.load(f)
                return data.get("metrics", data)

    return None


def _compute_e2e_percent_error(sim_metrics, gt_metrics) -> float:
    """Compute percent error between simulated and ground truth metrics.

    Uses the average relative error across key metrics:
    throughput_tps, ttft_p50, e2e_p50, tbt_p50.
    """
    if gt_metrics is None:
        return float("inf")

    metric_keys = ["throughput_tps", "ttft_p50", "e2e_p50", "tbt_p50"]
    errors = []

    sim_dict = asdict(sim_metrics) if hasattr(sim_metrics, "__dataclass_fields__") else sim_metrics

    for key in metric_keys:
        sim_val = sim_dict.get(key, 0.0)
        gt_val = gt_metrics.get(key, 0.0) if isinstance(gt_metrics, dict) else getattr(gt_metrics, key, 0.0)

        if gt_val is None or gt_val == 0:
            continue

        error = abs(sim_val - gt_val) / abs(gt_val)
        errors.append(error)

    if not errors:
        return float("inf")

    return float(np.mean(errors))


def _evaluate_category_subset(
    measurements: list,
    categories: Set[str],
    estimator_type: str,
    model_spec,
    hw_spec,
    gt_metrics,
    lut_scratch_dir: Path,
    workload_type: str = DEFAULT_WORKLOAD,
    qps: float = DEFAULT_QPS,
    duration_s: float = DEFAULT_DURATION_S,
    seed: int = 42,
) -> float:
    """Evaluate E2E error for a given operator category subset.

    Returns the E2E percent error (0.0 = perfect, inf = failed).
    """
    filtered = _filter_measurements_by_categories(measurements, categories)

    if len(filtered) < MIN_MEASUREMENTS:
        logger.debug(
            "  Skipping subset %s: only %d measurements (need %d)",
            sorted(categories), len(filtered), MIN_MEASUREMENTS,
        )
        return float("inf")

    try:
        est = _train_estimator(filtered, estimator_type)
    except Exception as e:
        logger.warning("  Estimator training failed for %s: %s", sorted(categories), e)
        return float("inf")

    try:
        bundle = _generate_luts(est, lut_scratch_dir, model_spec, hw_spec)
    except Exception as e:
        logger.warning("  LUT generation failed for %s: %s", sorted(categories), e)
        return float("inf")

    try:
        metrics = _run_simulation(
            bundle, model_spec, hw_spec,
            workload_type=workload_type, qps=qps,
            duration_s=duration_s, seed=seed,
        )
    except Exception as e:
        logger.warning("  Simulation failed for %s: %s", sorted(categories), e)
        return float("inf")

    error = _compute_e2e_percent_error(metrics, gt_metrics)
    return error


def _get_default_model_spec():
    """Get a default model spec for ablation (smallest profiled model)."""
    try:
        from inference_search.cli import _EXAMPLE_MODELS
        for key in ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]:
            if key in _EXAMPLE_MODELS:
                return _EXAMPLE_MODELS[key]
    except ImportError:
        pass

    # Fallback: build a minimal ModelSpec
    from inference_simulator.types.model_spec import ModelSpec
    return ModelSpec(
        model_id="ablation-model",
        total_params_billion=0.6,
        hidden_dim=1024,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=8,
        vocab_size=151936,
        intermediate_dim=2816,
    )


def _get_default_hw_spec():
    """Get the default hardware spec."""
    from inference_simulator.types import HardwareSpec
    return HardwareSpec.from_registry(HARDWARE_KEY)


# ---------------------------------------------------------------------------
# Phase 1: Forward Selection
# ---------------------------------------------------------------------------

def forward_selection(
    measurements: list,
    estimator_type: str,
    model_spec,
    hw_spec,
    gt_metrics,
    output_dir: Path,
    error_target: float = 0.10,
    workload_type: str = DEFAULT_WORKLOAD,
    qps: float = DEFAULT_QPS,
    duration_s: float = DEFAULT_DURATION_S,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Greedy forward selection of operator categories.

    Returns a list of step results, each dict containing:
        step, candidate_added, categories_selected, e2e_error, num_measurements
    """
    # Determine which categories have enough measurements to be candidates
    from inference_simulator.types.operators import OperatorCategory
    category_counts: Dict[str, int] = {}
    for m in measurements:
        cat_name = m.category.name
        if cat_name in ALL_CATEGORIES:
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    available_candidates = [
        c for c in ALL_CATEGORIES if category_counts.get(c, 0) >= 1
    ]
    logger.info(
        "Available categories: %s",
        {c: category_counts.get(c, 0) for c in available_candidates},
    )

    selected: Set[str] = set()
    remaining = set(available_candidates)
    steps: List[Dict[str, Any]] = []
    lut_scratch = output_dir / "_lut_scratch"

    for step_idx in range(len(available_candidates)):
        logger.info(
            "=== Forward selection step %d (selected=%d, remaining=%d) ===",
            step_idx + 1, len(selected), len(remaining),
        )

        best_candidate = None
        best_error = float("inf")
        candidate_errors: Dict[str, float] = {}

        for candidate in sorted(remaining):
            trial_set = selected | {candidate}
            scratch = lut_scratch / f"step{step_idx}_{candidate}"
            scratch.mkdir(parents=True, exist_ok=True)

            error = _evaluate_category_subset(
                measurements, trial_set, estimator_type,
                model_spec, hw_spec, gt_metrics, scratch,
                workload_type=workload_type, qps=qps,
                duration_s=duration_s, seed=seed,
            )
            candidate_errors[candidate] = error

            logger.info(
                "  + %s -> error=%.4f (categories=%s)",
                candidate, error, sorted(trial_set),
            )

            if error < best_error:
                best_error = error
                best_candidate = candidate

        if best_candidate is None:
            logger.warning("No valid candidate found at step %d", step_idx + 1)
            break

        selected.add(best_candidate)
        remaining.discard(best_candidate)

        filtered_count = len(
            _filter_measurements_by_categories(measurements, selected)
        )

        step_result = {
            "step": step_idx + 1,
            "candidate_added": best_candidate,
            "categories_selected": sorted(selected),
            "e2e_error": best_error,
            "num_measurements": filtered_count,
            "candidate_errors": candidate_errors,
        }
        steps.append(step_result)

        logger.info(
            "  BEST: +%s -> error=%.4f (total categories=%d)",
            best_candidate, best_error, len(selected),
        )

        if best_error <= error_target:
            logger.info(
                "  Target error %.2f%% reached at step %d with %d categories",
                error_target * 100, step_idx + 1, len(selected),
            )
            break

    return steps


# ---------------------------------------------------------------------------
# Phase 2: Cross-Model Generalization
# ---------------------------------------------------------------------------

def cross_model_generalization(
    measurements_by_model: Dict[str, list],
    golden_categories: Set[str],
    estimator_type: str,
) -> Dict[str, Any]:
    """Leave-one-out cross-model evaluation using the golden operator set.

    For each model, trains on all other models' measurements (filtered to
    golden categories) and evaluates on the held-out model.

    Returns results dict with per-holdout R2 scores.
    """
    from inference_simulator.estimator.model_comparison import cross_model_evaluation

    if len(measurements_by_model) < 2:
        logger.warning("Need >= 2 models for cross-model evaluation, got %d", len(measurements_by_model))
        return {"error": "insufficient_models", "models": list(measurements_by_model.keys())}

    # Filter all model measurements to golden categories
    filtered_by_model: Dict[str, list] = {}
    for model_name, model_measurements in measurements_by_model.items():
        filtered = _filter_measurements_by_categories(model_measurements, golden_categories)
        if len(filtered) >= MIN_MEASUREMENTS:
            filtered_by_model[model_name] = filtered
        else:
            logger.warning(
                "Model %s has only %d measurements after filtering, skipping",
                model_name, len(filtered),
            )

    if len(filtered_by_model) < 2:
        return {"error": "insufficient_filtered_models", "models": list(filtered_by_model.keys())}

    # Use the specified estimator type
    estimator_classes = [estimator_type]

    try:
        results = cross_model_evaluation(
            measurements_by_model=filtered_by_model,
            estimator_classes=estimator_classes,
        )
    except Exception as e:
        logger.error("Cross-model evaluation failed: %s", e)
        return {"error": str(e)}

    # Also check cross-family transfer if possible
    cross_family = _cross_family_evaluation(filtered_by_model, estimator_type)

    return {
        "leave_one_out": results,
        "cross_family": cross_family,
        "models_used": list(filtered_by_model.keys()),
        "golden_categories": sorted(golden_categories),
        "measurements_per_model": {
            k: len(v) for k, v in filtered_by_model.items()
        },
    }


def _cross_family_evaluation(
    measurements_by_model: Dict[str, list],
    estimator_type: str,
) -> Dict[str, Any]:
    """Split models by family and test cross-family transfer.

    Groups models by name prefix (e.g., "Qwen" vs "GLM") and trains on
    one family to predict the other.
    """
    from inference_simulator.estimator.multi_output import _get_estimator_name_map

    # Group models by family (heuristic: split on first underscore or hyphen)
    families: Dict[str, List[str]] = {}
    for model_name in measurements_by_model:
        # Extract family prefix: e.g., "Qwen_Qwen3-0.6B" -> "Qwen"
        family = model_name.split("_")[0].split("-")[0].lower()
        families.setdefault(family, []).append(model_name)

    if len(families) < 2:
        return {"note": "Only one model family found, cannot do cross-family evaluation"}

    results: Dict[str, Any] = {"families": {k: v for k, v in families.items()}}

    name_map = _get_estimator_name_map()
    key = estimator_type.lower().replace("-", "_")
    est_cls = name_map.get(key)
    if est_cls is None:
        return {"error": f"Unknown estimator type: {estimator_type}"}

    # For each pair of families, train on one, evaluate on the other
    family_names = sorted(families.keys())
    for i, train_family in enumerate(family_names):
        for j, test_family in enumerate(family_names):
            if i == j:
                continue

            train_models = families[train_family]
            test_models = families[test_family]

            train_ms = []
            for m in train_models:
                train_ms.extend(measurements_by_model[m])

            if len(train_ms) < MIN_MEASUREMENTS:
                continue

            test_results_per_model: Dict[str, Dict[str, float]] = {}
            for test_model in test_models:
                test_ms = measurements_by_model[test_model]
                if len(test_ms) < 2:
                    continue

                try:
                    est = est_cls()
                    est.fit(train_ms, val_fraction=0.0)

                    y_true = np.array([m.time_s for m in test_ms])
                    y_pred = np.array([
                        est.estimate(m.category, m.batch_size, m.seq_len).time_s
                        for m in test_ms
                    ])

                    from inference_simulator.estimator.model_comparison import _compute_metrics
                    metrics = _compute_metrics(y_true, y_pred)
                    test_results_per_model[test_model] = metrics
                except Exception as e:
                    test_results_per_model[test_model] = {"error": str(e)}

            pair_key = f"{train_family}_to_{test_family}"
            results[pair_key] = {
                "train_models": train_models,
                "test_models": test_models,
                "results": test_results_per_model,
            }

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_summary_report(
    steps: List[Dict[str, Any]],
    golden_categories: Set[str],
    cross_model_results: Dict[str, Any],
    output_path: Path,
    error_target: float,
) -> None:
    """Write a human-readable summary.md report."""
    with open(output_path, "w") as f:
        f.write("# Golden Set Ablation Report\n\n")

        # Golden set summary
        f.write("## Golden Operator Set\n\n")
        if steps:
            final_error = steps[-1]["e2e_error"]
            f.write(f"**Target error**: {error_target:.1%}\n")
            f.write(f"**Achieved error**: {final_error:.4f} ({final_error:.1%})\n")
            f.write(f"**Categories selected**: {len(golden_categories)}/{len(ALL_CATEGORIES)}\n\n")

            f.write("Categories in golden set:\n")
            for cat in sorted(golden_categories):
                f.write(f"  - {cat}\n")
            f.write("\n")
        else:
            f.write("No forward selection steps completed.\n\n")

        # Forward selection trace
        f.write("## Forward Selection Trace\n\n")
        f.write("| Step | Category Added | E2E Error | Num Measurements |\n")
        f.write("|------|---------------|-----------|------------------|\n")
        for step in steps:
            f.write(
                f"| {step['step']} | {step['candidate_added']} | "
                f"{step['e2e_error']:.4f} | {step['num_measurements']} |\n"
            )
        f.write("\n")

        # Per-step candidate details
        f.write("## Per-Step Candidate Errors\n\n")
        for step in steps:
            f.write(f"### Step {step['step']}\n\n")
            f.write(f"Selected categories so far: {step['categories_selected']}\n\n")
            f.write("| Candidate | E2E Error |\n")
            f.write("|-----------|----------|\n")
            for cand, err in sorted(
                step.get("candidate_errors", {}).items(),
                key=lambda x: x[1],
            ):
                marker = " **" if cand == step["candidate_added"] else ""
                f.write(f"| {cand}{marker} | {err:.4f} |\n")
            f.write("\n")

        # Cross-model generalization
        f.write("## Cross-Model Generalization\n\n")
        if "error" in cross_model_results:
            f.write(f"Error: {cross_model_results['error']}\n\n")
        else:
            loo = cross_model_results.get("leave_one_out", {})
            for est_name, holdout_results in loo.items():
                f.write(f"### {est_name} (Leave-One-Out)\n\n")
                f.write("| Holdout Model | Time R2 | Time MAE | Time RMSE |\n")
                f.write("|--------------|---------|----------|----------|\n")
                for holdout, scores in holdout_results.items():
                    if isinstance(scores, dict) and "error" not in scores:
                        r2 = scores.get("time_holdout_r2", scores.get("r2", "N/A"))
                        mae = scores.get("time_holdout_mae", scores.get("mae", "N/A"))
                        rmse = scores.get("time_holdout_rmse", scores.get("rmse", "N/A"))
                        r2_str = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
                        mae_str = f"{mae:.6f}" if isinstance(mae, float) else str(mae)
                        rmse_str = f"{rmse:.6f}" if isinstance(rmse, float) else str(rmse)
                        f.write(f"| {holdout} | {r2_str} | {mae_str} | {rmse_str} |\n")
                    else:
                        f.write(f"| {holdout} | ERROR | - | - |\n")
                f.write("\n")

            cross_family = cross_model_results.get("cross_family", {})
            if cross_family and "note" not in cross_family:
                f.write("### Cross-Family Transfer\n\n")
                for pair_key, pair_data in cross_family.items():
                    if pair_key == "families":
                        continue
                    if not isinstance(pair_data, dict) or "results" not in pair_data:
                        continue
                    f.write(f"**{pair_key}** (train: {pair_data.get('train_models', [])}, "
                            f"test: {pair_data.get('test_models', [])})\n\n")
                    f.write("| Test Model | R2 | MAE | RMSE |\n")
                    f.write("|------------|-----|-----|------|\n")
                    for test_model, scores in pair_data["results"].items():
                        if "error" not in scores:
                            f.write(
                                f"| {test_model} | {scores.get('r2', 'N/A'):.4f} | "
                                f"{scores.get('mae', 'N/A'):.6f} | "
                                f"{scores.get('rmse', 'N/A'):.6f} |\n"
                            )
                        else:
                            f.write(f"| {test_model} | ERROR | - | - |\n")
                    f.write("\n")

    logger.info("Summary report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Golden set ablation: find the minimal operator categories for target E2E error"
    )
    parser.add_argument(
        "--profiling-dir", required=True,
        help="Directory containing profiling CSV files (or per-model subdirs)",
    )
    parser.add_argument(
        "--gt-dir", default=None,
        help="Directory containing ground truth benchmark results",
    )
    parser.add_argument(
        "--output-dir", default="data/ablation",
        help="Output directory for results (default: data/ablation)",
    )
    parser.add_argument(
        "--estimator-type", default="random_forest",
        help="Estimator type to use (default: random_forest)",
    )
    parser.add_argument(
        "--error-target", type=float, default=0.10,
        help="Target E2E percent error to stop forward selection (default: 0.10)",
    )
    parser.add_argument(
        "--workload-type", default=DEFAULT_WORKLOAD,
        help=f"Workload type for simulation (default: {DEFAULT_WORKLOAD})",
    )
    parser.add_argument(
        "--qps", type=float, default=DEFAULT_QPS,
        help=f"QPS for simulation (default: {DEFAULT_QPS})",
    )
    parser.add_argument(
        "--duration-s", type=float, default=DEFAULT_DURATION_S,
        help=f"Simulation duration in seconds (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-cross-model", action="store_true",
        help="Skip cross-model generalization evaluation",
    )

    args = parser.parse_args()

    profiling_dir = Path(args.profiling_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Golden Set Ablation Study")
    logger.info("=" * 60)
    logger.info("Profiling dir: %s", profiling_dir)
    logger.info("Ground truth dir: %s", gt_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Estimator: %s", args.estimator_type)
    logger.info("Error target: %.1f%%", args.error_target * 100)
    logger.info("Workload: %s, QPS: %.1f, Duration: %.1fs", args.workload_type, args.qps, args.duration_s)

    # Load measurements
    logger.info("")
    logger.info("Loading measurements...")
    measurements_by_model = _load_measurements_by_model(profiling_dir)
    all_measurements = []
    for model_ms in measurements_by_model.values():
        all_measurements.extend(model_ms)

    if not all_measurements:
        logger.error("No measurements found in %s", profiling_dir)
        return

    logger.info(
        "Loaded %d measurements from %d models",
        len(all_measurements), len(measurements_by_model),
    )

    # Category distribution
    from collections import Counter
    cat_counts = Counter(m.category.name for m in all_measurements)
    for cat_name in ALL_CATEGORIES:
        logger.info("  %s: %d measurements", cat_name, cat_counts.get(cat_name, 0))

    # Load model/hardware specs
    model_spec = _get_default_model_spec()
    hw_spec = _get_default_hw_spec()

    # Load ground truth
    gt_metrics = None
    if gt_dir is not None:
        gt_metrics = _load_ground_truth(gt_dir, DEFAULT_CONFIG_ID, args.workload_type, args.qps)
        if gt_metrics is not None:
            logger.info("Ground truth loaded from %s", gt_dir)
        else:
            logger.warning("No ground truth found in %s, using simulation self-comparison", gt_dir)

    if gt_metrics is None:
        # Use a full-category simulation as the ground truth baseline
        logger.info("Generating baseline simulation with all categories as ground truth...")
        all_cats = {c for c in ALL_CATEGORIES if cat_counts.get(c, 0) >= 1}
        baseline_scratch = output_dir / "_baseline"
        baseline_scratch.mkdir(parents=True, exist_ok=True)

        try:
            baseline_filtered = _filter_measurements_by_categories(all_measurements, all_cats)
            baseline_est = _train_estimator(baseline_filtered, args.estimator_type)
            baseline_bundle = _generate_luts(baseline_est, baseline_scratch, model_spec, hw_spec)
            baseline_metrics = _run_simulation(
                baseline_bundle, model_spec, hw_spec,
                workload_type=args.workload_type, qps=args.qps,
                duration_s=args.duration_s, seed=args.seed,
            )
            gt_metrics = asdict(baseline_metrics)
            logger.info("Baseline throughput: %.1f tok/s", gt_metrics.get("throughput_tps", 0))
        except Exception as e:
            logger.error("Failed to generate baseline: %s", e)
            logger.info("Proceeding without ground truth (errors will be relative to zero)")
            gt_metrics = {"throughput_tps": 1.0, "ttft_p50": 1.0, "e2e_p50": 1.0, "tbt_p50": 0.1}

    # Phase 1: Forward selection
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1: Forward Selection")
    logger.info("=" * 60)

    steps = forward_selection(
        measurements=all_measurements,
        estimator_type=args.estimator_type,
        model_spec=model_spec,
        hw_spec=hw_spec,
        gt_metrics=gt_metrics,
        output_dir=output_dir,
        error_target=args.error_target,
        workload_type=args.workload_type,
        qps=args.qps,
        duration_s=args.duration_s,
        seed=args.seed,
    )

    # Determine golden set
    golden_categories: Set[str] = set()
    final_error = float("inf")
    if steps:
        golden_categories = set(steps[-1]["categories_selected"])
        final_error = steps[-1]["e2e_error"]

    # Save forward_selection.json
    fs_path = output_dir / "forward_selection.json"
    with open(fs_path, "w") as f:
        json.dump(steps, f, indent=2, default=str)
    logger.info("Forward selection results: %s", fs_path)

    # Save golden_set.json
    golden_info = {
        "golden_categories": sorted(golden_categories),
        "final_error": final_error,
        "num_operators": sum(
            cat_counts.get(c, 0) for c in golden_categories
        ),
        "total_steps": len(steps),
        "error_target": args.error_target,
        "estimator_type": args.estimator_type,
    }
    gs_path = output_dir / "golden_set.json"
    with open(gs_path, "w") as f:
        json.dump(golden_info, f, indent=2)
    logger.info("Golden set: %s", gs_path)

    logger.info("")
    logger.info("Golden set: %s (error=%.4f)", sorted(golden_categories), final_error)

    # Phase 2: Cross-model generalization
    cross_model_results: Dict[str, Any] = {}
    if not args.skip_cross_model and len(measurements_by_model) >= 2 and golden_categories:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Phase 2: Cross-Model Generalization")
        logger.info("=" * 60)

        cross_model_results = cross_model_generalization(
            measurements_by_model=measurements_by_model,
            golden_categories=golden_categories,
            estimator_type=args.estimator_type,
        )

        cm_path = output_dir / "cross_model_generalization.json"
        with open(cm_path, "w") as f:
            json.dump(cross_model_results, f, indent=2, default=str)
        logger.info("Cross-model results: %s", cm_path)
    elif not golden_categories:
        logger.info("Skipping cross-model evaluation (no golden categories found)")
    elif len(measurements_by_model) < 2:
        logger.info(
            "Skipping cross-model evaluation (need >= 2 models, got %d)",
            len(measurements_by_model),
        )
    else:
        logger.info("Skipping cross-model evaluation (--skip-cross-model)")

    # Generate summary report
    summary_path = output_dir / "summary.md"
    generate_summary_report(
        steps=steps,
        golden_categories=golden_categories,
        cross_model_results=cross_model_results,
        output_path=summary_path,
        error_target=args.error_target,
    )

    elapsed = time.time() - start_time

    # Console summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Ablation Study Complete")
    logger.info("=" * 60)
    logger.info("Elapsed: %.1fs", elapsed)
    logger.info("Golden categories (%d/%d): %s", len(golden_categories), len(ALL_CATEGORIES), sorted(golden_categories))
    logger.info("Final E2E error: %.4f (%.1f%%)", final_error, final_error * 100)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
