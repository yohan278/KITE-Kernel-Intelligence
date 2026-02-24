#!/usr/bin/env python3
"""Estimator Sweep: compare all 11 sklearn estimators on profiling data.

Loads profiling CSVs from the e2e_v2 profiles directory, runs four comparison
analyses (global, per-category, cross-model, and pick-best), and writes
JSON results plus a human-readable summary.

Usage:
    python -m evals.scripts.run_estimator_sweep \
        --profiling-dir data/e2e_v2/profiles \
        --output-dir data/estimator_sweep \
        --val-fraction 0.2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("estimator_sweep")

# ---------------------------------------------------------------------------
# Model slug -> inference_search key mapping
# ---------------------------------------------------------------------------
# Profile directories use HuggingFace slug format (Qwen_Qwen3-0.6B),
# while _EXAMPLE_MODELS uses short keys (qwen3-0.6b).
_SLUG_TO_MODEL_KEY = {
    "Qwen_Qwen3-0.6B": "qwen3-0.6b",
    "Qwen_Qwen3-1.7B": "qwen3-1.7b",
    "Qwen_Qwen3-4B": "qwen3-4b",
    "Qwen_Qwen3-8B": "qwen3-8b",
    "Qwen_Qwen3-14B": "qwen3-14b",
    "Qwen_Qwen3-32B": "qwen3-32b",
}

# All 11 estimator string names supported by _get_estimator_name_map()
ALL_ESTIMATOR_NAMES = [
    "random_forest",
    "linear_regression",
    "ridge",
    "lasso",
    "knn",
    "bayesian_linear",
    "mlp",
    "svr",
    "gaussian_process",
    "xgboost",
    "lightgbm",
]


def _get_operator_name_to_category() -> Dict[str, Any]:
    """Return the operator_name -> OperatorCategory mapping from lut_generator."""
    from inference_simulator.types.operators import OperatorCategory

    return {
        "linear_qkv": OperatorCategory.LINEAR,
        "linear_o": OperatorCategory.LINEAR,
        "mlp_up": OperatorCategory.LINEAR,
        "mlp_gate": OperatorCategory.LINEAR,
        "mlp_down": OperatorCategory.LINEAR,
        "lm_head": OperatorCategory.LINEAR,
        "embedding": OperatorCategory.EMBEDDING,
        "rotary_embedding": OperatorCategory.EMBEDDING,
        "layernorm": OperatorCategory.NORMALIZATION,
        "rmsnorm": OperatorCategory.NORMALIZATION,
        "gelu_activation": OperatorCategory.ACTIVATION,
        "silu_activation": OperatorCategory.ACTIVATION,
        "softmax": OperatorCategory.ACTIVATION,
        "dropout": OperatorCategory.ACTIVATION,
        "cross_entropy_loss": OperatorCategory.ACTIVATION,
        "residual_add": OperatorCategory.ACTIVATION,
        "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
        "attention_decode": OperatorCategory.ATTENTION_DECODE,
        "sliding_window_attention": OperatorCategory.ATTENTION_DECODE,
        "kv_cache_append": OperatorCategory.KV_CACHE,
        "kv_cache_evict": OperatorCategory.KV_CACHE,
        "mqa_gqa_expansion": OperatorCategory.KV_CACHE,
        "fused_prefill": OperatorCategory.FUSED_PREFILL,
        "fused_decode_step": OperatorCategory.FUSED_DECODE_STEP,
        "fused_attention": OperatorCategory.FUSED_ATTENTION,
        "fused_mlp": OperatorCategory.FUSED_MLP,
        "fused_norm_attn": OperatorCategory.FUSED_NORM_ATTN,
    }


def _model_spec_to_dims(model_spec: Any) -> Dict[str, float]:
    """Extract numeric dimension features from a ModelSpec for use as model_dims."""
    return {
        "hidden_dim": float(model_spec.hidden_dim),
        "intermediate_dim": float(model_spec.intermediate_dim),
        "num_attention_heads": float(model_spec.num_attention_heads),
        "num_kv_heads": float(model_spec.num_kv_heads),
        "num_layers": float(model_spec.num_layers),
        "head_dim": float(model_spec.head_dim),
        "vocab_size": float(model_spec.vocab_size),
    }


def _discover_models(profiling_dir: Path) -> Dict[str, Path]:
    """Auto-discover model profile directories.

    Returns:
        {model_slug: path_to_csv_dir} for each model that has CSV files.
    """
    models: Dict[str, Path] = {}
    if not profiling_dir.is_dir():
        return models

    for model_dir in sorted(profiling_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # Look for CSV files under any hw_slug/precision subdirectory
        for csv_dir in model_dir.glob("*/fp16"):
            csv_files = list(csv_dir.glob("*.csv"))
            if csv_files:
                models[model_dir.name] = csv_dir
                break
        # Also check for CSVs directly in hw_slug dirs (no precision subdir)
        if model_dir.name not in models:
            for hw_dir in model_dir.iterdir():
                if hw_dir.is_dir():
                    csv_files = list(hw_dir.glob("*.csv"))
                    if csv_files:
                        models[model_dir.name] = hw_dir
                        break

    return models


def _load_measurements_for_model(
    csv_dir: Path,
) -> List[Any]:
    """Load all measurements from a model's CSV directory."""
    from inference_simulator.estimator.sklearn_base import (
        load_csv_measurements_auto_category,
    )

    op_map = _get_operator_name_to_category()
    all_measurements: List[Any] = []

    for csv_file in sorted(csv_dir.glob("*.csv")):
        measurements = load_csv_measurements_auto_category(csv_file, op_map)
        all_measurements.extend(measurements)

    return all_measurements


def _get_available_estimators() -> List[str]:
    """Return the subset of ALL_ESTIMATOR_NAMES that can be imported."""
    from inference_simulator.estimator.multi_output import _get_estimator_name_map

    available = _get_estimator_name_map()
    return [name for name in ALL_ESTIMATOR_NAMES if name in available]


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/non-serializable types for JSON output."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


def run_sweep(
    profiling_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Run the full estimator sweep and write results.

    Args:
        profiling_dir: Root profiling directory (contains model slug subdirs).
        output_dir: Where to write JSON results and summary.md.
        val_fraction: Validation split fraction.

    Returns:
        Dict with all comparison results.
    """
    from inference_simulator.estimator.model_comparison import (
        compare_estimators,
        compare_estimators_by_category,
        cross_model_evaluation,
        pick_best_estimator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Discover models ---
    models = _discover_models(profiling_dir)
    if not models:
        raise FileNotFoundError(
            f"No model profile directories found in {profiling_dir}"
        )
    logger.info("Discovered %d models: %s", len(models), list(models.keys()))

    # --- Load model specs for model_dims ---
    model_specs: Dict[str, Any] = {}
    model_dims_by_model: Dict[str, Dict[str, float]] = {}
    try:
        from inference_search.cli import _EXAMPLE_MODELS

        for slug, _csv_dir in models.items():
            key = _SLUG_TO_MODEL_KEY.get(slug)
            if key and key in _EXAMPLE_MODELS:
                spec = _EXAMPLE_MODELS[key]
                model_specs[slug] = spec
                model_dims_by_model[slug] = _model_spec_to_dims(spec)
                logger.info("  %s -> model_dims loaded (%s)", slug, key)
            else:
                logger.warning("  %s -> no ModelSpec found (key=%s)", slug, key)
    except ImportError:
        logger.warning("Could not import inference_search.cli._EXAMPLE_MODELS")

    # --- Load measurements per model ---
    measurements_by_model: Dict[str, List[Any]] = {}
    all_measurements: List[Any] = []

    for slug, csv_dir in models.items():
        ms = _load_measurements_for_model(csv_dir)
        if ms:
            measurements_by_model[slug] = ms
            all_measurements.extend(ms)
            logger.info("  %s: %d measurements loaded", slug, len(ms))
        else:
            logger.warning("  %s: no measurements loaded", slug)

    if not all_measurements:
        raise ValueError("No measurements loaded from any model")

    logger.info(
        "Total: %d measurements across %d models",
        len(all_measurements),
        len(measurements_by_model),
    )

    # --- Determine available estimators ---
    estimator_names = _get_available_estimators()
    logger.info("Available estimators (%d): %s", len(estimator_names), estimator_names)

    # Use model_dims from the first available model for global comparison
    # (or None if no model specs available)
    first_model_dims: Optional[Dict[str, float]] = None
    if model_dims_by_model:
        first_model_dims = next(iter(model_dims_by_model.values()))

    results: Dict[str, Any] = {
        "meta": {
            "profiling_dir": str(profiling_dir),
            "models": list(models.keys()),
            "num_measurements": len(all_measurements),
            "estimators": estimator_names,
            "val_fraction": val_fraction,
        },
    }

    # --- 1. Global comparison ---
    logger.info("Running global comparison (%d estimators)...", len(estimator_names))
    global_results = compare_estimators(
        all_measurements,
        model_dims=first_model_dims,
        estimator_classes=estimator_names,
        val_fraction=val_fraction,
    )
    results["global_comparison"] = global_results

    try:
        best_global = pick_best_estimator(global_results, "time_r2")
        results["best_global_estimator"] = best_global
        logger.info("Best global estimator (time_r2): %s", best_global)
    except ValueError as e:
        logger.warning("Could not pick best global estimator: %s", e)
        results["best_global_estimator"] = None

    # --- 2. Per-category comparison ---
    logger.info("Running per-category comparison...")
    per_category_results = compare_estimators_by_category(
        all_measurements,
        model_dims=first_model_dims,
        estimator_classes=estimator_names,
        val_fraction=val_fraction,
        include_per_operator=True,
    )
    results["per_category_comparison"] = per_category_results

    # --- 3. Cross-model evaluation ---
    if len(measurements_by_model) >= 2:
        logger.info(
            "Running cross-model evaluation (leave-one-out, %d models)...",
            len(measurements_by_model),
        )
        cross_model_results = cross_model_evaluation(
            measurements_by_model,
            estimator_classes=estimator_names,
            model_dims_by_model=model_dims_by_model or None,
        )
        results["cross_model_evaluation"] = cross_model_results
    else:
        logger.warning(
            "Skipping cross-model evaluation (need >= 2 models, have %d)",
            len(measurements_by_model),
        )
        results["cross_model_evaluation"] = {}

    # --- 4. Pick best per category ---
    best_per_category: Dict[str, str] = {}
    for cat_name, cat_results in per_category_results.items():
        try:
            best = pick_best_estimator(cat_results, "time_r2")
            best_per_category[cat_name] = best
        except ValueError:
            pass
    results["best_per_category"] = best_per_category

    elapsed = time.time() - t0
    results["meta"]["elapsed_seconds"] = round(elapsed, 2)
    logger.info("Sweep completed in %.1fs", elapsed)

    # --- Write outputs ---
    _write_json(output_dir / "global_comparison.json", results["global_comparison"])
    _write_json(output_dir / "per_category_comparison.json", results["per_category_comparison"])
    _write_json(output_dir / "cross_model_evaluation.json", results["cross_model_evaluation"])
    _write_json(output_dir / "full_results.json", results)
    _write_summary(output_dir / "summary.md", results)

    logger.info("Results written to %s", output_dir)
    return results


def _write_json(path: Path, data: Any) -> None:
    """Write data as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(_make_serializable(data), f, indent=2)
    logger.info("  -> %s", path)


def _write_summary(path: Path, results: Dict[str, Any]) -> None:
    """Write a human-readable Markdown summary."""
    lines: List[str] = []
    meta = results["meta"]

    lines.append("# Estimator Sweep Summary")
    lines.append("")
    lines.append(f"- **Models**: {', '.join(meta['models'])}")
    lines.append(f"- **Total measurements**: {meta['num_measurements']}")
    lines.append(f"- **Estimators**: {len(meta['estimators'])}")
    lines.append(f"- **Validation fraction**: {meta['val_fraction']}")
    lines.append(f"- **Elapsed**: {meta.get('elapsed_seconds', '?')}s")
    lines.append("")

    # --- Global comparison table ---
    lines.append("## Global Comparison (all models pooled)")
    lines.append("")
    global_results = results.get("global_comparison", [])
    if global_results:
        lines.append("| Estimator | time_r2 | time_mae | time_rmse | error |")
        lines.append("|-----------|---------|----------|-----------|-------|")
        # Sort by time_r2 descending
        sorted_results = sorted(
            global_results,
            key=lambda x: x.get("time_r2", float("-inf")),
            reverse=True,
        )
        for entry in sorted_results:
            name = entry.get("estimator", "?")
            r2 = entry.get("time_r2")
            mae = entry.get("time_mae")
            rmse = entry.get("time_rmse")
            error = entry.get("error", "")
            r2_str = f"{r2:.4f}" if r2 is not None else "-"
            mae_str = f"{mae:.6f}" if mae is not None else "-"
            rmse_str = f"{rmse:.6f}" if rmse is not None else "-"
            lines.append(f"| {name} | {r2_str} | {mae_str} | {rmse_str} | {error} |")
        lines.append("")

    best = results.get("best_global_estimator")
    if best:
        lines.append(f"**Best global estimator (time_r2)**: {best}")
        lines.append("")

    # --- Per-category comparison ---
    lines.append("## Per-Category Comparison")
    lines.append("")
    per_cat = results.get("per_category_comparison", {})
    best_per_cat = results.get("best_per_category", {})
    for cat_name in sorted(per_cat.keys()):
        cat_results = per_cat[cat_name]
        lines.append(f"### {cat_name}")
        lines.append("")
        lines.append("| Estimator | time_r2 | time_mae | time_rmse | error |")
        lines.append("|-----------|---------|----------|-----------|-------|")
        sorted_cat = sorted(
            cat_results.items(),
            key=lambda x: x[1].get("time_r2", float("-inf")),
            reverse=True,
        )
        for est_name, metrics in sorted_cat:
            r2 = metrics.get("time_r2")
            mae = metrics.get("time_mae")
            rmse = metrics.get("time_rmse")
            error = metrics.get("error", "")
            r2_str = f"{r2:.4f}" if r2 is not None else "-"
            mae_str = f"{mae:.6f}" if mae is not None else "-"
            rmse_str = f"{rmse:.6f}" if rmse is not None else "-"
            lines.append(f"| {est_name} | {r2_str} | {mae_str} | {rmse_str} | {error} |")
        lines.append("")
        if cat_name in best_per_cat:
            lines.append(f"**Best**: {best_per_cat[cat_name]}")
            lines.append("")

    # --- Cross-model evaluation ---
    lines.append("## Cross-Model Evaluation (leave-one-out)")
    lines.append("")
    cross_model = results.get("cross_model_evaluation", {})
    if cross_model:
        # Collect all holdout models
        holdout_models = set()
        for est_name, model_results in cross_model.items():
            holdout_models.update(model_results.keys())
        holdout_models_sorted = sorted(holdout_models)

        # Build header
        header = "| Estimator |"
        sep = "|-----------|"
        for hm in holdout_models_sorted:
            header += f" {hm} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        # Sort estimators by average holdout R2
        est_avg: List[tuple] = []
        for est_name, model_results in cross_model.items():
            scores = [
                v.get("time_holdout_r2", float("-inf"))
                for v in model_results.values()
                if "error" not in v
            ]
            avg = sum(scores) / len(scores) if scores else float("-inf")
            est_avg.append((est_name, avg))
        est_avg.sort(key=lambda x: x[1], reverse=True)

        for est_name, _avg in est_avg:
            model_results = cross_model[est_name]
            row = f"| {est_name} |"
            for hm in holdout_models_sorted:
                hr = model_results.get(hm, {})
                if "error" in hr:
                    row += f" err |"
                else:
                    r2 = hr.get("time_holdout_r2")
                    row += f" {r2:.4f} |" if r2 is not None else " - |"
            lines.append(row)
        lines.append("")
    else:
        lines.append("_Skipped (fewer than 2 models)._")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("  -> %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all sklearn estimators on profiling data",
    )
    parser.add_argument(
        "--profiling-dir",
        type=Path,
        default=Path("data/e2e_v2/profiles"),
        help="Root directory containing model profile subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/estimator_sweep"),
        help="Directory for output JSON and summary files",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    args = parser.parse_args()

    run_sweep(
        profiling_dir=args.profiling_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
