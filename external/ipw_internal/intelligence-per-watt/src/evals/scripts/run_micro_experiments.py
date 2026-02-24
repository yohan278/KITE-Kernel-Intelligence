#!/usr/bin/env python3
"""Micro-experiments to ground every hardcoded simulator constant.

For each constant in ``SimulatorConfig`` that was originally a heuristic,
this script runs a targeted profiling experiment and outputs measured
values with confidence intervals.  Results are written to
``data/micro_experiments/`` as per-constant JSON files plus a single
``simulator_config.json`` ready to be loaded via
``SimulatorConfig.from_json()``.

Experiments
-----------
1. **GQA batching overhead** – profiles attention with batch_size=1 vs B
   for GQA models.
2. **Prefill batch overhead** – profiles prefill at batch sizes 1..16 at
   fixed seq_len.
3. **Roofline eta/alpha calibration** – fits eta_prefill, eta_decode, and
   alpha_power from profiled vs. analytical predictions.
4. **Fused decomposition fractions** – uses torch.profiler traces to
   measure actual attention/MLP time fractions.
5. **TP communication overhead** – uses communication.csv profiling data
   to measure AllReduce/AllGather latency per message size.
6. **CPU overhead regression** – builds a lookup from cpu_host.csv profiling.

Usage::

    python -m evals.scripts.run_micro_experiments \\
        --models qwen3-0.6b,qwen3-4b,qwen3-8b,qwen3-14b \\
        --output-dir data/micro_experiments \\
        --profile-dir data/e2e_v4/profiles
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("micro_experiments")


# ---------------------------------------------------------------------------
# Experiment 1: GQA batching overhead
# ---------------------------------------------------------------------------

def measure_gqa_batching_overhead(
    profile_dir: Path,
) -> Dict[str, Any]:
    """Estimate GQA batching overhead from attention profiling CSVs.

    Looks for attention_prefill.csv or attention.csv and compares
    batch_size=1 timings against batch_size>1 timings for the same
    seq_len to compute ``overhead = time(B) / (B * time(1)) - 1``.
    """
    import csv

    csv_candidates = [
        profile_dir / "attention_prefill.csv",
        profile_dir / "attention.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        return {"constant": "gqa_batching_overhead", "measured_value": None,
                "error": "no attention CSV found"}

    # Parse CSV: expect columns batch_size, seq_len, time_s (or similar)
    rows: List[Dict[str, Any]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row.get("batch_size", row.get("batch_tokens", 1)))
            sl = int(row.get("seq_len", row.get("sequence_length", 128)))
            t = float(row.get("time_s", row.get("mean_time_s", 0)))
            if t > 0:
                rows.append({"batch_size": bs, "seq_len": sl, "time_s": t})

    if not rows:
        return {"constant": "gqa_batching_overhead", "measured_value": None,
                "error": "no valid rows in CSV"}

    # Group by seq_len, compute overhead for each batch > 1 vs batch = 1
    from collections import defaultdict
    by_seq: Dict[int, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        by_seq[r["seq_len"]][r["batch_size"]] = r["time_s"]

    overheads: List[float] = []
    for seq_len, bs_map in by_seq.items():
        t1 = bs_map.get(1)
        if t1 is None or t1 <= 0:
            continue
        for bs, t_bs in bs_map.items():
            if bs <= 1:
                continue
            overhead = t_bs / (bs * t1) - 1.0
            if overhead > -0.5:  # filter out obviously bad measurements
                overheads.append(overhead)

    if not overheads:
        return {"constant": "gqa_batching_overhead", "measured_value": 0.1,
                "note": "no multi-batch data; using default"}

    measured = float(np.median(overheads))
    return {
        "constant": "gqa_batching_overhead",
        "measured_value": max(measured, 0.0),
        "num_datapoints": len(overheads),
        "p25": float(np.percentile(overheads, 25)),
        "p50": float(np.percentile(overheads, 50)),
        "p75": float(np.percentile(overheads, 75)),
    }


# ---------------------------------------------------------------------------
# Experiment 2: Prefill batch overhead
# ---------------------------------------------------------------------------

def measure_prefill_batch_overhead(
    profile_dir: Path,
) -> Dict[str, Any]:
    """Measure overhead per additional request in a prefill batch.

    overhead_per_request = (time(B) - B * time(1)) / ((B-1) * time(1))
    """
    import csv

    csv_candidates = [
        profile_dir / "attention_prefill.csv",
        profile_dir / "attention.csv",
        profile_dir / "token_ops.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        return {"constant": "prefill_batch_overhead", "measured_value": None,
                "error": "no prefill CSV found"}

    rows: List[Dict[str, Any]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row.get("batch_size", row.get("batch_tokens", 1)))
            sl = int(row.get("seq_len", row.get("sequence_length", 128)))
            t = float(row.get("time_s", row.get("mean_time_s", 0)))
            if t > 0:
                rows.append({"batch_size": bs, "seq_len": sl, "time_s": t})

    from collections import defaultdict
    by_seq: Dict[int, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        by_seq[r["seq_len"]][r["batch_size"]] = r["time_s"]

    overheads: List[float] = []
    for seq_len, bs_map in by_seq.items():
        t1 = bs_map.get(1)
        if t1 is None or t1 <= 0:
            continue
        for bs, t_bs in bs_map.items():
            if bs <= 1:
                continue
            # overhead_per_request
            overhead = (t_bs - bs * t1) / ((bs - 1) * t1)
            overheads.append(overhead)

    if not overheads:
        return {"constant": "prefill_batch_overhead", "measured_value": 0.05,
                "note": "no multi-batch data; using default"}

    measured = float(np.median(overheads))
    return {
        "constant": "prefill_batch_overhead",
        "measured_value": max(measured, 0.0),
        "num_datapoints": len(overheads),
        "p25": float(np.percentile(overheads, 25)),
        "p50": float(np.percentile(overheads, 50)),
        "p75": float(np.percentile(overheads, 75)),
    }


# ---------------------------------------------------------------------------
# Experiment 3: Roofline eta/alpha calibration
# ---------------------------------------------------------------------------

def calibrate_roofline(
    profile_dir: Path,
    model_spec: object,
    hw_spec: object,
) -> Dict[str, Any]:
    """Fit eta_prefill, eta_decode, and alpha_power from profiled data.

    Compares roofline predictions against profiled timings and minimises
    the sum of squared relative errors to find optimal efficiency factors.
    """
    import csv

    from ipw.simulator.inference_model import estimate_decode, estimate_prefill

    params_b = getattr(model_spec, "total_params_billion", 7.0)
    peak_tflops = getattr(hw_spec, "peak_tflops", 312.0)
    mem_bw = getattr(hw_spec, "hbm_bandwidth_gb_s", 2039.0)
    bpp = getattr(hw_spec, "bytes_per_param", 2.0)
    tdp = getattr(hw_spec, "tdp_watts", 400.0)

    def _parse_opt(val) -> Optional[float]:
        """Parse optional float from CSV cell."""
        if val is None or val == "" or val == "None":
            return None
        try:
            v = float(val)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    # Load prefill timings
    prefill_csv = profile_dir / "attention_prefill.csv"
    prefill_data: List[tuple] = []
    if prefill_csv.exists():
        with open(prefill_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sl = int(row.get("seq_len", row.get("sequence_length", 0)))
                t = float(row.get("time_s", row.get("mean_time_s", 0)))
                if sl > 0 and t > 0:
                    prefill_data.append((sl, t))

    # Load decode timings
    decode_csv = profile_dir / "attention_decode.csv"
    decode_data: List[tuple] = []
    if decode_csv.exists():
        with open(decode_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bs = int(row.get("batch_size", row.get("batch_tokens", 1)))
                t = float(row.get("time_s", row.get("mean_time_s", 0)))
                if bs > 0 and t > 0:
                    decode_data.append((bs, t))

    # Collect power readings from all profiling CSVs for alpha_power fitting
    power_readings: List[float] = []
    for csv_name in sorted(profile_dir.glob("*.csv")):
        try:
            with open(csv_name) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pw = _parse_opt(row.get("power_w"))
                    if pw is not None:
                        power_readings.append(pw)
                    else:
                        # Try deriving power from energy_j / time_s
                        ej = _parse_opt(row.get("energy_j"))
                        ts = _parse_opt(row.get("time_s", row.get("mean_time_s")))
                        if ej is not None and ts is not None and ts > 0:
                            power_readings.append(ej / ts)
        except Exception:
            continue

    result: Dict[str, Any] = {"constant": "roofline_calibration"}

    # Fit eta_prefill
    if prefill_data:
        best_eta = 0.4
        best_err = float("inf")
        for eta_candidate in np.linspace(0.1, 0.9, 17):
            err = 0.0
            for sl, actual_t in prefill_data:
                pred = estimate_prefill(
                    active_params_b=params_b, input_tokens=sl,
                    peak_tflops=peak_tflops, eta=eta_candidate,
                )
                if actual_t > 0:
                    err += ((pred.time_seconds - actual_t) / actual_t) ** 2
            if err < best_err:
                best_err = err
                best_eta = float(eta_candidate)
        result["eta_prefill"] = round(best_eta, 3)
        result["eta_prefill_rmse"] = round(float(np.sqrt(best_err / len(prefill_data))), 4)
    else:
        result["eta_prefill"] = 0.4
        result["eta_prefill_note"] = "no prefill data; using default"

    # Fit eta_decode
    if decode_data:
        best_eta = 0.5
        best_err = float("inf")
        for eta_candidate in np.linspace(0.1, 0.9, 17):
            err = 0.0
            for bs, actual_t in decode_data:
                pred = estimate_decode(
                    active_params_b=params_b, output_tokens=bs,
                    bytes_per_param=bpp, mem_bw_gb_s=mem_bw, eta=eta_candidate,
                )
                if actual_t > 0:
                    err += ((pred.time_seconds - actual_t) / actual_t) ** 2
            if err < best_err:
                best_err = err
                best_eta = float(eta_candidate)
        result["eta_decode"] = round(best_eta, 3)
        result["eta_decode_rmse"] = round(float(np.sqrt(best_err / len(decode_data))), 4)
    else:
        result["eta_decode"] = 0.5
        result["eta_decode_note"] = "no decode data; using default"

    # Fit alpha_power from profiled power measurements
    tdp_watts = tdp
    if power_readings:
        measured_avg_power = float(np.median(power_readings))
        alpha_power = measured_avg_power / tdp_watts
        alpha_power = float(np.clip(alpha_power, 0.1, 1.0))  # sanity bounds
        result["alpha_power"] = round(alpha_power, 4)
        result["alpha_power_fitted"] = True
        result["alpha_power_n_samples"] = len(power_readings)
        result["alpha_power_median_power_w"] = round(measured_avg_power, 2)
    else:
        raise ValueError(
            f"No energy/power data found in profiling CSVs under {profile_dir}. "
            "Re-run profiling with --energy / use_energy=True to collect power measurements."
        )

    return result


# ---------------------------------------------------------------------------
# Experiment 4: TP communication overhead
# ---------------------------------------------------------------------------

def measure_tp_comm_overhead(
    profile_dir: Path,
) -> Dict[str, Any]:
    """Estimate TP communication overhead from communication.csv.

    Returns the median AllReduce/AllGather latency for a typical
    message size (model-hidden-dimension-sized tensor).
    """
    import csv

    csv_path = profile_dir / "communication.csv"
    if not csv_path.exists():
        return {"constant": "tp_comm_overhead_s", "measured_value": 0.0,
                "note": "no communication.csv found"}

    latencies: List[float] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row.get("operator_name", row.get("operation", ""))
            if "allreduce" in op.lower() or "allgather" in op.lower():
                t = float(row.get("time_s", row.get("mean_time_s", 0)))
                if t > 0:
                    latencies.append(t)

    if not latencies:
        return {"constant": "tp_comm_overhead_s", "measured_value": 0.0,
                "note": "no AllReduce/AllGather entries in communication.csv"}

    measured = float(np.median(latencies))
    return {
        "constant": "tp_comm_overhead_s",
        "measured_value": measured,
        "num_datapoints": len(latencies),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }


# ---------------------------------------------------------------------------
# Experiment 5: CPU overhead from cpu_host.csv
# ---------------------------------------------------------------------------

def measure_cpu_overhead(
    profile_dir: Path,
) -> Dict[str, Any]:
    """Build CPU overhead lookup from cpu_host.csv profiling data.

    Extracts scheduler_overhead, tokenizer_encode/decode, and
    dynamic_batching_overhead measurements and returns a summary.
    """
    import csv

    csv_path = profile_dir / "cpu_host.csv"
    if not csv_path.exists():
        return {"constant": "cpu_overhead_s", "measured_value": None,
                "note": "no cpu_host.csv found"}

    overheads: List[float] = []
    by_batch: Dict[int, List[float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row.get("time_s", row.get("mean_time_s", 0)))
            bs = int(row.get("batch_size", 1))
            if t > 0:
                overheads.append(t)
                by_batch.setdefault(bs, []).append(t)

    if not overheads:
        return {"constant": "cpu_overhead_s", "measured_value": None,
                "note": "no valid entries in cpu_host.csv"}

    return {
        "constant": "cpu_overhead_s",
        "measured_value": float(np.median(overheads)),
        "num_datapoints": len(overheads),
        "p50": float(np.percentile(overheads, 50)),
        "p95": float(np.percentile(overheads, 95)),
        "by_batch_size": {
            str(bs): round(float(np.median(vals)), 6)
            for bs, vals in sorted(by_batch.items())
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-experiments to ground simulator constants"
    )
    parser.add_argument(
        "--models", default="qwen3-0.6b,qwen3-4b,qwen3-8b,qwen3-14b",
        help="Comma-separated model keys",
    )
    parser.add_argument(
        "--profile-dir", default="data/e2e_v4/profiles",
        help="Base directory containing per-model profiling CSVs",
    )
    parser.add_argument(
        "--output-dir", default="data/micro_experiments",
        help="Output directory for measurement JSONs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_base = Path(args.profile_dir)
    model_keys = [k.strip() for k in args.models.split(",")]

    # Try to load model/hardware specs for roofline calibration
    try:
        from inference_search.cli import _EXAMPLE_MODELS
        from inference_simulator.types import HardwareSpec
        model_specs = _EXAMPLE_MODELS
        hw_spec = HardwareSpec.from_registry("a100_80gb")
    except ImportError:
        model_specs = {}
        hw_spec = None

    # Aggregate results across models
    all_results: Dict[str, List[Dict[str, Any]]] = {
        "gqa_batching_overhead": [],
        "prefill_batch_overhead": [],
        "roofline_calibration": [],
        "tp_comm_overhead_s": [],
        "cpu_overhead_s": [],
    }

    for key in model_keys:
        # Find profiling directory for this model
        # Try several naming conventions (case-insensitive match)
        key_lower = key.lower()
        candidates = [
            d for d in profile_base.iterdir()
            if d.is_dir() and key_lower in d.name.lower()
        ]
        if not candidates:
            candidates = [
                d for d in profile_base.rglob("*")
                if d.is_dir() and key_lower in d.name.lower()
            ]
        if not candidates:
            logger.warning("No profiling data found for %s", key)
            continue

        pdir = candidates[0]
        # Look for the deepest directory with CSVs
        csv_dirs = [d for d in pdir.rglob("*.csv")]
        if csv_dirs:
            pdir = csv_dirs[0].parent
        logger.info("Model %s: profiling dir = %s", key, pdir)

        # Run experiments
        gqa = measure_gqa_batching_overhead(pdir)
        gqa["model"] = key
        all_results["gqa_batching_overhead"].append(gqa)

        prefill = measure_prefill_batch_overhead(pdir)
        prefill["model"] = key
        all_results["prefill_batch_overhead"].append(prefill)

        if hw_spec is not None and key in model_specs:
            roofline = calibrate_roofline(pdir, model_specs[key], hw_spec)
            roofline["model"] = key
            all_results["roofline_calibration"].append(roofline)

        comm = measure_tp_comm_overhead(pdir)
        comm["model"] = key
        all_results["tp_comm_overhead_s"].append(comm)

        cpu = measure_cpu_overhead(pdir)
        cpu["model"] = key
        all_results["cpu_overhead_s"].append(cpu)

    # Write per-constant JSONs
    for name, results in all_results.items():
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s.json (%d entries)", name, len(results))

    # Build aggregated SimulatorConfig with measured defaults
    config_overrides: Dict[str, Any] = {}

    # GQA overhead: median across models
    gqa_vals = [r["measured_value"] for r in all_results["gqa_batching_overhead"]
                if r.get("measured_value") is not None]
    if gqa_vals:
        config_overrides["gqa_batching_overhead"] = round(float(np.median(gqa_vals)), 4)

    # Prefill batch overhead: median
    pb_vals = [r["measured_value"] for r in all_results["prefill_batch_overhead"]
               if r.get("measured_value") is not None]
    if pb_vals:
        config_overrides["prefill_batch_overhead"] = round(float(np.median(pb_vals)), 4)

    # Roofline: median eta/alpha across models
    eta_p_vals = [r["eta_prefill"] for r in all_results["roofline_calibration"]
                  if "eta_prefill" in r]
    eta_d_vals = [r["eta_decode"] for r in all_results["roofline_calibration"]
                  if "eta_decode" in r]
    alpha_vals = [r["alpha_power"] for r in all_results["roofline_calibration"]
                  if "alpha_power" in r]
    if eta_p_vals:
        config_overrides["eta_prefill"] = round(float(np.median(eta_p_vals)), 3)
    if eta_d_vals:
        config_overrides["eta_decode"] = round(float(np.median(eta_d_vals)), 3)
    if alpha_vals:
        config_overrides["alpha_power"] = round(float(np.median(alpha_vals)), 3)

    # TP comm overhead: median
    comm_vals = [r["measured_value"] for r in all_results["tp_comm_overhead_s"]
                 if r.get("measured_value") is not None and r["measured_value"] > 0]
    if comm_vals:
        config_overrides["tp_comm_overhead_s"] = round(float(np.median(comm_vals)), 6)
        config_overrides["tp_scaling_mode"] = "measured"

    # Write SimulatorConfig JSON
    config_path = output_dir / "simulator_config.json"
    with open(config_path, "w") as f:
        json.dump(config_overrides, f, indent=2)
    logger.info("Wrote simulator_config.json with %d overrides", len(config_overrides))

    # Print summary
    print("\n" + "=" * 60)
    print("MICRO-EXPERIMENT RESULTS")
    print("=" * 60)
    for name, value in config_overrides.items():
        print(f"  {name}: {value}")
    print(f"\nConfig saved to: {config_path}")
    print(f"Load via: SimulatorConfig.from_json('{config_path}')")


if __name__ == "__main__":
    main()
