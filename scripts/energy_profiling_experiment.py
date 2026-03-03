#!/usr/bin/env python3
"""Energy Profiling Experiment: Characterize how kernel implementation
patterns affect GPU energy consumption.

Runs all KernelBench reference kernels, measures energy + rich GPU telemetry,
extracts source code features, and produces an analysis report.

Designed to complete in <2 hours on 1x L40S GPU.

Usage:
    python scripts/energy_profiling_experiment.py \
        --kernelbench-root external/KernelBench \
        --output-dir results/energy_profiling \
        --levels 1,2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Source code feature extraction
# ---------------------------------------------------------------------------

@dataclass
class KernelSourceFeatures:
    """Features extracted from kernel source code via AST/regex."""
    num_lines: int = 0
    num_torch_ops: int = 0
    uses_custom_cuda: bool = False
    uses_triton: bool = False
    uses_shared_memory: bool = False
    uses_inplace_ops: bool = False
    num_matmul_calls: int = 0
    num_conv_calls: int = 0
    num_relu_calls: int = 0
    num_linear_calls: int = 0
    num_norm_calls: int = 0
    num_softmax_calls: int = 0
    has_backward: bool = False
    num_custom_autograd: int = 0
    num_for_loops: int = 0
    estimated_param_count: str = "unknown"
    uses_view_reshape: bool = False
    uses_contiguous: bool = False
    uses_transpose: bool = False
    num_tensor_creates: int = 0


def extract_source_features(source: str) -> KernelSourceFeatures:
    """Extract structural features from kernel source code."""
    f = KernelSourceFeatures()
    if not source:
        return f

    lines = source.strip().split("\n")
    f.num_lines = len(lines)
    low = source.lower()

    f.num_torch_ops = len(re.findall(r"torch\.\w+", source))
    f.uses_custom_cuda = bool(re.search(r'(cuda_source|load_inline|\.cu["\x27])', source))
    f.uses_triton = "triton" in low or "@triton.jit" in source
    f.uses_shared_memory = bool(re.search(r"(__shared__|shared_memory|tl\.load)", source))
    f.uses_inplace_ops = bool(re.search(r"\.(add_|mul_|sub_|div_|relu_|zero_)\(", source))

    f.num_matmul_calls = len(re.findall(r"(torch\.matmul|torch\.mm|torch\.bmm|@\s)", source))
    f.num_conv_calls = len(re.findall(r"(nn\.Conv[123]d|F\.conv[123]d)", source))
    f.num_relu_calls = len(re.findall(r"(F\.relu|nn\.ReLU|torch\.relu)", source))
    f.num_linear_calls = len(re.findall(r"(nn\.Linear|F\.linear)", source))
    f.num_norm_calls = len(re.findall(
        r"(nn\.BatchNorm|nn\.LayerNorm|nn\.GroupNorm|nn\.InstanceNorm|F\.batch_norm|F\.layer_norm)",
        source,
    ))
    f.num_softmax_calls = len(re.findall(r"(F\.softmax|nn\.Softmax|torch\.softmax)", source))
    f.has_backward = "def backward" in source
    f.num_custom_autograd = len(re.findall(r"torch\.autograd\.Function", source))
    f.num_for_loops = len(re.findall(r"\bfor\s+\w+\s+in\s+", source))

    param_match = re.search(r"def get_init_inputs.*?return\s*\[([^\]]*)\]", source, re.DOTALL)
    if param_match:
        f.estimated_param_count = param_match.group(1).strip() or "none"

    f.uses_view_reshape = bool(re.search(r"\.(view|reshape)\(", source))
    f.uses_contiguous = ".contiguous()" in source
    f.uses_transpose = bool(re.search(r"\.(transpose|permute|t\(\))", source))
    f.num_tensor_creates = len(re.findall(r"torch\.(zeros|ones|randn|rand|empty|full|arange)\(", source))

    return f


# ---------------------------------------------------------------------------
# Kernel profiling
# ---------------------------------------------------------------------------

@dataclass
class KernelProfile:
    """Complete profile for one kernel."""
    task_id: str
    kernel_type: str
    level: int
    runtime_ms: float | None = None
    energy_j: float | None = None
    avg_power_w: float | None = None
    avg_gpu_util_pct: float | None = None
    avg_mem_util_pct: float | None = None
    avg_temp_c: float | None = None
    avg_sm_clock_mhz: float | None = None
    avg_mem_clock_mhz: float | None = None
    avg_mem_used_mb: float | None = None
    eval_wall_s: float | None = None
    energy_per_ms: float | None = None
    power_eff_ms_per_j: float | None = None
    compute_to_mem_ratio: float | None = None
    source_features: dict = field(default_factory=dict)
    error: str | None = None


def profile_reference_kernel(
    ref_arch_src: str,
    num_trials: int = 3,
    device_index: int = 0,
) -> dict:
    """Load and run a reference kernel directly, measuring energy + telemetry."""
    import torch
    import torch.nn as nn
    if not torch.cuda.is_available():
        return {"error": "no CUDA"}

    from kite.measurement.nvml_power import NvmlRichSampler, GpuSample
    from kite.measurement.energy_integrate import integrate_rich_energy

    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{device_index}")

    # Load the Model class and its input generator from the reference source
    namespace: dict = {}
    try:
        exec(ref_arch_src, namespace)
    except Exception as exc:
        return {"error": f"exec failed: {exc}"}

    ModelClass = namespace.get("Model")
    get_init = namespace.get("get_init_inputs")
    get_inputs = namespace.get("get_inputs")
    if ModelClass is None or get_inputs is None:
        return {"error": "missing Model or get_inputs in source"}

    try:
        init_args = get_init() if get_init else []
        model = ModelClass(*init_args).to(device).eval()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    except Exception as exc:
        return {"error": f"model init failed: {exc}"}

    # Warmup
    try:
        with torch.no_grad():
            for _ in range(2):
                model(*inputs)
        torch.cuda.synchronize(device)
    except Exception as exc:
        return {"error": f"warmup failed: {exc}"}

    # Timed runs with NVML sampling
    sampler = NvmlRichSampler(device_index=device_index, sampling_interval_ms=20.0)
    sampler.start()
    t0 = time.perf_counter()

    runtimes_ms = []
    try:
        for _ in range(num_trials):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with torch.no_grad():
                model(*inputs)
            end_evt.record()
            torch.cuda.synchronize(device)
            runtimes_ms.append(start_evt.elapsed_time(end_evt))
    except Exception as exc:
        sampler.stop()
        sampler.close()
        return {"error": f"runtime measurement failed: {exc}"}

    t1 = time.perf_counter()
    rich_samples = sampler.stop()
    duration_s = max(0.001, t1 - t0)

    if len(rich_samples) < 2:
        fb = sampler.read_sample()
        rich_samples = [
            GpuSample(0.0, fb.power_w, fb.gpu_util_pct, fb.mem_util_pct,
                      fb.temp_c, fb.sm_clock_mhz, fb.mem_clock_mhz, fb.mem_used_mb),
            GpuSample(duration_s, fb.power_w, fb.gpu_util_pct, fb.mem_util_pct,
                      fb.temp_c, fb.sm_clock_mhz, fb.mem_clock_mhz, fb.mem_used_mb),
        ]
    window = integrate_rich_energy(rich_samples)
    sampler.close()

    # Cleanup
    del model, inputs
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    median_rt = sorted(runtimes_ms)[len(runtimes_ms) // 2] if runtimes_ms else None

    return {
        "energy_j": window.energy_j,
        "runtime_ms": median_rt,
        "runtime_all_ms": runtimes_ms,
        "avg_power_w": window.avg_power_w,
        "avg_gpu_util_pct": window.avg_gpu_util_pct,
        "avg_mem_util_pct": window.avg_mem_util_pct,
        "avg_temp_c": window.avg_temp_c,
        "avg_sm_clock_mhz": window.avg_sm_clock_mhz,
        "avg_mem_clock_mhz": window.avg_mem_clock_mhz,
        "avg_mem_used_mb": window.avg_mem_used_mb,
        "eval_wall_s": duration_s,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _safe_mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _safe_std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _pct(a: float | None, b: float | None) -> str:
    if a is None or b is None or b == 0:
        return "n/a"
    return f"{a/b*100:.1f}%"


def analyze_profiles(profiles: list[KernelProfile]) -> dict:
    """Run full analysis on collected profiles."""
    successful = [p for p in profiles if p.error is None and p.energy_j is not None]
    by_type: dict[str, list[KernelProfile]] = defaultdict(list)
    for p in successful:
        by_type[p.kernel_type].append(p)

    # Per-type statistics
    type_stats = {}
    for kt, ps in sorted(by_type.items()):
        energies = [p.energy_j for p in ps if p.energy_j]
        runtimes = [p.runtime_ms for p in ps if p.runtime_ms]
        powers = [p.avg_power_w for p in ps if p.avg_power_w]
        gpu_utils = [p.avg_gpu_util_pct for p in ps if p.avg_gpu_util_pct is not None]
        mem_utils = [p.avg_mem_util_pct for p in ps if p.avg_mem_util_pct is not None]
        epm = [p.energy_per_ms for p in ps if p.energy_per_ms is not None]
        c2m = [p.compute_to_mem_ratio for p in ps if p.compute_to_mem_ratio is not None]
        temps = [p.avg_temp_c for p in ps if p.avg_temp_c is not None]

        type_stats[kt] = {
            "count": len(ps),
            "energy_j": {"mean": _safe_mean(energies), "std": _safe_std(energies),
                         "min": min(energies) if energies else None,
                         "max": max(energies) if energies else None},
            "runtime_ms": {"mean": _safe_mean(runtimes), "std": _safe_std(runtimes),
                           "min": min(runtimes) if runtimes else None,
                           "max": max(runtimes) if runtimes else None},
            "avg_power_w": {"mean": _safe_mean(powers), "std": _safe_std(powers)},
            "avg_gpu_util_pct": {"mean": _safe_mean(gpu_utils), "std": _safe_std(gpu_utils)},
            "avg_mem_util_pct": {"mean": _safe_mean(mem_utils), "std": _safe_std(mem_utils)},
            "energy_per_ms": {"mean": _safe_mean(epm), "std": _safe_std(epm)},
            "compute_to_mem_ratio": {"mean": _safe_mean(c2m), "std": _safe_std(c2m)},
            "avg_temp_c": {"mean": _safe_mean(temps)},
        }

    # Within-type variance analysis: for types with 5+ kernels,
    # how much does energy vary for similar runtime?
    variance_analysis = {}
    for kt, ps in by_type.items():
        if len(ps) < 3:
            continue
        energies = [p.energy_j for p in ps if p.energy_j]
        runtimes = [p.runtime_ms for p in ps if p.runtime_ms]
        if not energies or not runtimes:
            continue
        mean_e = sum(energies) / len(energies)
        mean_r = sum(runtimes) / len(runtimes)
        cv_energy = (_safe_std(energies) / mean_e * 100) if mean_e > 0 and _safe_std(energies) else None
        cv_runtime = (_safe_std(runtimes) / mean_r * 100) if mean_r > 0 and _safe_std(runtimes) else None
        variance_analysis[kt] = {
            "n": len(energies),
            "energy_cv_pct": cv_energy,
            "runtime_cv_pct": cv_runtime,
            "energy_range_ratio": max(energies) / min(energies) if min(energies) > 0 else None,
            "runtime_range_ratio": max(runtimes) / min(runtimes) if min(runtimes) > 0 else None,
        }

    # Source feature correlations with energy
    feature_energy_corr = _compute_feature_correlations(successful)

    # Compute/memory bound classification
    bound_classification = {}
    for p in successful:
        if p.avg_gpu_util_pct is not None and p.avg_mem_util_pct is not None:
            if p.avg_gpu_util_pct > p.avg_mem_util_pct * 1.5:
                bound = "compute_bound"
            elif p.avg_mem_util_pct > p.avg_gpu_util_pct * 1.5:
                bound = "memory_bound"
            else:
                bound = "balanced"
            bound_classification[p.task_id] = {
                "bound_type": bound,
                "gpu_util": p.avg_gpu_util_pct,
                "mem_util": p.avg_mem_util_pct,
                "energy_per_ms": p.energy_per_ms,
                "kernel_type": p.kernel_type,
            }

    bound_summary = defaultdict(lambda: {"count": 0, "types": defaultdict(int)})
    for v in bound_classification.values():
        bt = v["bound_type"]
        bound_summary[bt]["count"] += 1
        bound_summary[bt]["types"][v["kernel_type"]] += 1

    # Most/least energy-efficient kernels
    ranked = sorted(
        [p for p in successful if p.energy_per_ms is not None],
        key=lambda p: p.energy_per_ms,
    )
    top_efficient = [
        {"task_id": p.task_id, "type": p.kernel_type, "energy_per_ms": p.energy_per_ms,
         "runtime_ms": p.runtime_ms, "energy_j": p.energy_j}
        for p in ranked[:10]
    ]
    least_efficient = [
        {"task_id": p.task_id, "type": p.kernel_type, "energy_per_ms": p.energy_per_ms,
         "runtime_ms": p.runtime_ms, "energy_j": p.energy_j}
        for p in ranked[-10:]
    ]

    return {
        "summary": {
            "total_profiled": len(profiles),
            "successful": len(successful),
            "failed": len(profiles) - len(successful),
            "kernel_types": len(by_type),
        },
        "per_type_stats": type_stats,
        "within_type_variance": variance_analysis,
        "feature_energy_correlations": feature_energy_corr,
        "bound_classification_summary": {k: dict(v) for k, v in bound_summary.items()},
        "top_10_most_efficient": top_efficient,
        "top_10_least_efficient": least_efficient,
    }


def _compute_feature_correlations(profiles: list[KernelProfile]) -> dict:
    """Compute Pearson correlation between source features and energy metrics."""
    if len(profiles) < 5:
        return {}

    feature_keys = [
        "num_lines", "num_torch_ops", "num_matmul_calls", "num_conv_calls",
        "num_linear_calls", "num_norm_calls", "num_for_loops", "num_tensor_creates",
    ]
    bool_keys = [
        "uses_custom_cuda", "uses_triton", "uses_shared_memory",
        "uses_inplace_ops", "uses_view_reshape", "uses_contiguous", "uses_transpose",
    ]

    results = {}
    for fk in feature_keys + bool_keys:
        xs = []
        ys = []
        for p in profiles:
            fv = p.source_features.get(fk)
            if fv is None or p.energy_per_ms is None:
                continue
            xs.append(float(fv) if isinstance(fv, bool) else float(fv))
            ys.append(p.energy_per_ms)
        if len(xs) < 5:
            continue
        corr = _pearson(xs, ys)
        if corr is not None:
            results[fk] = round(corr, 3)

    return dict(sorted(results.items(), key=lambda kv: abs(kv[1]), reverse=True))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(analysis: dict, profiles: list[KernelProfile]) -> str:
    """Generate and print a human-readable report. Returns the text."""
    lines = []

    def p(s=""):
        lines.append(s)

    s = analysis["summary"]
    p("=" * 90)
    p("KERNEL ENERGY PROFILING EXPERIMENT -- RESULTS")
    p("=" * 90)
    p(f"Total kernels profiled: {s['total_profiled']}")
    p(f"Successful:             {s['successful']}")
    p(f"Failed:                 {s['failed']}")
    p(f"Kernel types observed:  {s['kernel_types']}")
    p()

    p("--- PER-TYPE ENERGY SUMMARY ---")
    p(f"{'Type':<16} {'N':>4} {'Avg Energy(J)':>13} {'Avg RT(ms)':>10} "
      f"{'E/ms':>8} {'GPU%':>6} {'Mem%':>6} {'Power(W)':>9}")
    p("-" * 90)

    for kt, st in sorted(analysis["per_type_stats"].items(),
                          key=lambda kv: kv[1]["energy_j"]["mean"] or 0, reverse=True):
        n = st["count"]
        ae = st["energy_j"]["mean"]
        ar = st["runtime_ms"]["mean"]
        epm = st["energy_per_ms"]["mean"]
        gu = st["avg_gpu_util_pct"]["mean"]
        mu = st["avg_mem_util_pct"]["mean"]
        pw = st["avg_power_w"]["mean"]
        p(f"{kt:<16} {n:>4} {ae:>13.4f} {ar or 0:>10.2f} "
          f"{epm or 0:>8.4f} {gu or 0:>6.1f} {mu or 0:>6.1f} {pw or 0:>9.1f}")
    p()

    p("--- WITHIN-TYPE ENERGY VARIANCE (key finding: same op, different energy) ---")
    p(f"{'Type':<16} {'N':>4} {'Energy CV%':>10} {'Runtime CV%':>11} "
      f"{'Energy Range':>13} {'RT Range':>10}")
    p("-" * 75)
    for kt, va in sorted(analysis["within_type_variance"].items(),
                          key=lambda kv: kv[1].get("energy_cv_pct") or 0, reverse=True):
        ecv = f"{va['energy_cv_pct']:.1f}%" if va.get("energy_cv_pct") else "n/a"
        rcv = f"{va['runtime_cv_pct']:.1f}%" if va.get("runtime_cv_pct") else "n/a"
        er = f"{va['energy_range_ratio']:.1f}x" if va.get("energy_range_ratio") else "n/a"
        rr = f"{va['runtime_range_ratio']:.1f}x" if va.get("runtime_range_ratio") else "n/a"
        p(f"{kt:<16} {va['n']:>4} {ecv:>10} {rcv:>11} {er:>13} {rr:>10}")
    p()

    p("--- COMPUTE vs MEMORY BOUND CLASSIFICATION ---")
    for bt, bd in analysis["bound_classification_summary"].items():
        types_str = ", ".join(f"{k}={v}" for k, v in sorted(bd.get("types", {}).items(),
                                                             key=lambda x: -x[1]))
        p(f"  {bt:<16} {bd['count']:>3} kernels  ({types_str})")
    p()

    p("--- SOURCE FEATURE CORRELATIONS WITH ENERGY EFFICIENCY (energy_per_ms) ---")
    for feat, corr in list(analysis["feature_energy_correlations"].items())[:10]:
        direction = "higher energy" if corr > 0 else "lower energy"
        p(f"  {feat:<25} r={corr:+.3f}  ({direction})")
    p()

    p("--- TOP 10 MOST ENERGY-EFFICIENT KERNELS (lowest J/ms) ---")
    for i, k in enumerate(analysis["top_10_most_efficient"]):
        p(f"  {i+1:>2}. {k['task_id']:<30} {k['type']:<14} "
          f"E/ms={k['energy_per_ms']:.5f}  rt={k['runtime_ms']:.2f}ms  E={k['energy_j']:.4f}J")
    p()

    p("--- TOP 10 LEAST ENERGY-EFFICIENT KERNELS (highest J/ms) ---")
    for i, k in enumerate(analysis["top_10_least_efficient"]):
        p(f"  {i+1:>2}. {k['task_id']:<30} {k['type']:<14} "
          f"E/ms={k['energy_per_ms']:.5f}  rt={k['runtime_ms']:.2f}ms  E={k['energy_j']:.4f}J")
    p()

    # Key findings
    p("=" * 90)
    p("KEY FINDINGS")
    p("=" * 90)
    va = analysis["within_type_variance"]
    high_var_types = [kt for kt, v in va.items()
                      if v.get("energy_cv_pct") and v["energy_cv_pct"] > 30]
    if high_var_types:
        p(f"1. High energy variance within type (>30% CV): {', '.join(high_var_types)}")
        p("   -> Same operation type, very different energy. Implementation matters.")
    else:
        p("1. Most kernel types show relatively consistent energy usage.")

    ts = analysis["per_type_stats"]
    if ts:
        epm_vals = [(kt, st["energy_per_ms"]["mean"])
                    for kt, st in ts.items() if st["energy_per_ms"]["mean"]]
        if epm_vals:
            epm_vals.sort(key=lambda x: x[1])
            p(f"2. Most energy-efficient type: {epm_vals[0][0]} ({epm_vals[0][1]:.5f} J/ms)")
            p(f"   Least energy-efficient type: {epm_vals[-1][0]} ({epm_vals[-1][1]:.5f} J/ms)")
            if epm_vals[-1][1] > 0:
                ratio = epm_vals[-1][1] / epm_vals[0][1]
                p(f"   Energy efficiency varies {ratio:.1f}x across kernel types.")

    fc = analysis["feature_energy_correlations"]
    if fc:
        top_feat = list(fc.items())[0]
        p(f"3. Strongest code feature predictor of energy: '{top_feat[0]}' (r={top_feat[1]:+.3f})")

    p("=" * 90)
    p()

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    kernelbench_root: Path,
    output_dir: Path,
    levels: list[int],
    num_trials: int = 3,
    timeout_s: float = 60.0,
) -> dict:
    from kite.adapters.kernelbench_adapter import KernelBenchAdapter

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = KernelBenchAdapter(
        kernelbench_root=kernelbench_root,
        enable_kernelbench_eval=True,
        levels=levels,
    )
    tasks = adapter.discover_tasks()
    print(f"Discovered {len(tasks)} tasks (levels={levels})")
    print(f"Output: {output_dir}")
    print(f"Timeout: {timeout_s}s, Perf trials: {num_trials}")
    print()

    # Phase 1: Profile all reference kernels
    print("=" * 60)
    print("PHASE 1: Profiling reference kernels")
    print("=" * 60)
    t_start = time.time()

    profiles: list[KernelProfile] = []
    for i, task in enumerate(tasks):
        ref_src = task.metadata.get("ref_arch_src")
        if not ref_src or not isinstance(ref_src, str):
            print(f"  [{i+1}/{len(tasks)}] {task.task_id} -- SKIP (no source)")
            continue

        features = extract_source_features(ref_src)

        print(f"  [{i+1}/{len(tasks)}] {task.task_id} ({task.kernel_type}) ...", end=" ", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(profile_reference_kernel, ref_src, num_trials)
            try:
                result = fut.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                print("TIMEOUT")
                profiles.append(KernelProfile(
                    task_id=task.task_id, kernel_type=task.kernel_type, level=task.level,
                    source_features=asdict(features), error=f"timeout after {timeout_s}s",
                ))
                continue

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            profiles.append(KernelProfile(
                task_id=task.task_id, kernel_type=task.kernel_type, level=task.level,
                source_features=asdict(features), error=result["error"],
            ))
            continue

        energy_j = result["energy_j"]
        runtime_ms = result["runtime_ms"]
        epm = energy_j / runtime_ms if energy_j and runtime_ms and runtime_ms > 0 else None
        peff = runtime_ms / energy_j if energy_j and runtime_ms and energy_j > 0 else None
        gpu_u = result["avg_gpu_util_pct"]
        mem_u = result["avg_mem_util_pct"]
        c2m = gpu_u / mem_u if gpu_u and mem_u and mem_u > 0 else None

        prof = KernelProfile(
            task_id=task.task_id,
            kernel_type=task.kernel_type,
            level=task.level,
            runtime_ms=runtime_ms,
            energy_j=energy_j,
            avg_power_w=result["avg_power_w"],
            avg_gpu_util_pct=gpu_u,
            avg_mem_util_pct=mem_u,
            avg_temp_c=result["avg_temp_c"],
            avg_sm_clock_mhz=result["avg_sm_clock_mhz"],
            avg_mem_clock_mhz=result["avg_mem_clock_mhz"],
            avg_mem_used_mb=result["avg_mem_used_mb"],
            eval_wall_s=result["eval_wall_s"],
            energy_per_ms=epm,
            power_eff_ms_per_j=peff,
            compute_to_mem_ratio=c2m,
            source_features=asdict(features),
        )
        profiles.append(prof)

        rt_str = f"rt={runtime_ms:.2f}ms" if runtime_ms else "rt=n/a"
        print(f"E={energy_j:.4f}J  {rt_str}  E/ms={epm:.5f}" if epm else f"E={energy_j:.4f}J  {rt_str}")

    profile_time = time.time() - t_start
    print(f"\nProfiling complete: {len(profiles)} kernels in {profile_time/60:.1f} min")

    # Save raw profiles
    raw_data = [asdict(p) for p in profiles]
    with open(output_dir / "kernel_profiles.json", "w") as f:
        json.dump(raw_data, f, indent=2, default=str)

    # Phase 2: Analysis
    print()
    print("=" * 60)
    print("PHASE 2: Analysis")
    print("=" * 60)

    analysis = analyze_profiles(profiles)
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    report_text = print_report(analysis, profiles)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time/60:.1f} min")
    print(f"Results saved to: {output_dir}/")
    print(f"  - kernel_profiles.json  (raw data)")
    print(f"  - analysis.json         (computed statistics)")
    print(f"  - report.txt            (human-readable report)")

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel Energy Profiling Experiment")
    parser.add_argument("--kernelbench-root", type=Path, default=ROOT / "external" / "KernelBench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "energy_profiling")
    parser.add_argument("--levels", type=str, default="1,2")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    run_experiment(
        kernelbench_root=args.kernelbench_root,
        output_dir=args.output_dir,
        levels=levels,
        num_trials=args.num_trials,
        timeout_s=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
