"""Measure energy and runtime of reference PyTorch kernels for each task.

Produces baseline_energy_profile.json -- the denominator for all
"X% energy savings" claims.  Run on a GPU node:

    python -m kite.eval.baseline_energy_eval \
        --kernelbench-root external/KernelBench \
        --output data/baseline_energy_profile.json \
        --levels 1,2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run_reference_kernel(
    ref_arch_src: str,
    num_trials: int = 5,
    device_index: int = 0,
) -> dict | None:
    """Run the reference kernel and measure energy/runtime via NVML."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
        from kite.measurement.nvml_power import NvmlRichSampler, GpuSample
        from kite.measurement.energy_integrate import integrate_rich_energy
    except Exception:
        return None

    sampler = NvmlRichSampler(device_index=device_index, sampling_interval_ms=50.0)
    sampler.start()
    t0 = time.perf_counter()
    result = None
    error = None
    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=ref_arch_src,
            num_correct_trials=1,
            num_perf_trials=num_trials,
            measure_performance=True,
            verbose=False,
            backend="inductor",
            precision=get_torch_dtype_from_string("fp32"),
            device=torch.device(f"cuda:{device_index}"),
        )
    except Exception as exc:
        error = exc
    t1 = time.perf_counter()
    rich_samples = sampler.stop()
    duration_s = max(0.0, t1 - t0)

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
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    runtime_ms = None
    if result is not None:
        runtime_us = float(getattr(result, "ref_runtime", -1.0) or -1.0)
        if runtime_us > 0:
            runtime_ms = runtime_us / 1000.0

    return {
        "energy_j": window.energy_j,
        "runtime_ms": runtime_ms,
        "avg_power_w": window.avg_power_w,
        "avg_gpu_util_pct": window.avg_gpu_util_pct,
        "avg_mem_util_pct": window.avg_mem_util_pct,
        "avg_temp_c": window.avg_temp_c,
        "avg_sm_clock_mhz": window.avg_sm_clock_mhz,
        "avg_mem_clock_mhz": window.avg_mem_clock_mhz,
        "avg_mem_used_mb": window.avg_mem_used_mb,
        "eval_wall_s": duration_s,
        "num_trials": num_trials,
        "error": str(error) if error else None,
    }


def run_baseline_eval(
    kernelbench_root: Path,
    output_path: Path,
    levels: list[int] | None = None,
    num_trials: int = 5,
    timeout_s: float = 120.0,
) -> dict:
    """Evaluate all reference kernels and save energy profile."""
    from kite.adapters.kernelbench_adapter import KernelBenchAdapter
    from kite.classification.inference_profile import (
        energy_weight_for_type,
        is_inference_critical,
    )
    from kite.utils.serialization import save_json

    adapter = KernelBenchAdapter(
        kernelbench_root=kernelbench_root,
        enable_kernelbench_eval=True,
        levels=levels,
    )
    tasks = adapter.discover_tasks()
    print(f"Evaluating {len(tasks)} reference kernels (levels={levels})")

    profile: dict[str, dict] = {}
    for i, task in enumerate(tasks):
        ref_src = task.metadata.get("ref_arch_src")
        if not ref_src or not isinstance(ref_src, str):
            print(f"  [{i+1}/{len(tasks)}] {task.task_id} -- skipped (no ref)")
            continue

        label = (
            f"  [{i+1}/{len(tasks)}] {task.task_id} "
            f"({task.kernel_type}, crit={is_inference_critical(task)})"
        )
        print(f"{label} ...", end=" ", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run_reference_kernel, ref_src, num_trials)
            try:
                result = fut.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                result = {"error": f"timeout after {timeout_s}s"}
                print("TIMEOUT")
                continue

        if result is None:
            print("SKIPPED (no CUDA)")
            continue

        entry = {
            "task_id": task.task_id,
            "kernel_type": task.kernel_type,
            "level": task.level,
            "inference_critical": is_inference_critical(task),
            "energy_weight": energy_weight_for_type(task.kernel_type),
            **result,
        }
        profile[task.task_id] = entry

        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            e = result.get("energy_j", 0)
            rt = result.get("runtime_ms")
            rt_str = f"  runtime={rt:.2f}ms" if rt else ""
            print(f"energy={e:.4f}J{rt_str}")

    save_json(output_path, profile)
    n = len(profile)
    print(f"\nBaseline profile saved to {output_path} ({n} tasks)")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure reference kernel energy baselines",
    )
    parser.add_argument(
        "--kernelbench-root", type=Path, default=Path("external/KernelBench"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/baseline_energy_profile.json"),
    )
    parser.add_argument("--levels", type=str, default="1,2")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    run_baseline_eval(
        kernelbench_root=args.kernelbench_root,
        output_path=args.output,
        levels=levels,
        num_trials=args.num_trials,
        timeout_s=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
