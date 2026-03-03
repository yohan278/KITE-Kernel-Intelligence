#!/usr/bin/env python3
"""Input Size Scaling Experiment: How does energy scale with input size?

For each kernel, runs at multiple input size scales (0.25x, 0.5x, 1x, 2x)
by modifying the size constants in the source code.

    python scripts/input_size_scaling_experiment.py \
        --kernelbench-root external/KernelBench \
        --output-dir results/input_scaling
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCALE_FACTORS = [0.25, 0.5, 1.0, 2.0]

SIZE_VAR_PATTERN = re.compile(
    r"^(batch_size|N|M|K|dim\d*|features|channels|in_channels|out_channels|"
    r"seq_len|sequence_length|num_features|hidden_size|input_size|output_size|"
    r"height|width|length|size|d_model|nhead|num_heads)\s*=\s*(\d+)",
    re.MULTILINE,
)


def scale_source(source, scale):
    def _replace(m):
        name = m.group(1)
        orig = int(m.group(2))
        new_val = max(1, int(orig * scale))
        return f"{name} = {new_val}"
    return SIZE_VAR_PATTERN.sub(_replace, source)


def profile_at_scale(ref_src, scale, num_trials=3, device_index=0):
    import torch
    if not torch.cuda.is_available():
        return {"error": "no CUDA", "scale": scale}
    from kite.measurement.nvml_power import NvmlRichSampler, GpuSample
    from kite.measurement.energy_integrate import integrate_rich_energy

    scaled_src = scale_source(ref_src, scale)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{device_index}")

    ns = {}
    try:
        exec(scaled_src, ns)
    except Exception as exc:
        return {"error": f"exec: {exc}", "scale": scale}

    ModelClass = ns.get("Model")
    get_init = ns.get("get_init_inputs")
    get_inputs = ns.get("get_inputs")
    if ModelClass is None or get_inputs is None:
        return {"error": "missing Model/get_inputs", "scale": scale}

    try:
        init_args = get_init() if get_init else []
        model = ModelClass(*init_args).to(device).eval()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                  for x in get_inputs()]
    except Exception as exc:
        return {"error": f"init: {exc}", "scale": scale}

    try:
        with torch.no_grad():
            for _ in range(2):
                model(*inputs)
        torch.cuda.synchronize(device)
    except Exception as exc:
        del model, inputs
        torch.cuda.empty_cache()
        return {"error": f"warmup: {exc}", "scale": scale}

    sampler = NvmlRichSampler(device_index=device_index, sampling_interval_ms=20.0)
    sampler.start()
    t0 = time.perf_counter()
    runtimes_ms = []
    try:
        for _ in range(num_trials):
            s_evt = torch.cuda.Event(enable_timing=True)
            e_evt = torch.cuda.Event(enable_timing=True)
            s_evt.record()
            with torch.no_grad():
                model(*inputs)
            e_evt.record()
            torch.cuda.synchronize(device)
            runtimes_ms.append(s_evt.elapsed_time(e_evt))
    except Exception as exc:
        sampler.stop(); sampler.close()
        del model, inputs; torch.cuda.empty_cache()
        return {"error": f"run: {exc}", "scale": scale}

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
    del model, inputs; torch.cuda.empty_cache()
    median_rt = sorted(runtimes_ms)[len(runtimes_ms) // 2] if runtimes_ms else None

    return {
        "scale": scale, "energy_j": window.energy_j, "runtime_ms": median_rt,
        "avg_power_w": window.avg_power_w,
        "avg_gpu_util_pct": window.avg_gpu_util_pct,
        "avg_mem_util_pct": window.avg_mem_util_pct,
        "avg_mem_used_mb": window.avg_mem_used_mb, "error": None,
    }


def run_scaling_experiment(kernelbench_root, output_dir, levels,
                           num_trials=3, timeout_s=60.0, max_kernels_per_type=3):
    from kite.adapters.kernelbench_adapter import KernelBenchAdapter
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = KernelBenchAdapter(kernelbench_root=kernelbench_root,
                                  enable_kernelbench_eval=True, levels=levels)
    tasks = adapter.discover_tasks()
    by_type = defaultdict(list)
    for t in tasks:
        by_type[t.kernel_type].append(t)
    selected = []
    for kt, tlist in sorted(by_type.items()):
        selected.extend(tlist[:max_kernels_per_type])

    print(f"Selected {len(selected)} kernels across {len(by_type)} types")
    print(f"Scales: {SCALE_FACTORS}, Total runs: ~{len(selected) * len(SCALE_FACTORS)}")
    print()

    results = []
    for i, task in enumerate(selected):
        ref_src = task.metadata.get("ref_arch_src")
        if not ref_src:
            continue
        for scale in SCALE_FACTORS:
            print(f"  [{i+1}/{len(selected)}] {task.task_id} ({task.kernel_type}) "
                  f"scale={scale}x ...", end=" ", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(profile_at_scale, ref_src, scale, num_trials)
                try:
                    r = fut.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    r = {"error": "timeout", "scale": scale}
                    print("TIMEOUT"); continue
            entry = {"task_id": task.task_id, "kernel_type": task.kernel_type,
                     "level": task.level, **r}
            results.append(entry)
            if r.get("error"):
                print(f"ERR: {str(r['error'])[:60]}")
            else:
                e = r.get("energy_j", 0); rt = r.get("runtime_ms")
                rt_s = f"rt={rt:.2f}ms" if rt else "rt=n/a"
                print(f"E={e:.4f}J  {rt_s}")

        # Save incrementally after each kernel (all scales)
        with open(output_dir / "scaling_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    with open(output_dir / "scaling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nScaling results saved: {len(results)} measurements")
    _print_scaling_summary(results)
    return results


def _print_scaling_summary(results):
    by_type_scale = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error"):
            continue
        by_type_scale[r["kernel_type"]][r["scale"]].append(r)

    print("\n" + "=" * 90)
    print("INPUT SIZE SCALING SUMMARY")
    print("=" * 90)
    header = f"{'Type':<16}"
    for s in SCALE_FACTORS:
        header += f"  E@{s}x(J)   RT@{s}x  "
    print(header)
    print("-" * 90)
    for kt in sorted(by_type_scale.keys()):
        line = f"{kt:<16}"
        for s in SCALE_FACTORS:
            rows = by_type_scale[kt][s]
            if rows:
                avg_e = sum(r["energy_j"] for r in rows) / len(rows)
                rts = [r["runtime_ms"] for r in rows if r.get("runtime_ms")]
                avg_rt = sum(rts) / len(rts) if rts else 0
                line += f"  {avg_e:>7.3f}J {avg_rt:>7.2f}ms"
            else:
                line += f"  {'n/a':>8} {'n/a':>8}"
        print(line)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Input size scaling experiment")
    parser.add_argument("--kernelbench-root", type=Path,
                        default=ROOT / "external" / "KernelBench")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "input_scaling")
    parser.add_argument("--levels", type=str, default="1,2")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-per-type", type=int, default=3)
    args = parser.parse_args()
    levels = [int(x.strip()) for x in args.levels.split(",")]
    run_scaling_experiment(
        kernelbench_root=args.kernelbench_root, output_dir=args.output_dir,
        levels=levels, num_trials=args.num_trials,
        timeout_s=args.timeout, max_kernels_per_type=args.max_per_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
