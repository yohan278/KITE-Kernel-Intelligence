#!/usr/bin/env python3
"""Utility script to validate energy monitor telemetry and run benchmarks."""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

from src.cli._console import error, info, success

if TYPE_CHECKING:
    from src.core.types import TelemetryReading
    from src.telemetry import EnergyMonitorCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test energy monitor telemetry and run benchmarks."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Energy monitor gRPC target (host:port). Defaults to the local launcher.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between printed samples (default: 1.0).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark workloads instead of just monitoring.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["memory", "compute", "gemm", "all"],
        default="all",
        help="Workload to run in benchmark mode (default: all).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration per workload in benchmark mode (default: 5.0).",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Don't launch energy monitor (assume it's already running).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.benchmark:
        _run_benchmarks(args)
    else:
        _run_telemetry_monitor(args)


def _run_telemetry_monitor(args: argparse.Namespace) -> None:
    """Stream and display telemetry readings."""
    from src.telemetry import EnergyMonitorCollector

    collector = EnergyMonitorCollector(target=args.target)

    try:
        if args.no_launch:
            _stream_readings(collector, args.interval)
        else:
            with collector.start():
                _stream_readings(collector, args.interval)
    except RuntimeError as exc:
        error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nStopping monitor")


def _stream_readings(collector: "EnergyMonitorCollector", interval: float) -> None:
    """Stream telemetry readings to console."""
    success(f"Streaming telemetry via collector '{collector.collector_name}'")

    info(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Time",
            "GPU(J)",
            "GPU(W)",
            "CPU(J)",
            "CPU(W)",
            "ANE(J)",
            "ANE(W)",
            "GPU MB",
            "CPU MB",
        )
    )
    info("-" * 98)

    start = time.time()
    last_emit = start - interval

    for reading in collector.stream_readings():
        now = time.time()
        if now - last_emit < max(interval, 0.05):
            continue
        last_emit = now
        info(_format_telemetry_line(now - start, reading))


def _format_telemetry_line(elapsed: float, reading: "TelemetryReading") -> str:
    return "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        _format_elapsed(elapsed),
        _format_metric(reading.energy_joules, width=10, precision=3),
        _format_metric(reading.power_watts, width=10, precision=2),
        _format_metric(reading.cpu_energy_joules, width=10, precision=3),
        _format_metric(reading.cpu_power_watts, width=10, precision=2),
        _format_metric(reading.ane_energy_joules, width=10, precision=3),
        _format_metric(reading.ane_power_watts, width=10, precision=2),
        _format_metric(reading.gpu_memory_usage_mb, width=10, precision=1),
        _format_metric(reading.cpu_memory_usage_mb, width=10, precision=1),
    )


def _run_benchmarks(args: argparse.Namespace) -> None:
    """Run benchmark workloads with energy measurement."""
    from src.benchmarks import BenchmarkRunner
    from src.benchmarks.types import DataType, WorkloadConfig, WorkloadType

    info("Initializing benchmark runner...")

    try:
        runner = BenchmarkRunner()
    except RuntimeError as exc:
        error(f"Failed to initialize: {exc}")
        sys.exit(1)

    success(f"Platform: {runner.suite.platform_name}")
    success(f"Hardware: {runner.hardware_name}")
    info("")

    workloads_to_run = []

    if args.workload in ("memory", "all"):
        workloads_to_run.append(
            (
                "Memory Bandwidth (1GB)",
                WorkloadType.MEMORY_BANDWIDTH,
                WorkloadConfig(
                    workload_type=WorkloadType.MEMORY_BANDWIDTH,
                    duration_seconds=args.duration,
                    params={"array_size_mb": 1024},
                ),
            )
        )

    if args.workload in ("compute", "all"):
        workloads_to_run.append(
            (
                "Compute (AI=16)",
                WorkloadType.COMPUTE_BOUND,
                WorkloadConfig(
                    workload_type=WorkloadType.COMPUTE_BOUND,
                    duration_seconds=args.duration,
                    params={"arithmetic_intensity": 16, "array_size_mb": 100},
                ),
            )
        )

    if args.workload in ("gemm", "all"):
        # Check which dtypes the GEMM workload supports
        gemm_workload = runner.suite.get_gemm_workload()
        gemm_dtypes = gemm_workload.supported_data_types()

        if DataType.FP32 in gemm_dtypes:
            workloads_to_run.append(
                (
                    "GEMM FP32 (4096x4096)",
                    WorkloadType.GEMM,
                    WorkloadConfig(
                        workload_type=WorkloadType.GEMM,
                        duration_seconds=args.duration,
                        data_type=DataType.FP32,
                        params={"matrix_size": 4096},
                    ),
                )
            )
        if DataType.FP16 in gemm_dtypes:
            workloads_to_run.append(
                (
                    "GEMM FP16 (4096x4096)",
                    WorkloadType.GEMM,
                    WorkloadConfig(
                        workload_type=WorkloadType.GEMM,
                        duration_seconds=args.duration,
                        data_type=DataType.FP16,
                        params={"matrix_size": 4096},
                ),
            )
        )

    info("=" * 70)
    info(f"{'Workload':<30} {'Throughput':>15} {'Energy':>12} {'Avg Power':>10}")
    info("=" * 70)

    # Run all workloads within a single monitor session for efficiency
    with runner:
        for name, wtype, config in workloads_to_run:
            info(f"Running {name}...")
            try:
                result = runner.run_workload(wtype, config)
                throughput = f"{result.workload.throughput:.2f} {result.workload.throughput_unit}"
                energy = f"{result.energy.total_energy_joules:.2f} J"
                power = f"{result.energy.avg_total_power_watts:.2f} W"
                success(f"{name:<30} {throughput:>15} {energy:>12} {power:>10}")

                # Show energy efficiency
                if result.energy_per_flop_pj is not None:
                    info(f"  -> {result.energy_per_flop_pj:.4f} pJ/FLOP")
                elif result.energy_per_bit_pj is not None:
                    info(f"  -> {result.energy_per_bit_pj:.4f} pJ/bit")

            except Exception as exc:
                error(f"{name:<30} FAILED: {exc}")

    info("=" * 70)
    success("Benchmark complete!")


def _format_elapsed(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs}s"


def _format_metric(value: float | None, *, width: int, precision: int) -> str:
    if value is None:
        return f"{'-':>{width}}"
    try:
        return f"{value:>{width}.{precision}f}"
    except (ValueError, TypeError):
        return f"{'-':>{width}}"


if __name__ == "__main__":
    main()
