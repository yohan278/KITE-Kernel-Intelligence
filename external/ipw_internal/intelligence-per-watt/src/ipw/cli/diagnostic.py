"""CLI commands for energy characterization benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from ipw.cli._console import success


@click.group(help="Energy characterization benchmarks")
def benchmark() -> None:
    """Benchmark command group."""


@benchmark.command(help="Run energy characterization benchmarks")
@click.option(
    "--platform",
    type=click.Choice(["macos", "nvidia", "rocm"]),
    default=None,
    help="Platform to benchmark (auto-detect if not specified)",
)
@click.option(
    "--duration",
    type=float,
    default=10.0,
    help="Duration per workload in seconds",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for results (JSON)",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Run quick characterization (fewer data points)",
)
@click.option(
    "--memory-only",
    is_flag=True,
    help="Run only memory bandwidth tests",
)
@click.option(
    "--compute-only",
    is_flag=True,
    help="Run only compute tests",
)
@click.option(
    "--gemm-only",
    is_flag=True,
    help="Run only GEMM tests",
)
def characterize(
    platform: Optional[str],
    duration: float,
    output: Optional[str],
    quick: bool,
    memory_only: bool,
    compute_only: bool,
    gemm_only: bool,
) -> None:
    """Run energy characterization benchmarks to extract energy parameters.

    This runs a suite of microbenchmarks at various configurations to
    determine the energy cost of memory access and computation on your
    hardware. Results can be used to model and predict energy consumption
    of arbitrary workloads.

    Example:
        ipw benchmark characterize --duration 5 --output results.json
    """
    from ipw.benchmarks.analysis import extract_parameters, summarize_results
    from ipw.benchmarks.runner import BenchmarkRunner, CharacterizationConfig
    from ipw.benchmarks.types import DataType

    # Configure characterization
    if quick:
        config = CharacterizationConfig(
            duration_per_workload=min(duration, 5.0),
            memory_sizes_mb=[10, 100, 1024],  # 10MB, 100MB, 1GB
            arithmetic_intensities=[1, 8, 64],
            matrix_sizes=[1024, 2048],
            data_types=[DataType.FP32],
        )
    else:
        config = CharacterizationConfig(
            duration_per_workload=duration,
            memory_sizes_mb=[1, 10, 100, 500, 1024, 2048, 4096],  # Up to 4GB
            arithmetic_intensities=[1, 2, 4, 8, 16, 32, 64, 128],
            matrix_sizes=[512, 1024, 2048, 4096],
            data_types=[DataType.FP32, DataType.FP16],
        )

    # Filter workload types if requested
    if memory_only or compute_only or gemm_only:
        if memory_only:
            config.arithmetic_intensities = []
            config.matrix_sizes = []
        if compute_only:
            config.memory_sizes_mb = []
            config.matrix_sizes = []
        if gemm_only:
            config.memory_sizes_mb = []
            config.arithmetic_intensities = []

    # Create runner
    click.echo("Initializing benchmark runner...")
    runner = BenchmarkRunner(platform=platform)
    click.echo(f"Platform: {runner.suite.platform_name}")
    click.echo(f"Hardware: {runner.hardware_name}")
    click.echo()

    # Progress callback
    def progress(step: int, total: int, desc: str) -> None:
        pct = (step / total) * 100
        click.echo(f"[{step:3d}/{total:3d}] ({pct:5.1f}%) {desc}")

    # Run all benchmarks within a single monitor session for efficiency
    with runner:
        # Measure idle power first
        click.echo("Measuring idle power (5 seconds)...")
        idle_cpu, idle_gpu = runner.measure_idle_power(5.0)
        click.echo(f"  CPU idle: {idle_cpu:.2f} W")
        click.echo(f"  GPU idle: {idle_gpu:.2f} W")
        click.echo()

        # Run characterization
        click.echo("Running characterization...")
        results = runner.run_characterization(config, progress_callback=progress)
        click.echo()

    # Extract parameters
    click.echo("Extracting energy parameters...")
    params = extract_parameters(
        results,
        platform=runner.suite.platform,
        hardware_name=runner.hardware_name,
        idle_cpu_watts=idle_cpu,
        idle_gpu_watts=idle_gpu,
    )

    # Display results
    click.echo()
    click.echo("=" * 60)
    click.echo("ENERGY PARAMETERS")
    click.echo("=" * 60)
    click.echo(f"Platform: {params.platform.value}")
    click.echo(f"Hardware: {params.hardware_name}")
    click.echo()

    click.echo("Memory Energy:")
    click.echo(f"  Total:    {params.e_memory_pj_per_bit:.4f} pJ/bit")
    click.echo(f"  Control:  {params.e_memory_control_pj_per_bit:.4f} pJ/bit")
    click.echo(f"  Datapath: {params.e_memory_datapath_pj_per_bit:.4f} pJ/bit")
    click.echo()

    click.echo("Compute Energy (pJ/FLOP):")
    for dtype, value in params.e_compute_pj_per_flop.items():
        control = params.e_compute_control_pj_per_flop.get(dtype, 0.0)
        datapath = params.e_compute_datapath_pj_per_flop.get(dtype, 0.0)
        click.echo(f"  {dtype.value}: total={value:.4f}, control={control:.4f}, datapath={datapath:.4f}")
    click.echo()

    click.echo("GEMM Energy (pJ/FLOP):")
    for dtype, value in params.e_gemm_pj_per_flop.items():
        control = params.e_gemm_control_pj_per_flop.get(dtype, 0.0)
        datapath = params.e_gemm_datapath_pj_per_flop.get(dtype, 0.0)
        click.echo(f"  {dtype.value}: total={value:.4f}, control={control:.4f}, datapath={datapath:.4f}")
    click.echo()

    click.echo("Idle Power:")
    click.echo(f"  CPU: {params.p_idle_cpu_watts:.2f} W")
    click.echo(f"  GPU: {params.p_idle_gpu_watts:.2f} W")
    click.echo("=" * 60)

    # Summary
    summary = summarize_results(results)
    click.echo()
    click.echo(f"Total runs: {summary['total_runs']}")
    click.echo(f"Total duration: {summary['total_duration_seconds']:.1f} seconds")
    click.echo(f"Total energy: {summary['total_energy_joules']:.2f} J")

    # Save results
    if output:
        output_path = Path(output)
        output_data = {
            "parameters": params.to_dict(),
            "summary": summary,
            "results": [
                {
                    "workload_type": r.workload.workload_type.value,
                    "config": {
                        "duration_seconds": r.workload.config.duration_seconds,
                        "use_zeros": r.workload.config.use_zeros,
                        "data_type": r.workload.config.data_type.value,
                        "params": r.workload.config.params,
                    },
                    "throughput": r.workload.throughput,
                    "throughput_unit": r.workload.throughput_unit,
                    "bytes_transferred": r.workload.bytes_transferred,
                    "flops_executed": r.workload.flops_executed,
                    "duration_seconds": r.workload.duration_seconds,
                    "energy": {
                        "cpu_joules": r.energy.cpu_energy_joules,
                        "gpu_joules": r.energy.gpu_energy_joules,
                        "ane_joules": r.energy.ane_energy_joules,
                        "total_joules": r.energy.total_energy_joules,
                        "avg_cpu_watts": r.energy.avg_cpu_power_watts,
                        "avg_gpu_watts": r.energy.avg_gpu_power_watts,
                        "avg_ane_watts": r.energy.avg_ane_power_watts,
                        "sample_count": r.energy.sample_count,
                    },
                }
                for r in results
            ],
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        click.echo()
        success(f"Results saved to {output_path}")


@benchmark.command(help="Run a single workload benchmark")
@click.option(
    "--workload",
    "-w",
    type=click.Choice(["memory", "compute", "gemm"]),
    required=True,
    help="Workload type to run",
)
@click.option(
    "--duration",
    type=float,
    default=10.0,
    help="Duration in seconds",
)
@click.option(
    "--zeros",
    is_flag=True,
    help="Use zero-initialized data (control energy)",
)
@click.option(
    "--array-size",
    type=int,
    default=100,
    help="Array size in MB (memory/compute workloads)",
)
@click.option(
    "--arithmetic-intensity",
    type=int,
    default=16,
    help="Arithmetic intensity (compute workload)",
)
@click.option(
    "--matrix-size",
    type=int,
    default=2048,
    help="Matrix dimension (GEMM workload)",
)
@click.option(
    "--dtype",
    type=click.Choice(["fp32", "fp16", "fp64"]),
    default="fp32",
    help="Data type",
)
@click.option(
    "--platform",
    type=click.Choice(["macos", "nvidia", "rocm"]),
    default=None,
    help="Platform (auto-detect if not specified)",
)
def run(
    workload: str,
    duration: float,
    zeros: bool,
    array_size: int,
    arithmetic_intensity: int,
    matrix_size: int,
    dtype: str,
    platform: Optional[str],
) -> None:
    """Run a single benchmark workload with energy measurement.

    Example:
        ipw benchmark run --workload memory --array-size 1024 --duration 10
        ipw benchmark run --workload gemm --matrix-size 4096 --dtype fp16
    """
    from ipw.benchmarks.runner import BenchmarkRunner
    from ipw.benchmarks.types import DataType, WorkloadConfig, WorkloadType

    # Map strings to enums
    workload_map = {
        "memory": WorkloadType.MEMORY_BANDWIDTH,
        "compute": WorkloadType.COMPUTE_BOUND,
        "gemm": WorkloadType.GEMM,
    }
    dtype_map = {
        "fp64": DataType.FP64,
        "fp32": DataType.FP32,
        "fp16": DataType.FP16,
    }

    workload_type = workload_map[workload]
    data_type = dtype_map[dtype]

    # Build params
    params = {}
    if workload_type == WorkloadType.MEMORY_BANDWIDTH:
        params["array_size_mb"] = array_size
    elif workload_type == WorkloadType.COMPUTE_BOUND:
        params["array_size_mb"] = array_size
        params["arithmetic_intensity"] = arithmetic_intensity
    elif workload_type == WorkloadType.GEMM:
        params["matrix_size"] = matrix_size

    config = WorkloadConfig(
        workload_type=workload_type,
        duration_seconds=duration,
        use_zeros=zeros,
        data_type=data_type,
        params=params,
    )

    # Run
    click.echo("Initializing benchmark runner...")
    runner = BenchmarkRunner(platform=platform)
    click.echo(f"Platform: {runner.suite.platform_name}")
    click.echo(f"Hardware: {runner.hardware_name}")
    click.echo()

    click.echo(f"Running {workload} benchmark...")
    click.echo(f"  Duration: {duration}s")
    click.echo(f"  Data: {'zeros' if zeros else 'random'}")
    click.echo(f"  Type: {dtype}")
    click.echo(f"  Params: {params}")
    click.echo()

    with runner:
        result = runner.run_workload(workload_type, config)

    click.echo("=" * 50)
    click.echo("RESULTS")
    click.echo("=" * 50)
    click.echo(f"Throughput: {result.workload.throughput:.4f} {result.workload.throughput_unit}")
    click.echo(f"Duration: {result.workload.duration_seconds:.2f} s")
    click.echo()
    click.echo("Energy:")
    click.echo(f"  CPU: {result.energy.cpu_energy_joules:.2f} J ({result.energy.avg_cpu_power_watts:.2f} W avg)")
    click.echo(f"  GPU: {result.energy.gpu_energy_joules:.2f} J ({result.energy.avg_gpu_power_watts:.2f} W avg)")
    if result.energy.ane_energy_joules > 0:
        click.echo(f"  ANE: {result.energy.ane_energy_joules:.2f} J ({result.energy.avg_ane_power_watts:.2f} W avg)")
    click.echo(f"  Total: {result.energy.total_energy_joules:.2f} J")
    click.echo(f"  Samples: {result.energy.sample_count}")

    if result.energy_per_bit_pj is not None:
        click.echo(f"  Energy/bit: {result.energy_per_bit_pj:.4f} pJ")
    if result.energy_per_flop_pj is not None:
        click.echo(f"  Energy/FLOP: {result.energy_per_flop_pj:.4f} pJ")
    click.echo("=" * 50)


@benchmark.command(help="List available benchmark platforms")
def platforms() -> None:
    """List available benchmark platforms and their workloads."""
    from ipw.core.registry import BenchmarkRegistry

    # Import to register platforms
    import ipw.benchmarks  # noqa: F401

    click.echo("Available Benchmark Platforms:")
    click.echo("-" * 40)

    for name, suite_cls in BenchmarkRegistry.items():
        available = suite_cls.is_available()
        status = "available" if available else "not available"
        click.echo(f"\n{name}: {status}")

        if available:
            suite = suite_cls()
            click.echo(f"  Platform: {suite.platform_name}")
            click.echo(f"  Hardware: {suite.detect_hardware()}")
            click.echo("  Workloads:")
            for workload in suite.get_workloads():
                dtypes = ", ".join(dt.value for dt in workload.supported_data_types())
                click.echo(f"    - {workload.workload_name} ({dtypes})")


__all__ = ["benchmark"]
