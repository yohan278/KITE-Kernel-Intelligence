"""Benchmark runner with energy telemetry integration.

This module provides a runner that executes benchmark workloads while
collecting energy measurements via the energy monitor. It supports
running individual workloads or full characterization sweeps.

The runner is designed as a context manager that maintains a single
long-lived energy monitor session for efficiency:

    with BenchmarkRunner() as runner:
        result1 = runner.run_workload(...)
        result2 = runner.run_workload(...)
        results = runner.run_characterization(...)

This avoids the overhead of starting/stopping the energy monitor
(which spawns powermetrics on macOS) for each individual workload.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from ipw.benchmarks.base import BenchmarkSuite
from ipw.benchmarks.types import (
    BenchmarkResult,
    DataType,
    EnergyMeasurement,
    WorkloadConfig,
    WorkloadResult,
    WorkloadType,
)
from ipw.core.registry import BenchmarkRegistry
from ipw.core.types import TelemetryReading
from ipw.telemetry import EnergyMonitorCollector


@dataclass(slots=True)
class CharacterizationConfig:
    """Configuration for a full characterization sweep.

    Attributes:
        duration_per_workload: Duration in seconds for each workload run.
        memory_sizes_mb: Array sizes for memory bandwidth tests (in MB).
        arithmetic_intensities: AI values for compute sweep.
        matrix_sizes: Matrix dimensions for GEMM tests.
        data_types: Data types to test for compute/GEMM (filtered by workload support).
        include_int32: Whether to include INT32 compute tests.
        int32_intensities: AI values for INT32 compute sweep.
        include_zeros: Whether to run zero-data (control) tests.
        include_random: Whether to run random-data (datapath) tests.
    """

    duration_per_workload: float = 10.0
    memory_sizes_mb: List[int] = field(
        default_factory=lambda: [1, 10, 100, 500, 1024, 2048, 4096]
    )  # Up to 4GB
    arithmetic_intensities: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )
    matrix_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    data_types: List[DataType] = field(
        default_factory=lambda: [DataType.FP32]
    )  # FP16 requires MPS/CUDA
    include_int32: bool = True  # Include INT32 compute tests
    int32_intensities: List[int] = field(
        default_factory=lambda: [1, 4, 16, 64, 128]  # Fewer points for INT32
    )
    include_zeros: bool = True
    include_random: bool = True
    inference_config: Optional[InferenceCharacterizationConfig] = None


@dataclass(slots=True)
class InferenceCharacterizationConfig:
    """Configuration for inference-level energy characterization sweeps.

    These benchmarks isolate specific inference components to extract
    parameters for the energy scaling law:
        E_call = k_HW * [E_prefill + E_decode + λ_comm]

    Attributes:
        gemm_batch_sizes: Batch sizes for inference GEMM.
        gemm_seq_lens: Sequence lengths for inference GEMM.
        gemm_hidden_dim: Model hidden dimension.
        gemm_ff_dim: Feed-forward dimension.
        attn_batch_sizes: Batch sizes for attention benchmarks.
        attn_seq_lens: Sequence lengths for attention benchmarks.
        attn_num_heads: Number of attention heads.
        attn_head_dim: Dimension per attention head.
        kv_cache_entries: KV cache sizes (number of token positions).
        nccl_message_sizes_mb: Message sizes for collective communication.
        decode_batch_sizes: Batch sizes for batched decode scaling.
        duration_per_workload: Duration per benchmark run in seconds.
    """

    gemm_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 16, 64])
    gemm_seq_lens: List[int] = field(
        default_factory=lambda: [128, 512, 2048, 8192]
    )
    gemm_hidden_dim: int = 4096
    gemm_ff_dim: int = 11008
    attn_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 16])
    attn_seq_lens: List[int] = field(
        default_factory=lambda: [128, 512, 1024, 2048, 4096, 8192]
    )
    attn_num_heads: int = 32
    attn_head_dim: int = 128
    kv_cache_entries: List[int] = field(
        default_factory=lambda: [128, 512, 2048, 8192, 32768]
    )
    nccl_message_sizes_mb: List[int] = field(
        default_factory=lambda: [1, 10, 100, 500]
    )
    decode_batch_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    duration_per_workload: float = 10.0
    use_cuda_graphs: bool = False


class BenchmarkRunner:
    """Platform-agnostic benchmark runner with energy telemetry.

    Integrates with the existing EnergyMonitorCollector to measure
    energy consumption during workload execution.

    The runner is a context manager that maintains a single long-lived
    energy monitor session. This is much more efficient than starting/
    stopping the monitor for each workload.

    Example:
        with BenchmarkRunner() as runner:
            result = runner.run_workload(
                WorkloadType.MEMORY_BANDWIDTH,
                WorkloadConfig(
                    workload_type=WorkloadType.MEMORY_BANDWIDTH,
                    duration_seconds=10.0,
                    params={"array_size_mb": 1024},
                ),
            )
            print(f"Bandwidth: {result.workload.throughput} GB/s")
            print(f"Energy: {result.energy.total_energy_joules} J")

    For backwards compatibility, the runner can also be used without
    the context manager, but this will start/stop the monitor for each
    operation (less efficient).
    """

    def __init__(
        self,
        platform: Optional[str] = None,
        collector: Optional[EnergyMonitorCollector] = None,
    ):
        """Initialize the benchmark runner.

        Args:
            platform: Platform to use ("macos", "nvidia", "rocm").
                     If None, auto-detects the platform.
            collector: Energy collector to use. If None, creates a default one.
        """
        self._suite = self._get_suite(platform)
        self._collector = collector or EnergyMonitorCollector()
        self._hardware_name = self._suite.detect_hardware()
        self._monitor_context: Optional[ExitStack] = None
        self._is_running = False

    def _get_suite(self, platform: Optional[str]) -> BenchmarkSuite:
        """Get the appropriate benchmark suite."""
        if platform:
            suite_cls = BenchmarkRegistry.get(platform)
            if not suite_cls.is_available():
                raise RuntimeError(f"Platform '{platform}' is not available")
            return suite_cls()

        # Auto-detect platform
        for name, suite_cls in BenchmarkRegistry.items():
            if suite_cls.is_available():
                return suite_cls()

        raise RuntimeError("No benchmark platform available")

    def __enter__(self) -> "BenchmarkRunner":
        """Start the energy monitor for the benchmark session."""
        if self._is_running:
            return self
        self._monitor_context = ExitStack()
        self._monitor_context.enter_context(self._collector.start())
        self._is_running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the energy monitor."""
        if self._monitor_context is not None:
            self._monitor_context.close()
            self._monitor_context = None
        self._is_running = False

    @property
    def suite(self) -> BenchmarkSuite:
        """The benchmark suite being used."""
        return self._suite

    @property
    def hardware_name(self) -> str:
        """Detected hardware name."""
        return self._hardware_name

    @property
    def is_running(self) -> bool:
        """Whether the energy monitor is currently running."""
        return self._is_running

    def _ensure_monitor(self):
        """Context manager that ensures monitor is running.

        If the runner is already in a context (monitor running), this is a no-op.
        Otherwise, it starts and stops the monitor for this operation only.
        """
        if self._is_running:
            # Monitor already running, return a no-op context
            from contextlib import nullcontext
            return nullcontext()
        else:
            # Not in a session, start/stop for this operation (backwards compat)
            return self._collector.start()

    def run_workload(
        self,
        workload_type: WorkloadType,
        config: WorkloadConfig,
    ) -> BenchmarkResult:
        """Run a single workload with energy measurement.

        Args:
            workload_type: Type of workload to run.
            config: Workload configuration.

        Returns:
            BenchmarkResult containing workload results and energy measurement.
        """
        # Get the appropriate workload
        if workload_type == WorkloadType.MEMORY_BANDWIDTH:
            workload = self._suite.get_memory_workload()
        elif workload_type == WorkloadType.COMPUTE_BOUND:
            workload = self._suite.get_compute_workload()
        elif workload_type == WorkloadType.GEMM:
            workload = self._suite.get_gemm_workload()
        elif workload_type == WorkloadType.CACHE_L1:
            # Check if platform supports cache-level workloads
            if hasattr(self._suite, 'get_l1_cache_workload'):
                workload = self._suite.get_l1_cache_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support L1 cache workloads")
        elif workload_type == WorkloadType.CACHE_L2:
            if hasattr(self._suite, 'get_l2_cache_workload'):
                workload = self._suite.get_l2_cache_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support L2 cache workloads")
        elif workload_type == WorkloadType.HBM_BANDWIDTH:
            if hasattr(self._suite, 'get_hbm_workload'):
                workload = self._suite.get_hbm_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support HBM workloads")
        elif workload_type == WorkloadType.INFERENCE_GEMM:
            if hasattr(self._suite, 'get_inference_gemm_workload'):
                workload = self._suite.get_inference_gemm_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support inference GEMM workloads")
        elif workload_type == WorkloadType.ATTENTION:
            if hasattr(self._suite, 'get_attention_workload'):
                workload = self._suite.get_attention_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support attention workloads")
        elif workload_type == WorkloadType.KV_CACHE_IO:
            if hasattr(self._suite, 'get_kv_cache_workload'):
                workload = self._suite.get_kv_cache_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support KV cache workloads")
        elif workload_type == WorkloadType.NCCL_COLLECTIVE:
            if hasattr(self._suite, 'get_nccl_collective_workload'):
                workload = self._suite.get_nccl_collective_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support NCCL collective workloads")
        elif workload_type == WorkloadType.BATCHED_DECODE:
            if hasattr(self._suite, 'get_batched_decode_workload'):
                workload = self._suite.get_batched_decode_workload()
            else:
                raise ValueError(f"Platform {self._suite.platform_name} does not support batched decode workloads")
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")

        # Warmup
        warmup_config = WorkloadConfig(
            workload_type=config.workload_type,
            duration_seconds=min(2.0, config.duration_seconds / 2),
            use_zeros=config.use_zeros,
            data_type=config.data_type,
            params=config.params,
        )
        workload.warmup(warmup_config)

        # Collect energy during workload execution
        # Use _ensure_monitor to avoid starting/stopping if already in a session
        with self._ensure_monitor():
            readings: List[TelemetryReading] = []
            start_time = time.time()

            # Start collecting readings in a thread
            stop_collection = threading.Event()

            def collect_readings():
                try:
                    for reading in self._collector.stream_readings():
                        if stop_collection.is_set():
                            break
                        readings.append(reading)
                except Exception:
                    pass  # Stream closed

            collector_thread = threading.Thread(target=collect_readings, daemon=True)
            collector_thread.start()

            # Give collector time to start
            time.sleep(0.1)

            # Run the workload
            workload_result = workload.run(config)

            # Stop collection
            stop_collection.set()
            # Wait for collector thread to finish - use longer timeout and
            # copy readings while holding implicit GIL to avoid race condition
            collector_thread.join(timeout=5.0)
            if collector_thread.is_alive():
                # Thread still running after timeout - make a copy of readings
                # to avoid race condition with ongoing appends
                readings = list(readings)

            end_time = time.time()

        # Process readings into energy measurement
        energy = self._process_readings(readings, end_time - start_time)

        return BenchmarkResult(workload=workload_result, energy=energy)

    def _process_readings(
        self, readings: List[TelemetryReading], duration: float
    ) -> EnergyMeasurement:
        """Process telemetry readings into an energy measurement."""
        if not readings:
            return EnergyMeasurement(
                cpu_energy_joules=0.0,
                gpu_energy_joules=0.0,
                ane_energy_joules=0.0,
                avg_cpu_power_watts=0.0,
                avg_gpu_power_watts=0.0,
                avg_ane_power_watts=0.0,
                max_cpu_power_watts=0.0,
                max_gpu_power_watts=0.0,
                max_ane_power_watts=0.0,
                duration_seconds=duration,
                sample_count=0,
            )

        # Extract power readings for max calculation
        cpu_powers = [r.cpu_power_watts for r in readings if r.cpu_power_watts]
        gpu_powers = [r.power_watts for r in readings if r.power_watts]
        ane_powers = [r.ane_power_watts for r in readings if r.ane_power_watts]

        max_cpu = max(cpu_powers) if cpu_powers else 0.0
        max_gpu = max(gpu_powers) if gpu_powers else 0.0
        max_ane = max(ane_powers) if ane_powers else 0.0

        first_reading = readings[0]
        last_reading = readings[-1]

        cpu_energy = 0.0
        gpu_energy = 0.0
        ane_energy = 0.0

        # Try to use cumulative energy if available (more accurate)
        if (
            first_reading.cpu_energy_joules is not None
            and last_reading.cpu_energy_joules is not None
        ):
            cpu_energy = last_reading.cpu_energy_joules - first_reading.cpu_energy_joules

        if (
            first_reading.energy_joules is not None
            and last_reading.energy_joules is not None
        ):
            gpu_energy = last_reading.energy_joules - first_reading.energy_joules

        if (
            first_reading.ane_energy_joules is not None
            and last_reading.ane_energy_joules is not None
        ):
            ane_energy = last_reading.ane_energy_joules - first_reading.ane_energy_joules

        # Calculate avg power from energy/duration (more accurate than avg of instantaneous)
        # This ensures consistency between energy and power values
        avg_cpu = cpu_energy / duration if duration > 0 else 0.0
        avg_gpu = gpu_energy / duration if duration > 0 else 0.0
        avg_ane = ane_energy / duration if duration > 0 else 0.0

        # If we didn't get cumulative energy, fall back to instantaneous power averages
        if cpu_energy == 0.0 and cpu_powers:
            avg_cpu = sum(cpu_powers) / len(cpu_powers)
            cpu_energy = avg_cpu * duration

        if gpu_energy == 0.0 and gpu_powers:
            avg_gpu = sum(gpu_powers) / len(gpu_powers)
            gpu_energy = avg_gpu * duration

        if ane_energy == 0.0 and ane_powers:
            avg_ane = sum(ane_powers) / len(ane_powers)
            ane_energy = avg_ane * duration

        return EnergyMeasurement(
            cpu_energy_joules=max(0.0, cpu_energy),
            gpu_energy_joules=max(0.0, gpu_energy),
            ane_energy_joules=max(0.0, ane_energy),
            avg_cpu_power_watts=avg_cpu,
            avg_gpu_power_watts=avg_gpu,
            avg_ane_power_watts=avg_ane,
            max_cpu_power_watts=max_cpu,
            max_gpu_power_watts=max_gpu,
            max_ane_power_watts=max_ane,
            duration_seconds=duration,
            sample_count=len(readings),
        )

    def measure_idle_power(self, duration: float = 5.0) -> Tuple[float, float]:
        """Measure idle power consumption.

        Args:
            duration: How long to measure idle power.

        Returns:
            Tuple of (cpu_idle_watts, gpu_idle_watts).
        """
        readings: List[TelemetryReading] = []

        # Use _ensure_monitor to avoid starting/stopping if already in a session
        with self._ensure_monitor():
            stop_collection = threading.Event()

            def collect_readings():
                try:
                    for reading in self._collector.stream_readings():
                        if stop_collection.is_set():
                            break
                        readings.append(reading)
                except Exception:
                    pass

            collector_thread = threading.Thread(target=collect_readings, daemon=True)
            collector_thread.start()

            # Just wait (idle)
            time.sleep(duration)

            stop_collection.set()
            collector_thread.join(timeout=5.0)
            if collector_thread.is_alive():
                # Thread still running - make a copy to avoid race condition
                readings = list(readings)

        cpu_powers = [r.cpu_power_watts for r in readings if r.cpu_power_watts]
        gpu_powers = [r.power_watts for r in readings if r.power_watts]

        avg_cpu = sum(cpu_powers) / len(cpu_powers) if cpu_powers else 0.0
        avg_gpu = sum(gpu_powers) / len(gpu_powers) if gpu_powers else 0.0

        return (avg_cpu, avg_gpu)

    def run_characterization(
        self,
        config: Optional[CharacterizationConfig] = None,
        progress_callback=None,
    ) -> List[BenchmarkResult]:
        """Run a full characterization sweep.

        This runs a comprehensive set of benchmarks to extract energy parameters:
        - Memory bandwidth at various array sizes
        - Compute at various arithmetic intensities
        - GEMM at various matrix sizes
        - Each with both zero (control) and random (datapath) data

        Args:
            config: Characterization configuration. Uses defaults if None.
            progress_callback: Optional callback(step, total, description) for progress.

        Returns:
            List of all BenchmarkResult from the characterization.
        """
        if config is None:
            config = CharacterizationConfig()

        results: List[BenchmarkResult] = []

        # Count total steps for progress
        use_zeros_values = []
        if config.include_zeros:
            use_zeros_values.append(True)
        if config.include_random:
            use_zeros_values.append(False)

        total_memory = len(config.memory_sizes_mb) * len(use_zeros_values)
        total_compute = (
            len(config.arithmetic_intensities)
            * len(use_zeros_values)
            * len([dt for dt in config.data_types if dt in [DataType.FP32, DataType.FP64]])
        )

        # Check if INT32 workload is available
        has_int32 = False
        try:
            self._suite.get_integer_compute_workload()
            has_int32 = True
        except NotImplementedError:
            pass
        total_int32 = (
            len(config.int32_intensities) * len(use_zeros_values)
            if config.include_int32 and has_int32
            else 0
        )

        # Get GEMM supported dtypes to correctly count progress
        gemm_workload = self._suite.get_gemm_workload()
        gemm_supported_dtypes = gemm_workload.supported_data_types()
        gemm_dtypes = [dt for dt in config.data_types if dt in gemm_supported_dtypes]
        total_gemm = len(config.matrix_sizes) * len(use_zeros_values) * len(gemm_dtypes)
        total_steps = total_memory + total_compute + total_int32 + total_gemm
        current_step = 0

        def report_progress(desc: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, desc)

        # Note: Idle power should be measured separately before calling this method
        # (e.g., via runner.measure_idle_power()). The analysis phase uses those values.

        # 1. Memory bandwidth sweep
        for size_mb in config.memory_sizes_mb:
            for use_zeros in use_zeros_values:
                desc = f"Memory {size_mb}MB {'zeros' if use_zeros else 'random'}"
                report_progress(desc)

                workload_config = WorkloadConfig(
                    workload_type=WorkloadType.MEMORY_BANDWIDTH,
                    duration_seconds=config.duration_per_workload,
                    use_zeros=use_zeros,
                    data_type=DataType.FP32,
                    params={"array_size_mb": size_mb},
                )
                try:
                    result = self.run_workload(
                        WorkloadType.MEMORY_BANDWIDTH, workload_config
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Warning: {desc} failed: {e}")

        # 2. Compute sweep (arithmetic intensity)
        compute_dtypes = [
            dt for dt in config.data_types if dt in [DataType.FP32, DataType.FP64]
        ]
        for ai in config.arithmetic_intensities:
            for use_zeros in use_zeros_values:
                for dtype in compute_dtypes:
                    desc = f"Compute AI={ai} {dtype.value} {'zeros' if use_zeros else 'random'}"
                    report_progress(desc)

                    workload_config = WorkloadConfig(
                        workload_type=WorkloadType.COMPUTE_BOUND,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=use_zeros,
                        data_type=dtype,
                        params={"arithmetic_intensity": ai, "array_size_mb": 100},
                    )
                    try:
                        result = self.run_workload(
                            WorkloadType.COMPUTE_BOUND, workload_config
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"Warning: {desc} failed: {e}")

        # 3. INT32 compute sweep
        if config.include_int32 and has_int32:
            for ai in config.int32_intensities:
                for use_zeros in use_zeros_values:
                    desc = f"INT32 AI={ai} {'zeros' if use_zeros else 'random'}"
                    report_progress(desc)

                    workload_config = WorkloadConfig(
                        workload_type=WorkloadType.COMPUTE_BOUND,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=use_zeros,
                        data_type=DataType.INT32,
                        params={"arithmetic_intensity": ai, "array_size_mb": 100},
                    )
                    try:
                        # Use the integer compute workload
                        int_workload = self._suite.get_integer_compute_workload()
                        # Run with warmup
                        warmup_config = WorkloadConfig(
                            workload_type=workload_config.workload_type,
                            duration_seconds=min(2.0, config.duration_per_workload / 2),
                            use_zeros=use_zeros,
                            data_type=DataType.INT32,
                            params=workload_config.params,
                        )
                        int_workload.warmup(warmup_config)

                        # Collect energy during workload execution
                        with self._ensure_monitor():
                            import threading
                            readings: List[TelemetryReading] = []
                            start_time = time.time()
                            stop_collection = threading.Event()

                            def collect_readings():
                                try:
                                    for reading in self._collector.stream_readings():
                                        if stop_collection.is_set():
                                            break
                                        readings.append(reading)
                                except Exception:
                                    pass

                            collector_thread = threading.Thread(target=collect_readings, daemon=True)
                            collector_thread.start()
                            time.sleep(0.1)

                            workload_result = int_workload.run(workload_config)

                            stop_collection.set()
                            collector_thread.join(timeout=5.0)
                            if collector_thread.is_alive():
                                readings = list(readings)
                            end_time = time.time()

                        energy = self._process_readings(readings, end_time - start_time)
                        result = BenchmarkResult(workload=workload_result, energy=energy)
                        results.append(result)
                    except Exception as e:
                        print(f"Warning: {desc} failed: {e}")

        # 4. GEMM sweep (gemm_workload and gemm_dtypes already computed above)
        for matrix_size in config.matrix_sizes:
            for use_zeros in use_zeros_values:
                for dtype in gemm_dtypes:

                    desc = f"GEMM {matrix_size}x{matrix_size} {dtype.value} {'zeros' if use_zeros else 'random'}"
                    report_progress(desc)

                    workload_config = WorkloadConfig(
                        workload_type=WorkloadType.GEMM,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=use_zeros,
                        data_type=dtype,
                        params={"matrix_size": matrix_size},
                    )
                    try:
                        result = self.run_workload(WorkloadType.GEMM, workload_config)
                        results.append(result)
                    except Exception as e:
                        print(f"Warning: {desc} failed: {e}")

        # 5. Cache-level characterization (NVIDIA/AMD only)
        # L1 cache workloads
        if hasattr(self._suite, 'get_l1_cache_workload'):
            for data_type in config.data_types:
                for use_zeros in use_zeros_values:
                    desc = f"L1 cache {data_type.value} {'zeros' if use_zeros else 'random'}"
                    workload_config = WorkloadConfig(
                        workload_type=WorkloadType.CACHE_L1,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=use_zeros,
                        data_type=data_type,
                        params={},
                    )
                    try:
                        result = self.run_workload(WorkloadType.CACHE_L1, workload_config)
                        results.append(result)
                        if progress_callback:
                            progress_callback(current_step, total_steps, desc)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(current_step, total_steps, f"L1 cache failed: {e}")

        # L2 cache workloads
        if hasattr(self._suite, 'get_l2_cache_workload'):
            for data_type in config.data_types:
                for use_zeros in use_zeros_values:
                    desc = f"L2 cache {data_type.value} {'zeros' if use_zeros else 'random'}"
                    workload_config = WorkloadConfig(
                        workload_type=WorkloadType.CACHE_L2,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=use_zeros,
                        data_type=data_type,
                        params={},
                    )
                    try:
                        result = self.run_workload(WorkloadType.CACHE_L2, workload_config)
                        results.append(result)
                        if progress_callback:
                            progress_callback(current_step, total_steps, desc)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(current_step, total_steps, f"L2 cache failed: {e}")

        # HBM bandwidth workloads
        if hasattr(self._suite, 'get_hbm_workload'):
            for data_type in config.data_types:
                for use_zeros in use_zeros_values:
                    desc = f"HBM bandwidth {data_type.value} {'zeros' if use_zeros else 'random'}"
                    workload_config = WorkloadConfig(
                        workload_type=WorkloadType.HBM_BANDWIDTH,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=use_zeros,
                        data_type=data_type,
                        params={},
                    )
                    try:
                        result = self.run_workload(WorkloadType.HBM_BANDWIDTH, workload_config)
                        results.append(result)
                        if progress_callback:
                            progress_callback(current_step, total_steps, desc)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(current_step, total_steps, f"HBM bandwidth failed: {e}")

        # 6. Inference-level characterization (if configured)
        if config.inference_config is not None:
            inference_results = self.run_inference_characterization(
                config.inference_config, progress_callback=progress_callback,
            )
            results.extend(inference_results)

        return results

    def run_inference_characterization(
        self,
        config: InferenceCharacterizationConfig,
        progress_callback=None,
    ) -> List[BenchmarkResult]:
        """Run inference-level characterization sweeps.

        Measures energy of inference components at various configurations
        to extract parameters for E_call scaling law.

        Args:
            config: Inference characterization configuration.
            progress_callback: Optional callback(step, total, description).

        Returns:
            List of BenchmarkResult from inference characterization.
        """
        results: List[BenchmarkResult] = []

        # Helper to attempt a workload run
        def _try_run(wtype: WorkloadType, wconfig: WorkloadConfig, desc: str):
            try:
                result = self.run_workload(wtype, wconfig)
                results.append(result)
            except (ValueError, NotImplementedError, RuntimeError) as e:
                if progress_callback:
                    progress_callback(0, 0, f"{desc} skipped: {e}")

        # 1. Inference GEMM sweep (prefill + decode modes)
        try:
            self._suite.get_inference_gemm_workload()
            has_inference_gemm = True
        except NotImplementedError:
            has_inference_gemm = False

        if has_inference_gemm:
            for mode in ["prefill", "decode"]:
                for bs in config.gemm_batch_sizes:
                    for seq_len in config.gemm_seq_lens:
                        desc = f"InfGEMM {mode} B={bs} S={seq_len}"
                        if progress_callback:
                            progress_callback(0, 0, desc)
                        wconfig = WorkloadConfig(
                            workload_type=WorkloadType.INFERENCE_GEMM,
                            duration_seconds=config.duration_per_workload,
                            use_zeros=False,
                            use_cuda_graphs=config.use_cuda_graphs,
                            data_type=DataType.FP16,
                            params={
                                "batch_size": bs,
                                "seq_len": seq_len,
                                "hidden_dim": config.gemm_hidden_dim,
                                "ff_dim": config.gemm_ff_dim,
                                "mode": mode,
                            },
                        )
                        _try_run(WorkloadType.INFERENCE_GEMM, wconfig, desc)

        # 2. Attention sweep
        try:
            self._suite.get_attention_workload()
            has_attention = True
        except NotImplementedError:
            has_attention = False

        if has_attention:
            for bs in config.attn_batch_sizes:
                for seq_len in config.attn_seq_lens:
                    desc = f"Attention B={bs} S={seq_len}"
                    if progress_callback:
                        progress_callback(0, 0, desc)
                    wconfig = WorkloadConfig(
                        workload_type=WorkloadType.ATTENTION,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=False,
                        use_cuda_graphs=config.use_cuda_graphs,
                        data_type=DataType.FP16,
                        params={
                            "batch_size": bs,
                            "seq_len": seq_len,
                            "num_heads": config.attn_num_heads,
                            "head_dim": config.attn_head_dim,
                        },
                    )
                    _try_run(WorkloadType.ATTENTION, wconfig, desc)

        # 3. KV Cache sweep
        try:
            self._suite.get_kv_cache_workload()
            has_kv_cache = True
        except NotImplementedError:
            has_kv_cache = False

        if has_kv_cache:
            for mode in ["read", "write"]:
                for entries in config.kv_cache_entries:
                    desc = f"KVCache {mode} entries={entries}"
                    if progress_callback:
                        progress_callback(0, 0, desc)
                    wconfig = WorkloadConfig(
                        workload_type=WorkloadType.KV_CACHE_IO,
                        duration_seconds=config.duration_per_workload,
                        use_zeros=False,
                        use_cuda_graphs=config.use_cuda_graphs,
                        data_type=DataType.FP16,
                        params={
                            "cache_entries": entries,
                            "num_heads": config.attn_num_heads,
                            "head_dim": config.attn_head_dim,
                            "batch_size": 1,
                            "mode": mode,
                        },
                    )
                    _try_run(WorkloadType.KV_CACHE_IO, wconfig, desc)

        # 4. NCCL collective sweep
        try:
            self._suite.get_nccl_collective_workload()
            has_nccl = True
        except NotImplementedError:
            has_nccl = False

        if has_nccl:
            for msg_size in config.nccl_message_sizes_mb:
                desc = f"NCCL all_reduce {msg_size}MB"
                if progress_callback:
                    progress_callback(0, 0, desc)
                wconfig = WorkloadConfig(
                    workload_type=WorkloadType.NCCL_COLLECTIVE,
                    duration_seconds=config.duration_per_workload,
                    use_zeros=False,
                    use_cuda_graphs=config.use_cuda_graphs,
                    data_type=DataType.FP32,
                    params={
                        "message_size_mb": msg_size,
                        "collective_type": "all_reduce",
                    },
                )
                _try_run(WorkloadType.NCCL_COLLECTIVE, wconfig, desc)

        # 5. Batched decode sweep
        try:
            self._suite.get_batched_decode_workload()
            has_batched_decode = True
        except NotImplementedError:
            has_batched_decode = False

        if has_batched_decode:
            for bs in config.decode_batch_sizes:
                desc = f"BatchDecode B={bs}"
                if progress_callback:
                    progress_callback(0, 0, desc)
                wconfig = WorkloadConfig(
                    workload_type=WorkloadType.BATCHED_DECODE,
                    duration_seconds=config.duration_per_workload,
                    use_zeros=False,
                    use_cuda_graphs=config.use_cuda_graphs,
                    data_type=DataType.FP16,
                    params={
                        "batch_size": bs,
                        "hidden_dim": config.gemm_hidden_dim,
                        "ff_dim": config.gemm_ff_dim,
                        "num_layers": 1,
                    },
                )
                _try_run(WorkloadType.BATCHED_DECODE, wconfig, desc)

        return results


__all__ = [
    "BenchmarkRunner",
    "CharacterizationConfig",
    "InferenceCharacterizationConfig",
]
