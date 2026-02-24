"""Profiler runner orchestration."""

from __future__ import annotations

import json
import math
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from datasets import Dataset
from tqdm.auto import tqdm

from ipw.clients.base import InferenceClient
from ipw.core.registry import ClientRegistry, DatasetRegistry
from ipw.core.types import (
    DatasetRecord,
    GpuInfo,
    ProfilerConfig,
    Response,
    SystemInfo,
    TelemetryReading,
)
from ipw.telemetry import EnergyMonitorCollector
from .hardware import derive_hardware_label
from .telemetry_session import TelemetrySample, TelemetrySession
from .types import (
    ComputeMetrics,
    EnergyMetrics,
    LatencyMetrics,
    MemoryMetrics,
    MetricStats,
    HardwareUtilization,
    HardwareUtilizationDerived,
    HardwareUtilizationGpu,
    ModelMetrics,
    PhaseMetrics,
    PowerComponentMetrics,
    PowerMetrics,
    ProfilingRecord,
    TokenMetrics,
)


class ProfilerRunner:
    """Coordinate dataset iteration, inference calls, telemetry capture, and persistence."""

    _FLUSH_INTERVAL = 100
    # Keep enough telemetry history to cover long first-token latencies and
    # long-running requests so phased attribution does not lose request-start
    # boundaries before completion.
    _TELEMETRY_BUFFER_SECONDS = 900.0
    _TELEMETRY_MAX_SAMPLES = 200_000

    # The runner is intentionally a slim orchestrator, but it still handles a
    # fair amount of coordination work:
    #
    # 1. Resolve dataset / client implementations from the registries so that we
    #    only depend on the registry surface, not the old resolution helpers.
    # 2. Spin up the `TelemetrySession`, which hides the threaded sampling loop
    #    that continuously pulls energy/power/memory readings into a rolling
    #    buffer while the run executes.
    # 3. For each dataset record, send the request to the client, collect the
    #    telemetry samples that overlap the query window, and transform the raw
    #    response + telemetry into the strongly typed `ProfilingRecord` payload
    #    defined in `ipw.execution.types`.
    # 4. Accumulate all records in-memory and write a HuggingFace dataset to the
    #    configured output directory once the run completes, along with a
    #    `summary.json` containing run metadata and aggregate energy totals.
    #
    # The actual measurements and conversions stay localized to helper methods
    # (`_compute_energy_metrics`, `_stat_summary`, etc.) so that the control flow
    # remains readable. Any future refactor (e.g., streaming writes or different
    # telemetry aggregation) should only need to touch the helpers and the final
    # persistence step.

    def __init__(self, config: ProfilerConfig) -> None:
        self._config = config
        self._records: Dict[int, ProfilingRecord] = {}
        self._output_path: Optional[Path] = None
        self._hardware_label: Optional[str] = None
        self._system_info: Optional[SystemInfo] = None
        self._gpu_info: Optional[GpuInfo] = None
        self._peak_compute_tflops: Optional[float] = None
        self._peak_memory_bandwidth_gbps: Optional[float] = None
        self._hardware_benchmark_info: Optional[dict[str, Any]] = None
        # For time-slice based energy attribution
        self._all_samples: list[TelemetrySample] = []
        self._request_timings: Dict[int, Tuple[float, float]] = {}

    def run(self) -> None:
        dataset = self._resolve_dataset(
            self._config.dataset_id, self._config.dataset_params
        )
        client = self._resolve_client(
            self._config.client_id,
            self._config.client_base_url,
            self._config.client_params,
        )

        self._benchmark_hardware_if_needed()

        collector = EnergyMonitorCollector()

        self._ensure_client_ready(client)

        try:
            with TelemetrySession(
                collector,
                buffer_seconds=self._TELEMETRY_BUFFER_SECONDS,
                max_samples=self._TELEMETRY_MAX_SAMPLES,
            ) as telemetry:
                self._process_records(dataset, client, telemetry)
                # Recompute energy metrics using time-slice attribution
                self._recompute_metrics()
        finally:
            close_client = getattr(client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception:
                    pass

        if not self._records:
            return

        self._persist_records(dataset)

    def _process_records(
        self,
        dataset,
        client,
        telemetry: TelemetrySession,
    ) -> None:
        """Process all records and collect telemetry for time-slice energy attribution."""
        total_queries = self._config.max_queries or dataset.size()
        max_concurrency = getattr(self._config, "max_concurrency", 1) or 1

        records_list = list(dataset)[:total_queries]
        prompt_iter = self._prompt_iterator_from_list(records_list)
        payload: MutableMapping[str, object] = dict(self._config.additional_parameters)

        with tqdm(total=total_queries, desc="Profiling", unit="query") as progress:
            for index, response in client.run_concurrent(
                self._config.model,
                prompt_iter,
                max_concurrency,
                **payload,
            ):
                # Collect all telemetry samples seen so far
                current_readings = telemetry.readings()
                last_ts = self._all_samples[-1].timestamp if self._all_samples else -1.0
                for s in current_readings:
                    if s.timestamp > last_ts:
                        self._all_samples.append(s)

                record = records_list[index]
                start_time = response.request_start_time
                end_time = response.request_end_time
                if end_time < start_time:
                    end_time = start_time

                # Store request timing for energy attribution
                self._request_timings[index] = (start_time, end_time)

                # Get samples for this request's window (for non-energy metrics)
                samples = list(telemetry.window(start_time, end_time))

                # Build record with placeholder energy (will be recomputed later)
                built = self._build_record(
                    index,
                    record,
                    response,
                    samples,
                    start_time,
                    end_time,
                    phase_samples=current_readings,
                )
                if built is not None:
                    self._records[index] = built

                progress.update(1)

    def _recompute_metrics(self) -> None:
        """Recompute energy metrics using time-slice based attribution.

        For each telemetry sample interval, divide energy by the number of
        requests that were active during that interval. This properly handles
        overlapping concurrent requests without double-counting.
        """
        if not self._all_samples:
            return

        # Ensure samples are sorted by timestamp
        samples = sorted(self._all_samples, key=lambda s: s.timestamp)

        # Map of request_index -> list of (shared_energy, raw_energy, shared_power, raw_power)
        request_allocations: Dict[int, list[Tuple[float, float, float, float]]] = {
            idx: [] for idx in self._request_timings
        }

        # Phase boundaries: request_idx -> prefill_end_time (absolute timestamp)
        phase_boundaries: dict[int, float] = {}
        model_name = self._config.model
        for idx, (r_start, r_end) in self._request_timings.items():
            if idx in self._records and model_name in self._records[idx].model_metrics:
                pm = self._records[idx].model_metrics[model_name].phase_metrics
                if pm.prefill_duration_ms is not None and pm.prefill_duration_ms > 0:
                    phase_boundaries[idx] = r_start + pm.prefill_duration_ms / 1000.0

        # Per-request phase energy accumulators: [prefill_shared, decode_shared]
        phase_energy: dict[int, list[float]] = {idx: [0.0, 0.0] for idx in self._request_timings}

        for i in range(1, len(samples)):
            s_prev = samples[i - 1]
            s_curr = samples[i]

            t_start = s_prev.timestamp
            t_end = s_curr.timestamp
            t_mid = (t_start + t_end) / 2

            # Energy delta for this interval
            e_prev = s_prev.reading.energy_joules
            e_curr = s_curr.reading.energy_joules

            if e_prev is None or e_curr is None:
                continue

            delta_joules = e_curr - e_prev
            if delta_joules < 0:
                delta_joules = 0.0

            # Power reading
            power_watts = s_curr.reading.power_watts or 0.0

            # Find requests that were active during this interval
            active_indices = []
            for idx, (r_start, r_end) in self._request_timings.items():
                if r_start <= t_mid <= r_end:
                    active_indices.append(idx)

            concurrency = len(active_indices)
            if concurrency > 0:
                # Divide energy/power by number of concurrent requests
                share_energy = delta_joules / concurrency
                share_power = power_watts / concurrency

                for idx in active_indices:
                    request_allocations[idx].append(
                        (share_energy, delta_joules, share_power, power_watts)
                    )
                    # Phase attribution
                    if idx in phase_boundaries:
                        if t_mid <= phase_boundaries[idx]:
                            phase_energy[idx][0] += share_energy  # prefill
                        else:
                            phase_energy[idx][1] += share_energy  # decode
                    else:
                        phase_energy[idx][1] += share_energy  # no boundary -> all decode

        # Update records with computed energy metrics
        for idx, allocations in request_allocations.items():
            if idx not in self._records:
                continue

            record = self._records[idx]
            if model_name not in record.model_metrics:
                continue

            if not allocations:
                continue

            # Sum up allocated energy and collect power readings
            shared_energies = [x[0] for x in allocations]
            raw_energies = [x[1] for x in allocations]
            shared_powers = [x[2] for x in allocations]
            raw_powers = [x[3] for x in allocations]

            total_shared_energy = sum(shared_energies)
            total_raw_energy = sum(raw_energies)

            shared_power_stats = _stat_summary(shared_powers)
            raw_power_stats = _stat_summary(raw_powers)

            metrics = record.model_metrics[model_name]

            # Update energy metrics
            metrics.energy_metrics = EnergyMetrics(
                per_query_joules=total_shared_energy,
                total_joules=total_raw_energy,
            )

            # Update power metrics
            metrics.power_metrics.gpu.per_query_watts = shared_power_stats
            metrics.power_metrics.gpu.total_watts = MetricStats(
                avg=raw_power_stats.avg,
                max=raw_power_stats.max,
                median=raw_power_stats.median,
                min=raw_power_stats.min,
            )

        # Update phase metrics with concurrency-aware energy
        for idx, (prefill_e, decode_e) in phase_energy.items():
            if idx not in self._records:
                continue
            record = self._records[idx]
            if model_name not in record.model_metrics:
                continue

            metrics = record.model_metrics[model_name]
            pm = metrics.phase_metrics

            # Compute scaling ratio from raw GPU component (not total which
            # includes CPU/ANE).  phase_energy accumulators only track GPU
            # energy (delta_joules comes from the energy_joules counter), so
            # the scale must compare against the raw GPU component.
            raw_gpu_prefill = (
                pm.prefill_energy_components_j.get("gpu", 0.0)
                if pm.prefill_energy_components_j
                else 0.0
            )
            raw_gpu_decode = (
                pm.decode_energy_components_j.get("gpu", 0.0)
                if pm.decode_energy_components_j
                else 0.0
            )
            prefill_scale = (
                (prefill_e / raw_gpu_prefill) if raw_gpu_prefill > 0 else 1.0
            )
            decode_scale = (
                (decode_e / raw_gpu_decode) if raw_gpu_decode > 0 else 1.0
            )

            # Scale component breakdowns (GPU, CPU, ANE) by the same ratio
            if pm.prefill_energy_components_j:
                pm.prefill_energy_components_j = {
                    k: v * prefill_scale for k, v in pm.prefill_energy_components_j.items()
                }
            if pm.decode_energy_components_j:
                pm.decode_energy_components_j = {
                    k: v * decode_scale for k, v in pm.decode_energy_components_j.items()
                }

            # Recompute total phase energy as sum of scaled components
            if pm.prefill_energy_components_j:
                pm.prefill_energy_j = sum(pm.prefill_energy_components_j.values())
            elif prefill_e > 0:
                pm.prefill_energy_j = prefill_e
            if pm.decode_energy_components_j:
                pm.decode_energy_j = sum(pm.decode_energy_components_j.values())
            elif decode_e > 0:
                pm.decode_energy_j = decode_e

            # Update power (energy / duration)
            if pm.prefill_duration_ms and pm.prefill_duration_ms > 0 and pm.prefill_energy_j:
                pm.prefill_power_avg_w = pm.prefill_energy_j / (pm.prefill_duration_ms / 1000.0)
            if pm.decode_duration_ms and pm.decode_duration_ms > 0 and pm.decode_energy_j:
                pm.decode_power_avg_w = pm.decode_energy_j / (pm.decode_duration_ms / 1000.0)

            # Update per-token metrics
            tk = metrics.token_metrics
            if tk.input and tk.input > 0 and pm.prefill_energy_j is not None:
                pm.prefill_energy_per_input_token_j = pm.prefill_energy_j / tk.input
            if tk.output and tk.output > 0 and pm.decode_energy_j is not None:
                pm.decode_energy_per_output_token_j = pm.decode_energy_j / tk.output

    def _prompt_iterator(self, dataset, total_queries: int):
        """Yield (index, prompt) tuples for the dataset."""
        for index, record in enumerate(dataset):
            if index >= total_queries:
                break
            yield index, record.problem

    def _prompt_iterator_from_list(self, records_list: list):
        """Yield (index, prompt) tuples from a pre-built list."""
        for index, record in enumerate(records_list):
            yield index, record.problem

    def _build_record(
        self,
        index: int,
        record: DatasetRecord,
        response: Response,
        samples: Sequence[TelemetrySample],
        start_time: float,
        end_time: float,
        phase_samples: Sequence[TelemetrySample] | None = None,
    ) -> Optional[ProfilingRecord]:
        phase_source = phase_samples if phase_samples is not None else samples
        self._update_hardware_metadata(phase_source)
        telemetry_readings = [sample.reading for sample in samples]

        # Energy metrics will be computed later by _recompute_metrics
        energy_metrics = EnergyMetrics()
        # Power stats are placeholders, will be updated by _recompute_metrics
        power_stats = _stat_summary(
            [reading.power_watts for reading in telemetry_readings]
        )
        cpu_power_stats = _stat_summary(
            [reading.cpu_power_watts for reading in telemetry_readings]
        )
        temperature_stats = _stat_summary(
            [reading.temperature_celsius for reading in telemetry_readings]
        )
        cpu_memory_stats = _stat_summary(
            [reading.cpu_memory_usage_mb for reading in telemetry_readings]
        )
        gpu_memory_stats = _stat_summary(
            [reading.gpu_memory_usage_mb for reading in telemetry_readings]
        )
        compute_util_stats = _stat_summary(
            [reading.gpu_compute_utilization_pct for reading in telemetry_readings]
        )
        memory_bw_util_stats = _stat_summary(
            [
                reading.gpu_memory_bandwidth_utilization_pct
                for reading in telemetry_readings
            ]
        )
        tensor_util_stats = _stat_summary(
            [reading.gpu_tensor_core_utilization_pct for reading in telemetry_readings]
        )

        memory_used_gb = _max_gb(
            [reading.gpu_memory_usage_mb for reading in telemetry_readings]
        )
        memory_total_gb = _max_gb(
            [reading.gpu_memory_total_mb for reading in telemetry_readings]
        )

        usage = response.usage
        total_seconds = max(end_time - start_time, 0.0)

        # Defensive: ensure token counts are valid integers
        prompt_tokens = usage.prompt_tokens if usage.prompt_tokens is not None else 0
        completion_tokens = (
            usage.completion_tokens if usage.completion_tokens is not None else 0
        )

        per_token_ms = None
        throughput_tokens = None
        if completion_tokens > 0 and total_seconds > 0:
            per_token_ms = (total_seconds * 1000.0) / completion_tokens
            throughput_tokens = completion_tokens / total_seconds

        phase_metrics = (
            self._compute_phase_metrics(
                phase_source,
                response,
                start_time,
                end_time,
                prompt_tokens,
                completion_tokens,
            )
            if self._config.phased_profiling
            else PhaseMetrics()
        )

        latency_metrics = LatencyMetrics(
            per_token_ms=per_token_ms,
            throughput_tokens_per_sec=throughput_tokens,
            time_to_first_token_seconds=(
                response.time_to_first_token_ms / 1000.0
                if response.time_to_first_token_ms is not None
                else None
            ),
            total_query_seconds=total_seconds,
        )

        model_name = self._config.model

        hardware_utilization = self._compute_hardware_utilization(
            compute_util_stats.avg,
            memory_bw_util_stats.avg,
            tensor_util_stats.avg,
            memory_used_gb,
            memory_total_gb,
            latency_metrics.total_query_seconds,
        )

        model_metrics = ModelMetrics(
            compute_metrics=ComputeMetrics(),
            energy_metrics=energy_metrics,
            latency_metrics=latency_metrics,
            memory_metrics=MemoryMetrics(
                cpu_mb=cpu_memory_stats,
                gpu_mb=gpu_memory_stats,
            ),
            power_metrics=PowerMetrics(
                gpu=PowerComponentMetrics(
                    per_query_watts=power_stats,
                    total_watts=MetricStats(
                        avg=power_stats.avg,
                        max=power_stats.max,
                        median=power_stats.median,
                        min=power_stats.min,
                    ),
                ),
                cpu=PowerComponentMetrics(
                    per_query_watts=cpu_power_stats,
                    total_watts=MetricStats(
                        avg=cpu_power_stats.avg,
                        max=cpu_power_stats.max,
                        median=cpu_power_stats.median,
                        min=cpu_power_stats.min,
                    ),
                ),
            ),
            temperature_metrics=temperature_stats,
            token_metrics=TokenMetrics(
                input=prompt_tokens,
                output=completion_tokens,
                total=prompt_tokens + completion_tokens,
            ),
            phase_metrics=phase_metrics,
            hardware_utilization=hardware_utilization,
            gpu_info=self._gpu_info,
            system_info=self._system_info,
            lm_correctness=False,
            lm_response=response.content,
        )

        record_payload = ProfilingRecord(
            problem=record.problem,
            answer=record.answer,
            dataset_metadata=dict(record.dataset_metadata),
            subject=record.subject,
            model_answers={model_name: response.content},
            model_metrics={model_name: model_metrics},
        )

        return record_payload

    def _compute_energy_metrics(
        self, readings: Sequence[TelemetryReading]
    ) -> EnergyMetrics:
        """Compute energy metrics from telemetry readings.

        Energy values should be monotonically increasing cumulative counters.
        Negative deltas indicate counter reset or data anomaly and are treated as None.
        """
        # GPU energy
        gpu_energy_values = [
            reading.energy_joules
            for reading in readings
            if reading.energy_joules is not None
        ]
        gpu_per_query = self._compute_energy_delta(gpu_energy_values)

        # CPU energy
        cpu_energy_values = [
            reading.cpu_energy_joules
            for reading in readings
            if reading.cpu_energy_joules is not None
        ]
        cpu_per_query = self._compute_energy_delta(cpu_energy_values)

        # ANE energy (macOS only)
        ane_energy_values = [
            reading.ane_energy_joules
            for reading in readings
            if reading.ane_energy_joules is not None
        ]
        ane_per_query = self._compute_energy_delta(ane_energy_values)

        return EnergyMetrics(
            per_query_joules=gpu_per_query,
            total_joules=gpu_per_query,
            cpu_per_query_joules=cpu_per_query,
            cpu_total_joules=cpu_per_query,
            ane_per_query_joules=ane_per_query,
            ane_total_joules=ane_per_query,
        )

    def _compute_hardware_utilization(
        self,
        compute_util_pct: Optional[float],
        memory_bw_util_pct: Optional[float],
        tensor_util_pct: Optional[float],
        memory_used_gb: Optional[float],
        memory_total_gb: Optional[float],
        total_query_seconds: Optional[float],
    ) -> HardwareUtilization:
        peak_tflops = self._peak_compute_tflops
        peak_bw_gbps = self._peak_memory_bandwidth_gbps

        actual_tflops: Optional[float] = None
        if peak_tflops is not None and compute_util_pct is not None:
            actual_tflops = peak_tflops * (compute_util_pct / 100.0)

        actual_bw_gbps: Optional[float] = None
        if peak_bw_gbps is not None and memory_bw_util_pct is not None:
            actual_bw_gbps = peak_bw_gbps * (memory_bw_util_pct / 100.0)

        mfu = _safe_div(actual_tflops, peak_tflops) if peak_tflops else None
        mbu = _safe_div(actual_bw_gbps, peak_bw_gbps) if peak_bw_gbps else None

        arithmetic_intensity = None
        if actual_tflops is not None and actual_bw_gbps is not None:
            if actual_bw_gbps > 0:
                arithmetic_intensity = (actual_tflops * 1e12) / (actual_bw_gbps * 1e9)

        return HardwareUtilization(
            gpu=HardwareUtilizationGpu(
                compute_utilization_pct=compute_util_pct,
                memory_bandwidth_utilization_pct=memory_bw_util_pct,
                tensor_core_utilization_pct=tensor_util_pct,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
            ),
            derived=HardwareUtilizationDerived(
                mfu=mfu,
                mbu=mbu,
                arithmetic_intensity=arithmetic_intensity,
            ),
        )

    def _benchmark_hardware_if_needed(self) -> None:
        if self._hardware_benchmark_info is not None:
            return

        if not getattr(self._config, "run_hardware_benchmarks", True):
            return

        try:
            import src.benchmarks  # noqa: F401
            from src.benchmarks.types import DataType, WorkloadConfig, WorkloadType
            from src.core.registry import BenchmarkRegistry
        except Exception:
            return

        suite_cls = None
        for _, candidate in BenchmarkRegistry.items():
            try:
                if candidate.is_available():
                    suite_cls = candidate
                    break
            except Exception:
                continue

        if suite_cls is None:
            return

        try:
            suite = suite_cls()
        except Exception:
            return

        duration_s = 2.0
        array_size_mb = 512
        matrix_size = 2048
        arithmetic_intensity = 128

        bench_info: dict[str, Any] = {
            "platform": getattr(getattr(suite, "platform", None), "value", None),
            "hardware_name": None,
            "peak_compute_tflops": None,
            "peak_memory_bandwidth_gbps": None,
            "compute": None,
            "memory": None,
        }

        try:
            bench_info["hardware_name"] = suite.detect_hardware()
        except Exception:
            pass

        preferred_dtypes = [
            DataType.FP16,
            DataType.BF16,
            DataType.FP32,
            DataType.FP64,
        ]

        def _pick_dtype(supported):
            for candidate in preferred_dtypes:
                if candidate in supported:
                    return candidate
            return supported[0] if supported else None

        memory_peak = None
        memory_meta = None
        try:
            workload_type = WorkloadType.MEMORY_BANDWIDTH
            memory_workload = suite.get_memory_workload()
            if hasattr(suite, "get_hbm_workload"):
                try:
                    memory_workload = suite.get_hbm_workload()
                    workload_type = WorkloadType.HBM_BANDWIDTH
                except Exception:
                    memory_workload = suite.get_memory_workload()
                    workload_type = WorkloadType.MEMORY_BANDWIDTH

            mem_config = WorkloadConfig(
                workload_type=workload_type,
                duration_seconds=duration_s,
                use_zeros=False,
                data_type=DataType.FP32,
                params={"array_size_mb": array_size_mb},
            )
            memory_workload.warmup(mem_config)
            mem_result = memory_workload.run(mem_config)
            if mem_result.throughput_unit.lower().startswith("gb"):
                memory_peak = mem_result.throughput
            memory_meta = {
                "workload": workload_type.value,
                "array_size_mb": array_size_mb,
                "duration_seconds": mem_result.duration_seconds,
                "throughput_gbps": mem_result.throughput,
                "data_type": mem_config.data_type.value,
            }
        except Exception:
            memory_meta = None

        compute_peak = None
        compute_meta = None
        try:
            gemm_workload = suite.get_gemm_workload()
            dtype = _pick_dtype(gemm_workload.supported_data_types())
            if dtype is not None:
                compute_config = WorkloadConfig(
                    workload_type=WorkloadType.GEMM,
                    duration_seconds=duration_s,
                    use_zeros=False,
                    data_type=dtype,
                    params={"matrix_size": matrix_size},
                )
                gemm_workload.warmup(compute_config)
                gemm_result = gemm_workload.run(compute_config)
                if gemm_result.throughput_unit.lower().startswith("tflop"):
                    compute_peak = gemm_result.throughput
                compute_meta = {
                    "workload": WorkloadType.GEMM.value,
                    "matrix_size": matrix_size,
                    "duration_seconds": gemm_result.duration_seconds,
                    "throughput_tflops": gemm_result.throughput,
                    "data_type": compute_config.data_type.value,
                }
        except Exception:
            compute_meta = None

        if compute_peak is None:
            try:
                compute_workload = suite.get_compute_workload()
                dtype = _pick_dtype(compute_workload.supported_data_types())
                if dtype is not None:
                    compute_config = WorkloadConfig(
                        workload_type=WorkloadType.COMPUTE_BOUND,
                        duration_seconds=duration_s,
                        use_zeros=False,
                        data_type=dtype,
                        params={
                            "arithmetic_intensity": arithmetic_intensity,
                            "array_size_mb": 100,
                        },
                    )
                    compute_workload.warmup(compute_config)
                    compute_result = compute_workload.run(compute_config)
                    if compute_result.throughput_unit.lower().startswith("tflop"):
                        compute_peak = compute_result.throughput
                    compute_meta = {
                        "workload": WorkloadType.COMPUTE_BOUND.value,
                        "arithmetic_intensity": arithmetic_intensity,
                        "duration_seconds": compute_result.duration_seconds,
                        "throughput_tflops": compute_result.throughput,
                        "data_type": compute_config.data_type.value,
                    }
            except Exception:
                pass

        if memory_peak is not None:
            self._peak_memory_bandwidth_gbps = memory_peak
            bench_info["peak_memory_bandwidth_gbps"] = memory_peak
            bench_info["memory"] = memory_meta

        if compute_peak is not None:
            self._peak_compute_tflops = compute_peak
            bench_info["peak_compute_tflops"] = compute_peak
            bench_info["compute"] = compute_meta

        if memory_peak is not None or compute_peak is not None:
            self._hardware_benchmark_info = bench_info

    def _compute_phase_metrics(
        self,
        samples: Sequence[TelemetrySample],
        response: Response,
        start_time: float,
        end_time: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> PhaseMetrics:
        total_duration_s = max(end_time - start_time, 0.0)
        if total_duration_s <= 0.0:
            return PhaseMetrics()

        total_duration_ms = total_duration_s * 1000.0
        ttft_ms = response.time_to_first_token_ms or 0.0

        if completion_tokens <= 0:
            prefill_end_time = end_time
        else:
            first_token_time = getattr(response, "first_token_time", None)
            if (
                first_token_time is not None
                and isinstance(first_token_time, (int, float))
                and math.isfinite(first_token_time)
            ):
                prefill_end_time = min(max(float(first_token_time), start_time), end_time)
            else:
                prefill_duration_ms = min(max(ttft_ms, 0.0), total_duration_ms)
                prefill_end_time = start_time + (prefill_duration_ms / 1000.0)

        prefill_duration_ms = max((prefill_end_time - start_time) * 1000.0, 0.0)
        decode_duration_ms = max((end_time - prefill_end_time) * 1000.0, 0.0)

        gpu_prefill_energy = _compute_energy_delta_overlap_window(
            samples, start_time, prefill_end_time, attr="energy_joules"
        )
        gpu_decode_energy = _compute_energy_delta_overlap_window(
            samples, prefill_end_time, end_time, attr="energy_joules"
        )
        cpu_prefill_energy = _compute_energy_delta_overlap_window(
            samples, start_time, prefill_end_time, attr="cpu_energy_joules"
        )
        cpu_decode_energy = _compute_energy_delta_overlap_window(
            samples, prefill_end_time, end_time, attr="cpu_energy_joules"
        )
        ane_prefill_energy = _compute_energy_delta_overlap_window(
            samples, start_time, prefill_end_time, attr="ane_energy_joules"
        )
        ane_decode_energy = _compute_energy_delta_overlap_window(
            samples, prefill_end_time, end_time, attr="ane_energy_joules"
        )

        prefill_duration_s = prefill_duration_ms / 1000.0 if prefill_duration_ms else 0.0
        decode_duration_s = decode_duration_ms / 1000.0 if decode_duration_ms else 0.0

        prefill_energy = _sum_optional(
            gpu_prefill_energy, cpu_prefill_energy, ane_prefill_energy
        )
        decode_energy = _sum_optional(
            gpu_decode_energy, cpu_decode_energy, ane_decode_energy
        )

        prefill_power = _safe_div(prefill_energy, prefill_duration_s)
        if prefill_power is None:
            prefill_power = _compute_power_average_window_multi(
                samples, start_time, prefill_end_time
            )
        decode_power = _safe_div(decode_energy, decode_duration_s)
        if decode_power is None:
            decode_power = _compute_power_average_window_multi(
                samples, prefill_end_time, end_time
            )

        prefill_energy_per_token = None
        if prompt_tokens > 0 and prefill_energy is not None:
            prefill_energy_per_token = prefill_energy / prompt_tokens

        decode_energy_per_token = None
        if completion_tokens > 0 and decode_energy is not None:
            decode_energy_per_token = decode_energy / completion_tokens

        prefill_components = _build_energy_components(
            gpu_prefill_energy, cpu_prefill_energy, ane_prefill_energy
        )
        decode_components = _build_energy_components(
            gpu_decode_energy, cpu_decode_energy, ane_decode_energy
        )

        return PhaseMetrics(
            prefill_energy_j=prefill_energy,
            decode_energy_j=decode_energy,
            prefill_duration_ms=prefill_duration_ms,
            decode_duration_ms=decode_duration_ms,
            prefill_power_avg_w=prefill_power,
            decode_power_avg_w=decode_power,
            prefill_energy_per_input_token_j=prefill_energy_per_token,
            decode_energy_per_output_token_j=decode_energy_per_token,
            prefill_energy_components_j=prefill_components,
            decode_energy_components_j=decode_components,
        )

    def _summarize_phase_metrics(
        self, metrics_entries: Sequence[ModelMetrics]
    ) -> Optional[dict[str, Any]]:
        if not metrics_entries:
            return None

        prefill_energy = [
            entry.phase_metrics.prefill_energy_j
            for entry in metrics_entries
            if entry.phase_metrics.prefill_energy_j is not None
        ]
        decode_energy = [
            entry.phase_metrics.decode_energy_j
            for entry in metrics_entries
            if entry.phase_metrics.decode_energy_j is not None
        ]
        prefill_power = [
            entry.phase_metrics.prefill_power_avg_w
            for entry in metrics_entries
            if entry.phase_metrics.prefill_power_avg_w is not None
        ]
        decode_power = [
            entry.phase_metrics.decode_power_avg_w
            for entry in metrics_entries
            if entry.phase_metrics.decode_power_avg_w is not None
        ]
        prefill_duration = [
            entry.phase_metrics.prefill_duration_ms
            for entry in metrics_entries
            if entry.phase_metrics.prefill_duration_ms is not None
        ]
        decode_duration = [
            entry.phase_metrics.decode_duration_ms
            for entry in metrics_entries
            if entry.phase_metrics.decode_duration_ms is not None
        ]
        prefill_energy_per_token = [
            entry.phase_metrics.prefill_energy_per_input_token_j
            for entry in metrics_entries
            if entry.phase_metrics.prefill_energy_per_input_token_j is not None
        ]
        decode_energy_per_token = [
            entry.phase_metrics.decode_energy_per_output_token_j
            for entry in metrics_entries
            if entry.phase_metrics.decode_energy_per_output_token_j is not None
        ]

        prefill_total = sum(prefill_energy) if prefill_energy else None
        decode_total = sum(decode_energy) if decode_energy else None
        combined_total = (
            (prefill_total or 0.0) + (decode_total or 0.0)
            if prefill_total is not None or decode_total is not None
            else None
        )

        def _mean(values: Sequence[float]) -> Optional[float]:
            if not values:
                return None
            return sum(values) / len(values)

        prefill_fraction = (
            prefill_total / combined_total
            if prefill_total is not None
            and combined_total is not None
            and combined_total > 0.0
            else None
        )
        decode_fraction = (
            decode_total / combined_total
            if decode_total is not None
            and combined_total is not None
            and combined_total > 0.0
            else None
        )

        return {
            "prefill": {
                "total_energy_j": prefill_total,
                "energy_fraction": prefill_fraction,
                "mean_power_w": _mean(prefill_power),
                "mean_duration_ms": _mean(prefill_duration),
                "mean_energy_per_input_token_j": _mean(prefill_energy_per_token),
            },
            "decode": {
                "total_energy_j": decode_total,
                "energy_fraction": decode_fraction,
                "mean_power_w": _mean(decode_power),
                "mean_duration_ms": _mean(decode_duration),
                "mean_energy_per_output_token_j": _mean(decode_energy_per_token),
            },
        }

    def _compute_energy_delta(
        self, energy_values: list[float]
    ) -> Optional[float]:
        """Compute energy delta from a list of cumulative energy values."""
        if not energy_values:
            return None

        start_value = energy_values[0]
        end_value = energy_values[-1]

        # Validate energy values are finite and non-negative
        if not (
            math.isfinite(start_value)
            and math.isfinite(end_value)
            and start_value >= 0
            and end_value >= 0
        ):
            return None

        per_query_delta = end_value - start_value
        return per_query_delta if per_query_delta >= 0 else None

    def _update_hardware_metadata(self, readings: Sequence[TelemetrySample]) -> None:
        for sample in readings:
            reading = sample.reading
            if reading.system_info is not None:
                self._system_info = reading.system_info
            if reading.gpu_info is not None:
                self._gpu_info = reading.gpu_info

        candidate = derive_hardware_label(self._system_info, self._gpu_info)
        if candidate and (self._hardware_label in (None, "UNKNOWN_HW")):
            self._hardware_label = candidate

    def _get_output_path(self) -> Path:
        if self._output_path is not None:
            return self._output_path

        hardware_label = self._hardware_label or "UNKNOWN_HW"
        model_slug = _slugify_model(self._config.model)
        default_runs_dir = Path(__file__).resolve().parents[4] / "runs"
        base_dir = self._config.output_dir or default_runs_dir
        profile_dir = f"profile_{hardware_label}_{model_slug}".strip("_")

        output_path = Path(base_dir) / profile_dir

        self._hardware_label = hardware_label
        self._output_path = output_path
        return output_path

    def _resolve_dataset(self, dataset_id: str, params: Mapping[str, Any]):
        try:
            dataset_cls = DatasetRegistry.get(dataset_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown dataset '{dataset_id}'") from exc

        resolved_params: dict[str, Any] = dict(params)
        if dataset_id == "synthetic":
            resolved_params.setdefault("tokenizer_model", self._config.model)

        try:
            return dataset_cls(**resolved_params)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate dataset '{dataset_id}' with params {resolved_params!r}: {exc}"
            ) from exc

    def _resolve_client(
        self,
        client_id: str,
        base_url: str | None,
        params: Mapping[str, Any],
    ) -> InferenceClient:
        try:
            client_cls = ClientRegistry.get(client_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown client '{client_id}'") from exc

        try:
            return client_cls(base_url, **params)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate client '{client_id}' with params {params!r}: {exc}"
            ) from exc

    def _ensure_client_ready(self, client: InferenceClient) -> None:
        if not client.health():
            raise RuntimeError(
                f"Client '{client.client_name}' at {getattr(client, 'base_url', '')} is unavailable"
            )
        client.prepare(self._config.model)

    def _flatten_record(self, record: ProfilingRecord, index: int) -> dict[str, Any]:
        row = asdict(record)

        query_id = None
        if isinstance(record.dataset_metadata, Mapping):
            metadata_id = record.dataset_metadata.get("query_id")
            if metadata_id is not None:
                query_id = str(metadata_id)
        if query_id is None:
            query_id = str(index)

        row.update(
            {
                "query_id": query_id,
                "input_tokens": None,
                "output_tokens": None,
                "energy_j": None,
                "power_avg_w": None,
                "latency_ms": None,
                "ttft_ms": None,
                "prefill_energy_j": None,
                "decode_energy_j": None,
                "prefill_duration_ms": None,
                "decode_duration_ms": None,
                "prefill_power_avg_w": None,
                "decode_power_avg_w": None,
                "prefill_energy_per_input_token_j": None,
                "decode_energy_per_output_token_j": None,
                "gpu_compute_utilization_pct": None,
                "gpu_memory_bandwidth_utilization_pct": None,
                "gpu_tensor_core_utilization_pct": None,
                "gpu_memory_used_gb": None,
                "gpu_memory_total_gb": None,
                "mfu": None,
                "mbu": None,
                "arithmetic_intensity": None,
            }
        )

        model_metrics = record.model_metrics.get(self._config.model)
        if model_metrics is None:
            return row

        token_metrics = model_metrics.token_metrics
        row["input_tokens"] = token_metrics.input
        row["output_tokens"] = token_metrics.output

        energy_metrics = model_metrics.energy_metrics
        total_energy = _sum_optional(
            energy_metrics.per_query_joules,
            energy_metrics.cpu_per_query_joules,
            energy_metrics.ane_per_query_joules,
        )
        row["energy_j"] = total_energy

        latency_metrics = model_metrics.latency_metrics
        if latency_metrics.total_query_seconds is not None:
            row["latency_ms"] = latency_metrics.total_query_seconds * 1000.0
        if latency_metrics.time_to_first_token_seconds is not None:
            row["ttft_ms"] = latency_metrics.time_to_first_token_seconds * 1000.0

        power_metrics = model_metrics.power_metrics
        total_power = None
        if latency_metrics.total_query_seconds is not None:
            total_power = _safe_div(total_energy, latency_metrics.total_query_seconds)
        if total_power is None:
            total_power = _sum_optional(
                power_metrics.gpu.per_query_watts.avg,
                power_metrics.cpu.per_query_watts.avg,
            )
        row["power_avg_w"] = total_power

        phase_metrics = model_metrics.phase_metrics
        row["prefill_energy_j"] = phase_metrics.prefill_energy_j
        row["decode_energy_j"] = phase_metrics.decode_energy_j
        row["prefill_duration_ms"] = phase_metrics.prefill_duration_ms
        row["decode_duration_ms"] = phase_metrics.decode_duration_ms
        row["prefill_power_avg_w"] = phase_metrics.prefill_power_avg_w
        row["decode_power_avg_w"] = phase_metrics.decode_power_avg_w
        row[
            "prefill_energy_per_input_token_j"
        ] = phase_metrics.prefill_energy_per_input_token_j
        row[
            "decode_energy_per_output_token_j"
        ] = phase_metrics.decode_energy_per_output_token_j

        hardware_util = model_metrics.hardware_utilization
        row["gpu_compute_utilization_pct"] = hardware_util.gpu.compute_utilization_pct
        row["gpu_memory_bandwidth_utilization_pct"] = (
            hardware_util.gpu.memory_bandwidth_utilization_pct
        )
        row["gpu_tensor_core_utilization_pct"] = (
            hardware_util.gpu.tensor_core_utilization_pct
        )
        row["gpu_memory_used_gb"] = hardware_util.gpu.memory_used_gb
        row["gpu_memory_total_gb"] = hardware_util.gpu.memory_total_gb
        row["mfu"] = hardware_util.derived.mfu
        row["mbu"] = hardware_util.derived.mbu
        row["arithmetic_intensity"] = hardware_util.derived.arithmetic_intensity

        return row

    def _persist_records(self, dataset) -> None:
        if not self._records:
            return

        output_path = self._get_output_path()
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort records by index for consistent output
        ordered_records = [self._records[idx] for idx in sorted(self._records.keys())]

        dataset_obj = Dataset.from_list(
            [
                self._flatten_record(record, index)
                for index, record in enumerate(ordered_records)
            ]
        )
        dataset_obj.save_to_disk(str(output_path))
        output_path.mkdir(parents=True, exist_ok=True)

        model_name = self._config.model
        metrics_entries = [
            record.model_metrics.get(model_name)
            for record in ordered_records
            if record.model_metrics.get(model_name) is not None
        ]
        energy_values = []
        for entry in metrics_entries:
            total = _sum_optional(
                entry.energy_metrics.per_query_joules,
                entry.energy_metrics.cpu_per_query_joules,
                entry.energy_metrics.ane_per_query_joules,
            )
            if total is not None:
                energy_values.append(total)
        total_energy = sum(energy_values) if energy_values else None
        phase_summary = (
            self._summarize_phase_metrics(metrics_entries)
            if self._config.phased_profiling
            else None
        )

        summary = {
            "model": self._config.model,
            "dataset": getattr(dataset, "dataset_id", self._config.dataset_id),
            "dataset_name": getattr(dataset, "dataset_name", None),
            "hardware_label": self._hardware_label,
            "max_concurrency": self._config.max_concurrency,
            "generated_at": time.time(),
            "total_queries": len(self._records),
            "totals": {
                "total_energy_j": total_energy,
                "total_queries": len(self._records),
            },
            "hardware_benchmarks": self._hardware_benchmark_info,
            "system_info": asdict(self._system_info) if self._system_info else None,
            "gpu_info": asdict(self._gpu_info) if self._gpu_info else None,
            "output_dir": str(output_path),
            "run_metadata": {
                "phased_profiling": self._config.phased_profiling,
                "max_concurrency": self._config.max_concurrency,
                "phase_detection_method": (
                    "streaming" if self._config.phased_profiling else None
                ),
            },
            "profiler_config": _jsonify(asdict(self._config)),
        }
        if phase_summary:
            summary["phase_summary"] = phase_summary
        summary_path = output_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))


def _stat_summary(values: Iterable[Optional[float]]) -> MetricStats:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return MetricStats()
    return MetricStats(
        avg=sum(filtered) / len(filtered),
        max=max(filtered),
        median=statistics.median(filtered),
        min=min(filtered),
    )


def _max_gb(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return max(filtered) / 1024.0


def _slugify_model(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"


def _jsonify(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonify(item) for item in value]
    return value


def _safe_div(numerator: Optional[float], denominator: float) -> Optional[float]:
    if numerator is None or denominator <= 0.0:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / denominator


def _sum_optional(*values: Optional[float]) -> Optional[float]:
    total = 0.0
    found = False
    for value in values:
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _extract_series(
    samples: Sequence[TelemetrySample], attr: str
) -> list[tuple[float, float]]:
    series: list[tuple[float, float]] = []
    for sample in samples:
        value = getattr(sample.reading, attr, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        series.append((sample.timestamp, numeric))
    series.sort(key=lambda item: item[0])
    return series


def _interpolate_series_value(
    series: Sequence[tuple[float, float]], timestamp: float
) -> Optional[float]:
    if not series:
        return None

    if timestamp <= series[0][0]:
        return series[0][1]
    if timestamp >= series[-1][0]:
        return series[-1][1]

    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= timestamp <= t1:
            if math.isclose(t0, t1):
                return v1
            ratio = (timestamp - t0) / (t1 - t0)
            return v0 + (v1 - v0) * ratio
    return None


def _slice_series(
    series: Sequence[tuple[float, float]], start_time: float, end_time: float
) -> list[tuple[float, float]]:
    if not series or end_time <= start_time:
        return []

    points: list[tuple[float, float]] = []
    start_val = _interpolate_series_value(series, start_time)
    if start_val is not None:
        points.append((start_time, start_val))

    for timestamp, value in series:
        if start_time < timestamp < end_time:
            points.append((timestamp, value))

    end_val = _interpolate_series_value(series, end_time)
    if end_val is not None:
        points.append((end_time, end_val))

    points.sort(key=lambda item: item[0])
    return points


def _compute_energy_delta_overlap_window(
    samples: Sequence[TelemetrySample],
    start_time: float,
    end_time: float,
    *,
    attr: str,
) -> Optional[float]:
    if end_time <= start_time:
        return None

    series = _extract_series(samples, attr)
    if len(series) < 2:
        return None

    total = 0.0
    has_overlap = False

    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        dt = t1 - t0
        if dt <= 0.0:
            continue

        overlap_start = max(start_time, t0)
        overlap_end = min(end_time, t1)
        if overlap_end <= overlap_start:
            continue

        delta = v1 - v0
        if delta < 0.0:
            continue

        overlap_ratio = (overlap_end - overlap_start) / dt
        total += delta * overlap_ratio
        has_overlap = True

    return total if has_overlap else None


def _compute_power_average_window(
    samples: Sequence[TelemetrySample],
    start_time: float,
    end_time: float,
    *,
    attr: str,
) -> Optional[float]:
    if end_time <= start_time:
        return None

    series = _extract_series(samples, attr)
    points = _slice_series(series, start_time, end_time)
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]

    total_energy = 0.0
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total_energy += 0.5 * (v0 + v1) * dt

    duration = end_time - start_time
    if duration <= 0.0:
        return None
    return total_energy / duration


def _compute_power_average_window_multi(
    samples: Sequence[TelemetrySample],
    start_time: float,
    end_time: float,
) -> Optional[float]:
    if end_time <= start_time:
        return None

    gpu_power = _compute_power_average_window(
        samples, start_time, end_time, attr="power_watts"
    )
    cpu_power = _compute_power_average_window(
        samples, start_time, end_time, attr="cpu_power_watts"
    )
    ane_power = _compute_power_average_window(
        samples, start_time, end_time, attr="ane_power_watts"
    )

    total_power = _sum_optional(gpu_power, cpu_power, ane_power)
    return total_power


def _build_energy_components(
    gpu_energy: Optional[float],
    cpu_energy: Optional[float],
    ane_energy: Optional[float] = None,
) -> Optional[dict[str, float]]:
    components: dict[str, float] = {}
    if gpu_energy is not None:
        components["gpu"] = gpu_energy
    if cpu_energy is not None:
        components["cpu"] = cpu_energy
    if ane_energy is not None:
        components["ane"] = ane_energy
    return components or None
