"""Simulator oracle interface and roofline implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec, WorkloadSpec

from ipw.simulator.hardware_specs import HardwareSpecs
from ipw.simulator.inference_model import predict as predict_single

logger = logging.getLogger(__name__)


@runtime_checkable
class SimulatorOracle(Protocol):
    """Abstract interface for a simulator backend.

    The search system calls this to evaluate a configuration at a given QPS.
    Implementations must return a flat dict of metric_name -> value.
    """

    def simulate(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        workload_spec: WorkloadSpec,
    ) -> Dict[str, float]: ...


class RooflineOracle:
    """Oracle backed by the ipw.simulator roofline model.

    Wraps the analytical prefill/decode predictor to compute per-request
    latency and energy, then scales to throughput metrics at the given QPS.

    The workload_spec.qps field controls the offered load. At higher QPS,
    latency degrades due to queuing (modeled as a simple M/D/1 approximation).

    Attributes:
        accuracy_score: Model accuracy score for IPW/IPJ computation.
        price_per_hour_usd: Cost per GPU-hour for cost_per_query_usd.
    """

    def __init__(
        self,
        accuracy_score: float = 1.0,
        price_per_hour_usd: float = 0.0,
    ) -> None:
        self.accuracy_score = accuracy_score
        self.price_per_hour_usd = price_per_hour_usd

    def simulate(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        workload_spec: WorkloadSpec,
    ) -> Dict[str, float]:
        """Run roofline simulation and return metrics dict.

        Metrics returned:
            ttft_s: Time to first token (prefill latency).
            tbt_s: Time between tokens (per-token decode latency).
            e2e_latency_s: End-to-end latency for one request.
            throughput_tps: Tokens per second (output throughput at given QPS).
            throughput_rps: Requests per second (equals offered QPS if sustainable).
            total_energy_j: Total energy for all requests over the workload duration.
            avg_power_w: Average power draw in watts.
        """
        # Build an ipw HardwareSpecs from the inference_simulator HardwareSpec
        hw = HardwareSpecs(
            name=hardware_spec.name,
            vendor=hardware_spec.vendor,
            memory_gb=hardware_spec.memory_gb,
            tdp_watts=hardware_spec.tdp_watts,
            peak_fp16_tflops=hardware_spec.peak_fp16_tflops,
            peak_fp8_tflops=hardware_spec.peak_fp8_tflops,
            peak_bf16_tflops=hardware_spec.peak_bf16_tflops,
            hbm_bandwidth_gb_s=hardware_spec.hbm_bandwidth_gb_s,
            nvlink_bandwidth_gb_s=hardware_spec.nvlink_bandwidth_gb_s,
            bytes_per_param_fp16=hardware_spec.bytes_per_param_fp16,
            bytes_per_param_fp8=hardware_spec.bytes_per_param_fp8,
        )

        # Determine bytes per parameter from precision
        if inference_spec.precision == "fp8" and hw.peak_fp8_tflops > 0:
            bytes_per_param = 1.0
        else:
            bytes_per_param = 2.0

        active_params_b = model_spec.total_params_billion
        num_gpus = inference_spec.num_gpus

        # Single-request prediction
        single = predict_single(
            hw=hw,
            active_params_b=active_params_b,
            input_tokens=workload_spec.avg_input_tokens,
            output_tokens=workload_spec.avg_output_tokens,
            bytes_per_param=bytes_per_param,
            num_gpus=num_gpus,
        )

        ttft_s = single.prefill.time_seconds
        tbt_s = (
            single.decode.time_seconds / workload_spec.avg_output_tokens
            if workload_spec.avg_output_tokens > 0
            else 0.0
        )
        e2e_latency_s = single.total_time_seconds

        # Service time per request
        service_time_s = e2e_latency_s
        qps = workload_spec.qps

        # M/D/1 queuing approximation for latency under load
        # utilization rho = arrival_rate * service_time
        if service_time_s > 0:
            max_capacity = 1.0 / service_time_s
            rho = qps / max_capacity if max_capacity > 0 else 1.0
        else:
            rho = 0.0
            max_capacity = float("inf")

        # Clamp rho — if rho >= 1 the system is overloaded
        if rho >= 1.0:
            # System cannot sustain this QPS
            queuing_factor = 100.0  # large penalty
        elif rho > 0:
            # M/D/1 waiting time: W = rho / (2 * mu * (1 - rho))
            queuing_factor = 1.0 + rho / (2.0 * (1.0 - rho))
        else:
            queuing_factor = 1.0

        effective_ttft = ttft_s * queuing_factor
        effective_e2e = e2e_latency_s * queuing_factor

        # Throughput: limited by capacity
        effective_rps = min(qps, max_capacity) if max_capacity > 0 else 0.0
        effective_tps = effective_rps * workload_spec.avg_output_tokens

        # Power model: alpha * TDP * num_gpus (with utilization scaling)
        alpha = 0.65
        power_w = hw.tdp_watts * num_gpus * alpha
        # Scale power with utilization (idle vs loaded)
        idle_fraction = 0.1
        power_w = power_w * max(rho, idle_fraction) / 1.0 if rho < 1.0 else power_w

        # Energy over a nominal 1-second window at this QPS
        total_energy_j = power_w  # power * 1 second = energy in joules

        # Energy per query
        energy_per_query = total_energy_j / effective_rps if effective_rps > 0 else float("inf")

        # IPW and IPJ: intelligence-per-watt / intelligence-per-joule
        ipw = self.accuracy_score / power_w if power_w > 0 else 0.0
        ipj = self.accuracy_score / energy_per_query if energy_per_query > 0 and energy_per_query != float("inf") else 0.0

        # Cost per query: (price_per_hour * num_gpus * e2e_time / 3600)
        cost_per_query_usd = (
            self.price_per_hour_usd * num_gpus * effective_e2e / 3600.0
        )

        # Percentile approximations using queuing factor
        # p50 ~ base * qf^0.5, p90 ~ base * qf^0.9, p95 ~ base * qf^0.95, p99 ~ base * qf^1.2
        def _percentiles(base: float, qf: float) -> tuple:
            return (
                base * qf**0.5,
                base * qf**0.9,
                base * qf**0.95,
                base * qf**1.2,
            )

        ttft_p50, ttft_p90, ttft_p95, ttft_p99 = _percentiles(ttft_s, queuing_factor)
        tbt_p50, tbt_p90, tbt_p95, tbt_p99 = _percentiles(tbt_s, queuing_factor)
        e2e_p50, e2e_p90, e2e_p95, e2e_p99 = _percentiles(e2e_latency_s, queuing_factor)

        return {
            "ttft_s": effective_ttft,
            "tbt_s": tbt_s,
            "e2e_latency_s": effective_e2e,
            "throughput_tps": effective_tps,
            "throughput_rps": effective_rps,
            "total_energy_j": total_energy_j,
            "avg_power_w": power_w,
            "ipw": ipw,
            "ipj": ipj,
            "cost_per_query_usd": cost_per_query_usd,
            "energy_per_query_j": energy_per_query,
            "ttft_p50": ttft_p50,
            "ttft_p90": ttft_p90,
            "ttft_p95": ttft_p95,
            "ttft_p99": ttft_p99,
            "tbt_p50": tbt_p50,
            "tbt_p90": tbt_p90,
            "tbt_p95": tbt_p95,
            "tbt_p99": tbt_p99,
            "e2e_p50": e2e_p50,
            "e2e_p90": e2e_p90,
            "e2e_p95": e2e_p95,
            "e2e_p99": e2e_p99,
        }


class EventDrivenOracle:
    """Oracle backed by the full event-driven simulator.

    Wraps EventDrivenSimulator with VLLMScheduler to run discrete-event
    simulation and return metrics matching the RooflineOracle interface.
    Caches LUT bundles across calls for the same model/hardware pair.

    Attributes:
        lut_bundle_dir: Directory containing LUT bundle files.
        accuracy_score: Model accuracy score for IPW/IPJ computation.
        price_per_hour_usd: Cost per GPU-hour for cost_per_query_usd.
        simulation_duration_s: Duration of each simulation run in seconds.
    """

    def __init__(
        self,
        lut_bundle_dir: Optional[Path] = None,
        accuracy_score: float = 1.0,
        price_per_hour_usd: float = 0.0,
        simulation_duration_s: float = 30.0,
    ) -> None:
        self.lut_bundle_dir = lut_bundle_dir
        self.accuracy_score = accuracy_score
        self.price_per_hour_usd = price_per_hour_usd
        self.simulation_duration_s = simulation_duration_s
        self._lut_cache: Dict[str, Any] = {}

    def _load_lut_bundle(
        self, model_spec: ModelSpec, hardware_spec: HardwareSpec, inference_spec: InferenceSpec
    ) -> Any:
        """Load LUT bundle, caching by model+hardware key."""
        if self.lut_bundle_dir is None:
            return None

        cache_key = f"{model_spec.model_id}:{hardware_spec.name}:{inference_spec.precision}"
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]

        from inference_simulator.types.lut_bundle import LUTBundle

        base_dir = Path(self.lut_bundle_dir)
        if not base_dir.exists():
            logger.warning("LUT bundle directory does not exist: %s", base_dir)
            self._lut_cache[cache_key] = None
            return None

        # Look for bundle files matching the model/hardware/precision
        token_ops = base_dir / "gpu_token_ops.npz"
        prefill_lut = base_dir / "gpu_attention_prefill.npz"
        decode_lut = base_dir / "gpu_attention_decode.npz"

        if not all(p.exists() for p in [token_ops, prefill_lut, decode_lut]):
            logger.warning("Required LUT files not found in %s", base_dir)
            self._lut_cache[cache_key] = None
            return None

        bundle = LUTBundle(
            base_dir=base_dir,
            model_id=model_spec.model_id if hasattr(model_spec, "model_id") else "",
            hardware_id=hardware_spec.name,
            quantization=inference_spec.precision,
            gpu_token_ops_lut=token_ops,
            gpu_attention_prefill_lut=prefill_lut,
            gpu_attention_decode_lut=decode_lut,
            gpu_moe_lut=base_dir / "gpu_moe.npz" if (base_dir / "gpu_moe.npz").exists() else None,
            tool_distributions=base_dir / "tool_distributions.pkl" if (base_dir / "tool_distributions.pkl").exists() else None,
        )
        self._lut_cache[cache_key] = bundle
        return bundle

    def simulate(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        workload_spec: WorkloadSpec,
    ) -> Dict[str, float]:
        """Run event-driven simulation and return metrics dict.

        Creates a VLLMScheduler and EventDrivenSimulator, runs for
        simulation_duration_s, then converts SimulationMetrics to a flat
        dict matching the RooflineOracle interface.
        """
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.scheduler.vllm import VLLMScheduler

        # Load LUT bundle (cached)
        lut_bundle = self._load_lut_bundle(model_spec, hardware_spec, inference_spec)

        # Create scheduler from inference_spec engine config
        max_num_seqs = inference_spec.engine_config.get("max_num_seqs", 256)
        max_num_batched_tokens = inference_spec.engine_config.get(
            "max_num_batched_tokens", 8192
        )
        enable_chunked_prefill = inference_spec.engine_config.get(
            "enable_chunked_prefill", True
        )
        scheduler = VLLMScheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
        )

        # Create and run simulator
        simulator = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
            lut_bundle=lut_bundle,
        )

        metrics = simulator.run(
            workload_spec=workload_spec,
            duration_s=self.simulation_duration_s,
            seed=42,
        )

        # Convert SimulationMetrics to dict matching RooflineOracle interface
        num_gpus = inference_spec.num_gpus
        total_time_s = self.simulation_duration_s

        # Energy per query
        energy_per_query = (
            metrics.total_energy_j / metrics.total_requests
            if metrics.total_requests > 0
            else float("inf")
        )

        # IPW and IPJ
        ipw = (
            self.accuracy_score / metrics.avg_power_w
            if metrics.avg_power_w > 0
            else 0.0
        )
        ipj = (
            self.accuracy_score / energy_per_query
            if energy_per_query > 0 and energy_per_query != float("inf")
            else 0.0
        )

        # Cost per query
        cost_per_query_usd = 0.0
        if self.price_per_hour_usd > 0 and metrics.total_requests > 0:
            total_cost = self.price_per_hour_usd * num_gpus * (total_time_s / 3600.0)
            cost_per_query_usd = total_cost / metrics.total_requests

        return {
            "ttft_s": metrics.ttft_p50,
            "tbt_s": metrics.tbt_p50,
            "e2e_latency_s": metrics.e2e_p50,
            "throughput_tps": metrics.throughput_tps,
            "throughput_rps": metrics.throughput_rps,
            "total_energy_j": metrics.total_energy_j,
            "avg_power_w": metrics.avg_power_w,
            "ipw": ipw,
            "ipj": ipj,
            "cost_per_query_usd": cost_per_query_usd,
            "energy_per_query_j": energy_per_query,
            "ttft_p50": metrics.ttft_p50,
            "ttft_p90": metrics.ttft_p90,
            "ttft_p95": metrics.ttft_p95,
            "ttft_p99": metrics.ttft_p99,
            "tbt_p50": metrics.tbt_p50,
            "tbt_p90": metrics.tbt_p90,
            "tbt_p95": metrics.tbt_p95,
            "tbt_p99": metrics.tbt_p99,
            "e2e_p50": metrics.e2e_p50,
            "e2e_p90": metrics.e2e_p90,
            "e2e_p95": metrics.e2e_p95,
            "e2e_p99": metrics.e2e_p99,
            "total_requests": float(metrics.total_requests),
            "total_tokens_generated": float(metrics.total_tokens_generated),
            "gpu_utilization": metrics.gpu_utilization,
            "total_prefill_time_s": metrics.total_prefill_time_s,
            "total_decode_time_s": metrics.total_decode_time_s,
            "total_tool_time_s": metrics.total_tool_time_s,
            "avg_num_steps": metrics.avg_num_steps,
        }


__all__ = ["EventDrivenOracle", "RooflineOracle", "SimulatorOracle"]
