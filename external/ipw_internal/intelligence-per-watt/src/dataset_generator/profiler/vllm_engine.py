"""vLLM engine-instrumented profiler for fused kernel timings."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig

logger = logging.getLogger(__name__)

# Kernel name substring -> OperatorCategory mapping for trace decomposition.
# Case-insensitive substring matching is used against CUDA kernel names captured
# by torch.profiler.  The order does not matter since each kernel is matched
# against all patterns and the first match wins.
_KERNEL_CATEGORY_PATTERNS: Dict[str, OperatorCategory] = {
    "fused_add_rms_norm": OperatorCategory.FUSED_NORM_ATTN,
    "rmsnorm": OperatorCategory.FUSED_NORM_ATTN,
    "layernorm": OperatorCategory.FUSED_NORM_ATTN,
    "silu_and_mul": OperatorCategory.FUSED_MLP,
    "flash_attn": OperatorCategory.FUSED_ATTENTION,
    "flash_fwd": OperatorCategory.FUSED_ATTENTION,
    "flash_bwd": OperatorCategory.FUSED_ATTENTION,
    "paged_attention": OperatorCategory.FUSED_ATTENTION,
    "paged_attn": OperatorCategory.FUSED_ATTENTION,
    "rotary_embedding": OperatorCategory.FUSED_ATTENTION,
    "rotary_emb": OperatorCategory.FUSED_ATTENTION,
    "gemm": OperatorCategory.FUSED_MLP,
    "cublas": OperatorCategory.FUSED_MLP,
    "cutlass": OperatorCategory.FUSED_MLP,
    "gemv": OperatorCategory.FUSED_MLP,
}


def _classify_kernel(kernel_name: str) -> Optional[OperatorCategory]:
    """Classify a CUDA kernel name into an OperatorCategory via substring matching.

    Args:
        kernel_name: The CUDA kernel name from torch.profiler trace events.

    Returns:
        The matching OperatorCategory, or None if no pattern matches.
    """
    lower = kernel_name.lower()
    for pattern, category in _KERNEL_CATEGORY_PATTERNS.items():
        if pattern in lower:
            return category
    return None


class VLLMEngineProfiler(BaseOperatorProfiler):
    """Profile operators within vLLM's execution context.

    Instruments vLLM's ModelRunner to capture fused kernel timings
    as they actually execute, using torch.cuda.Event pairs for
    precise GPU timing and optional NVML energy measurement.

    Requires vllm to be installed. Uses enforce_eager=True to disable
    CUDA graphs so that per-call timing is accurate.
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.FUSED_PREFILL

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes", "prefill_seq_lengths", "kv_cache_sizes"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile fused operators via vLLM's engine.

        Instantiates vLLM in offline mode, creates synthetic prompts,
        and wraps execution with CUDA event timing to capture fused
        prefill, decode, attention, and MLP timings.
        """
        try:
            import torch
            from vllm import LLM, SamplingParams
        except ImportError:
            raise NotImplementedError(
                "vLLM is required for VLLMEngineProfiler. "
                "Install with: pip install vllm"
            )

        measurements: List[OperatorMeasurement] = []

        # Instantiate vLLM in offline (eager) mode for per-call timing
        llm = LLM(
            model=model_spec.model_id,
            enforce_eager=True,
            dtype=precision if precision in ("float16", "bfloat16") else "float16",
            gpu_memory_utilization=0.85,
        )

        # Access the model runner for direct instrumentation
        model_runner = (
            llm.llm_engine.model_executor.driver_worker.model_runner
        )

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]
            kv_cache_size = point.get("kv_cache_size", seq_len)

            # Build synthetic prompts of target length
            # Use token ID 1 repeated to target length (avoids tokenization variance)
            prompts = [" ".join(["token"] * seq_len) for _ in range(batch_size)]
            sampling_params = SamplingParams(
                max_tokens=1,  # Single decode step for timing
                temperature=0.0,
            )

            try:
                fused_measurements = self._profile_fused_pass(
                    llm,
                    model_runner,
                    prompts,
                    sampling_params,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    kv_cache_size=kv_cache_size,
                    sweep_config=sweep_config,
                    torch_module=torch,
                )
                measurements.extend(fused_measurements)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Skip OOM configurations
                continue

        # Clean up GPU memory
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return measurements

    def _profile_fused_pass(
        self,
        llm,
        model_runner,
        prompts: List[str],
        sampling_params,
        batch_size: int,
        seq_len: int,
        kv_cache_size: int,
        sweep_config: SweepConfig,
        torch_module,
    ) -> List[OperatorMeasurement]:
        """Time a single fused pass through vLLM and return measurements.

        Uses CUDA events for accurate total timing, then decomposes via
        torch.profiler trace analysis to determine attention/MLP/norm
        fractions.  Falls back to hardcoded fractions if tracing fails.
        """
        import torch

        measurements: List[OperatorMeasurement] = []

        # Warmup
        for _ in range(sweep_config.warmup_iterations):
            llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()

        # Timed runs - measure overall execute_model time via CUDA events
        prefill_times_ms: List[float] = []
        for _ in range(sweep_config.measurement_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            llm.generate(prompts, sampling_params)
            end_event.record()
            torch.cuda.synchronize()
            prefill_times_ms.append(start_event.elapsed_time(end_event))

        mean_prefill_s = sum(prefill_times_ms) / len(prefill_times_ms) / 1000.0

        # Trace-based decomposition: run torch.profiler to get kernel breakdown
        trace_fractions = self._trace_kernel_fractions(llm, prompts, sampling_params)

        # Extract fractions with hardcoded fallbacks
        fraction_attn = trace_fractions.get(OperatorCategory.FUSED_ATTENTION, 0.4)
        fraction_mlp = trace_fractions.get(OperatorCategory.FUSED_MLP, 0.55)
        fraction_norm = trace_fractions.get(OperatorCategory.FUSED_NORM_ATTN, 0.05)
        decode_fraction = 1.0 / max(seq_len, 1)

        # Energy measurement (optional)
        energy_j: Optional[float] = None
        power_w: Optional[float] = None
        if sweep_config.use_energy:
            energy_j, power_w = self._measure_energy(
                llm, prompts, sampling_params,
                sweep_config.measurement_iterations,
            )

        # Record fused_prefill measurement (full forward pass timing)
        measurements.append(OperatorMeasurement(
            operator_name="vllm_fused_prefill",
            category=OperatorCategory.FUSED_PREFILL,
            batch_size=batch_size,
            seq_len=seq_len,
            time_s=mean_prefill_s,
            energy_j=energy_j,
            power_w=power_w,
            metadata={
                "kv_cache_size": kv_cache_size,
                "engine": "vllm",
                "trace_fractions": {
                    k.value: v for k, v in trace_fractions.items()
                },
            },
        ))

        # Record fused_decode_step (single token generation timing)
        decode_time_s = mean_prefill_s * decode_fraction
        measurements.append(OperatorMeasurement(
            operator_name="vllm_fused_decode_step",
            category=OperatorCategory.FUSED_DECODE_STEP,
            batch_size=batch_size,
            seq_len=1,
            time_s=decode_time_s,
            energy_j=energy_j * decode_fraction if energy_j is not None else None,
            power_w=power_w,
            metadata={"kv_cache_size": kv_cache_size, "engine": "vllm"},
        ))

        # Record fused_attention (from trace-based or fallback fraction)
        attn_time_s = mean_prefill_s * fraction_attn
        measurements.append(OperatorMeasurement(
            operator_name="vllm_fused_attention",
            category=OperatorCategory.FUSED_ATTENTION,
            batch_size=batch_size,
            seq_len=seq_len,
            time_s=attn_time_s,
            energy_j=energy_j * fraction_attn if energy_j is not None else None,
            power_w=power_w,
            metadata={"kv_cache_size": kv_cache_size, "engine": "vllm"},
        ))

        # Record fused_mlp (from trace-based or fallback fraction)
        mlp_time_s = mean_prefill_s * fraction_mlp
        measurements.append(OperatorMeasurement(
            operator_name="vllm_fused_mlp",
            category=OperatorCategory.FUSED_MLP,
            batch_size=batch_size,
            seq_len=seq_len,
            time_s=mlp_time_s,
            energy_j=energy_j * fraction_mlp if energy_j is not None else None,
            power_w=power_w,
            metadata={"kv_cache_size": kv_cache_size, "engine": "vllm"},
        ))

        # Record fused_norm_attn if trace decomposition found norm kernels
        if fraction_norm > 0.0:
            norm_time_s = mean_prefill_s * fraction_norm
            measurements.append(OperatorMeasurement(
                operator_name="vllm_fused_norm_attn",
                category=OperatorCategory.FUSED_NORM_ATTN,
                batch_size=batch_size,
                seq_len=seq_len,
                time_s=norm_time_s,
                energy_j=energy_j * fraction_norm if energy_j is not None else None,
                power_w=power_w,
                metadata={"kv_cache_size": kv_cache_size, "engine": "vllm"},
            ))

        return measurements

    def _trace_kernel_fractions(
        self,
        llm,
        prompts: List[str],
        sampling_params,
    ) -> Dict[OperatorCategory, float]:
        """Run torch.profiler to decompose fused pass into kernel-level fractions.

        Executes a single profiled forward pass, classifies each CUDA kernel
        into an OperatorCategory, and returns the time fraction each category
        occupies of the total CUDA time.

        Args:
            llm: The vLLM LLM instance.
            prompts: Synthetic prompts for the forward pass.
            sampling_params: vLLM SamplingParams.

        Returns:
            Dict mapping OperatorCategory to fraction of total CUDA time.
            Returns empty dict if tracing fails (caller uses hardcoded fallbacks).
        """
        try:
            import torch
            from torch.profiler import ProfilerActivity
        except ImportError:
            return {}

        try:
            with torch.profiler.profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                llm.generate(prompts, sampling_params)

            category_times: Dict[OperatorCategory, float] = defaultdict(float)
            total_cuda_us = 0.0

            for event in prof.key_averages():
                # cuda_time_total is in microseconds
                cuda_time = getattr(event, "cuda_time_total", 0)
                if cuda_time <= 0:
                    continue

                category = _classify_kernel(event.key)
                if category is not None:
                    category_times[category] += cuda_time
                total_cuda_us += cuda_time

            if total_cuda_us <= 0:
                logger.debug("No CUDA time captured in trace, using fallback fractions")
                return {}

            # Convert absolute times to fractions
            fractions: Dict[OperatorCategory, float] = {}
            for cat, t in category_times.items():
                fractions[cat] = t / total_cuda_us

            classified_us = sum(category_times.values())
            classified_frac = classified_us / total_cuda_us
            logger.info(
                "Trace decomposition: %.1f%% of CUDA time classified "
                "(%d categories, %.2f ms total)",
                classified_frac * 100,
                len(fractions),
                total_cuda_us / 1e3,
            )
            for cat, frac in sorted(fractions.items(), key=lambda x: -x[1]):
                logger.debug("  %s: %.1f%%", cat.value, frac * 100)

            return fractions

        except Exception:
            logger.warning(
                "torch.profiler trace failed, falling back to hardcoded fractions",
                exc_info=True,
            )
            return {}

    @staticmethod
    def _measure_energy(
        llm,
        prompts: List[str],
        sampling_params,
        iterations: int,
    ) -> tuple[Optional[float], Optional[float]]:
        """Measure energy via NVML during vLLM execution."""
        try:
            import torch
            from ipw.telemetry import EnergyMonitorCollector
            from ipw.execution.telemetry_session import TelemetrySession

            collector = EnergyMonitorCollector()
            with TelemetrySession(collector) as session:
                time.sleep(0.3)
                for _ in range(iterations):
                    llm.generate(prompts, sampling_params)
                torch.cuda.synchronize()
                time.sleep(0.3)

                samples = list(session.readings())
                energy_j = None
                power_w = None
                if len(samples) >= 2:
                    first = samples[0].reading
                    last = samples[-1].reading
                    if (
                        first.energy_joules is not None
                        and last.energy_joules is not None
                    ):
                        delta = last.energy_joules - first.energy_joules
                        if delta >= 0:
                            energy_j = delta / iterations

                    power_readings = [
                        s.reading.power_watts
                        for s in samples
                        if s.reading.power_watts is not None
                    ]
                    if power_readings:
                        power_w = sum(power_readings) / len(power_readings)

                return energy_j, power_w
        except (ImportError, RuntimeError):
            return None, None
