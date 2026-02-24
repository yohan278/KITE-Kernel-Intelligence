"""Roofline-based runtime estimator wrapping ipw.simulator.inference_model."""

from __future__ import annotations

from typing import Any

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.types.operators import OperatorCategory
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec


class RooflineEstimator(BaseRuntimeEstimator):
    """Runtime estimator using the analytical roofline model.

    Wraps ipw.simulator.inference_model to provide per-operator estimates
    based on peak hardware throughput and model dimensions.

    This estimator does not require training data — it uses theoretical
    peak performance with efficiency factors (eta) as corrections.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        num_gpus: int = 1,
        eta_prefill: float = 0.4,
        eta_decode: float = 0.58,
        alpha: float = 0.65,
    ) -> None:
        self._model_spec = model_spec
        self._hardware_spec = hardware_spec
        self._num_gpus = num_gpus
        self._eta_prefill = eta_prefill
        self._eta_decode = eta_decode
        self._alpha = alpha

        # Compute derived hardware values
        self._effective_tflops = hardware_spec.peak_tflops * num_gpus
        self._effective_bw_gb_s = hardware_spec.hbm_bandwidth_gb_s * num_gpus
        self._power_watts = hardware_spec.tdp_watts * num_gpus * alpha
        self._active_params_b = model_spec.total_params_billion

        # For MoE models, use active params per token
        if (
            model_spec.num_experts is not None
            and model_spec.experts_per_token is not None
            and model_spec.num_experts > 0
        ):
            expert_fraction = model_spec.experts_per_token / model_spec.num_experts
            self._active_params_b *= expert_fraction

    def is_fitted(self) -> bool:
        """Roofline estimator is always ready (no training needed)."""
        return True

    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Estimate operator runtime using roofline model.

        Compute-bound operators (prefill, linear) use TFLOPS ceiling.
        Memory-bound operators (decode attention) use bandwidth ceiling.
        """
        if operator_category in (
            OperatorCategory.ATTENTION_PREFILL,
            OperatorCategory.LINEAR,
        ):
            return self._estimate_compute_bound(batch_size, seq_len, operator_category)
        elif operator_category == OperatorCategory.ATTENTION_DECODE:
            kv_cache_len = kwargs.get("kv_cache_len", seq_len)
            return self._estimate_memory_bound(batch_size, kv_cache_len)
        elif operator_category in (
            OperatorCategory.NORMALIZATION,
            OperatorCategory.ACTIVATION,
        ):
            return self._estimate_elementwise(batch_size, seq_len)
        elif operator_category == OperatorCategory.EMBEDDING:
            return self._estimate_embedding(batch_size, seq_len)
        elif operator_category in (
            OperatorCategory.MOE_ROUTING,
            OperatorCategory.MOE_EXPERT,
        ):
            return self._estimate_compute_bound(batch_size, seq_len, operator_category)
        elif operator_category == OperatorCategory.COMMUNICATION:
            return self._estimate_communication(batch_size, seq_len, **kwargs)
        else:
            # Default: treat as compute-bound linear
            return self._estimate_compute_bound(batch_size, seq_len, operator_category)

    def _estimate_compute_bound(
        self, batch_size: int, seq_len: int, category: OperatorCategory
    ) -> EstimatorResult:
        """Estimate for compute-bound ops (prefill attention, linear projections)."""
        h = self._model_spec.hidden_dim
        # FLOPs per token ≈ 2 * params (multiply-accumulate)
        # For a layer's worth of linear ops
        flops_per_token = 2.0 * self._active_params_b * 1e9 / self._model_spec.num_layers
        total_tokens = batch_size * seq_len
        total_flops = flops_per_token * total_tokens

        effective_flops_per_s = self._effective_tflops * 1e12 * self._eta_prefill
        if effective_flops_per_s <= 0:
            return EstimatorResult(time_s=0.0)

        time_s = total_flops / effective_flops_per_s
        energy_j = self._power_watts * time_s if self._power_watts > 0 else None
        power_w = self._power_watts if self._power_watts > 0 else None

        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=power_w)

    def _estimate_memory_bound(
        self, batch_size: int, kv_cache_len: int
    ) -> EstimatorResult:
        """Estimate for memory-bound ops (decode attention)."""
        # Per decode step: read all model weights
        weight_bytes = self._active_params_b * 1e9 * self._hardware_spec.bytes_per_param
        # Also read KV cache
        head_dim = self._model_spec.head_dim
        num_kv_heads = self._model_spec.num_kv_heads
        num_layers = self._model_spec.num_layers
        kv_bytes = 2 * num_layers * num_kv_heads * head_dim * kv_cache_len * self._hardware_spec.bytes_per_param
        total_bytes = (weight_bytes + kv_bytes) * batch_size

        effective_bw_bytes_s = self._effective_bw_gb_s * 1e9 * self._eta_decode
        if effective_bw_bytes_s <= 0:
            return EstimatorResult(time_s=0.0)

        time_s = total_bytes / effective_bw_bytes_s
        energy_j = self._power_watts * time_s if self._power_watts > 0 else None
        power_w = self._power_watts if self._power_watts > 0 else None

        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=power_w)

    def _estimate_elementwise(self, batch_size: int, seq_len: int) -> EstimatorResult:
        """Estimate for elementwise ops (norms, activations) — negligible."""
        h = self._model_spec.hidden_dim
        total_elements = batch_size * seq_len * h
        # ~5 FLOPs per element for RMSNorm, ~4 for SiLU
        flops = 5.0 * total_elements
        effective_flops_per_s = self._effective_tflops * 1e12 * self._eta_prefill
        if effective_flops_per_s <= 0:
            return EstimatorResult(time_s=0.0)
        time_s = flops / effective_flops_per_s
        energy_j = self._power_watts * time_s if self._power_watts > 0 else None
        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=self._power_watts or None)

    def _estimate_embedding(self, batch_size: int, seq_len: int) -> EstimatorResult:
        """Estimate for embedding lookup — memory-bound."""
        h = self._model_spec.hidden_dim
        bytes_read = batch_size * seq_len * h * self._hardware_spec.bytes_per_param
        effective_bw_bytes_s = self._effective_bw_gb_s * 1e9 * self._eta_decode
        if effective_bw_bytes_s <= 0:
            return EstimatorResult(time_s=0.0)
        time_s = bytes_read / effective_bw_bytes_s
        return EstimatorResult(time_s=time_s)

    def _estimate_communication(
        self, batch_size: int, seq_len: int, **kwargs: Any
    ) -> EstimatorResult:
        """Estimate AllReduce/AllGather communication time."""
        message_bytes = kwargs.get("message_bytes", 0)
        if message_bytes <= 0:
            h = self._model_spec.hidden_dim
            message_bytes = batch_size * seq_len * h * self._hardware_spec.bytes_per_param

        nvlink_bw = self._hardware_spec.nvlink_bandwidth_gb_s
        if nvlink_bw <= 0:
            # No interconnect info, assume PCIe ~32 GB/s
            nvlink_bw = 32.0

        # Ring AllReduce: 2*(n-1)/n * message_size / bandwidth
        n_gpus = self._num_gpus
        if n_gpus <= 1:
            return EstimatorResult(time_s=0.0)

        effective_bw_bytes_s = nvlink_bw * 1e9 * 0.85  # 85% efficiency
        ring_factor = 2.0 * (n_gpus - 1) / n_gpus
        time_s = ring_factor * message_bytes / effective_bw_bytes_s
        energy_j = self._power_watts * time_s if self._power_watts > 0 else None

        return EstimatorResult(time_s=time_s, energy_j=energy_j, power_w=self._power_watts or None)

    def estimate_prefill(
        self,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Estimate full prefill phase using roofline model directly."""
        from ipw.simulator.inference_model import estimate_prefill, estimate_power

        power_w = estimate_power(
            type("HW", (), {"tdp_watts": self._hardware_spec.tdp_watts * self._num_gpus})(),
            self._alpha,
        )
        result = estimate_prefill(
            self._active_params_b,
            batch_size * seq_len,
            self._effective_tflops,
            self._eta_prefill,
            power_w,
        )
        return EstimatorResult(
            time_s=result.time_seconds,
            energy_j=result.energy_joules if result.energy_joules > 0 else None,
            power_w=power_w if power_w > 0 else None,
        )

    def estimate_decode_step(
        self,
        batch_size: int,
        kv_cache_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Estimate single decode step using roofline model directly."""
        from ipw.simulator.inference_model import estimate_decode, estimate_power

        power_w = estimate_power(
            type("HW", (), {"tdp_watts": self._hardware_spec.tdp_watts * self._num_gpus})(),
            self._alpha,
        )
        result = estimate_decode(
            self._active_params_b,
            batch_size,  # 1 token per request in batch
            self._hardware_spec.bytes_per_param,
            self._effective_bw_gb_s,
            self._eta_decode,
            power_w,
        )
        return EstimatorResult(
            time_s=result.time_seconds,
            energy_j=result.energy_joules if result.energy_joules > 0 else None,
            power_w=power_w if power_w > 0 else None,
        )
