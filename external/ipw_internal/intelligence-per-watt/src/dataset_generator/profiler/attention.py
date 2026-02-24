"""Attention operator profiler for prefill and decode phases."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class AttentionProfiler(BaseOperatorProfiler):
    """Profiles attention operators in both prefill and decode phases.

    Prefill: Full sequence attention (compute-bound).
    Decode: Single-token attending to KV cache (memory-bound).
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.ATTENTION_PREFILL

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes", "prefill_seq_lengths"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile both prefill and decode attention."""
        measurements: List[OperatorMeasurement] = []
        measurements.extend(self._profile_prefill(model_spec, hw_spec, sweep_config, precision))
        measurements.extend(self._profile_decode(model_spec, hw_spec, sweep_config, precision))
        measurements.extend(self._profile_kv_cache_ops(model_spec, hw_spec, sweep_config, precision))
        measurements.extend(self._profile_attention_variants(model_spec, hw_spec, sweep_config, precision))
        return measurements

    def _profile_prefill(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile prefill attention across batch_size x seq_len."""
        import torch
        import torch.nn.functional as F

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        nh = model_spec.num_attention_heads
        nkv = model_spec.num_kv_heads
        hd = model_spec.head_dim
        measurements = []

        for point in sweep_config.get_sweep_points(["batch_sizes", "prefill_seq_lengths"]):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]

            try:
                # GQA: expand KV heads to match Q heads
                q = torch.randn(batch_size, nh, seq_len, hd, device=device, dtype=dtype)
                k = torch.randn(batch_size, nkv, seq_len, hd, device=device, dtype=dtype)
                v = torch.randn(batch_size, nkv, seq_len, hd, device=device, dtype=dtype)

                # Expand KV for GQA
                if nkv != nh:
                    repeats = nh // nkv
                    k = k.repeat_interleave(repeats, dim=1)
                    v = v.repeat_interleave(repeats, dim=1)

                def prefill_fn(_q=q, _k=k, _v=v):
                    return F.scaled_dot_product_attention(_q, _k, _v, is_causal=True)

                # FLOPs: 2 * batch * num_heads * seq_len^2 * head_dim (QK^T + AV)
                flops = 2 * 2 * batch_size * nh * seq_len * seq_len * hd

                measurement = harness.measure(
                    prefill_fn,
                    operator_name="attention_prefill",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    flops=flops,
                )
                measurements.append(measurement)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                continue

        return measurements

    def _profile_decode(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile decode attention: single query token attending to KV cache."""
        import torch
        import torch.nn.functional as F

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        nh = model_spec.num_attention_heads
        nkv = model_spec.num_kv_heads
        hd = model_spec.head_dim
        measurements = []

        for point in sweep_config.get_sweep_points(["batch_sizes", "kv_cache_sizes"]):
            batch_size = point["batch_size"]
            kv_cache_size = point["kv_cache_size"]

            try:
                # Query: single token
                q = torch.randn(batch_size, nh, 1, hd, device=device, dtype=dtype)
                k = torch.randn(batch_size, nkv, kv_cache_size, hd, device=device, dtype=dtype)
                v = torch.randn(batch_size, nkv, kv_cache_size, hd, device=device, dtype=dtype)

                # Expand KV for GQA
                if nkv != nh:
                    repeats = nh // nkv
                    k = k.repeat_interleave(repeats, dim=1)
                    v = v.repeat_interleave(repeats, dim=1)

                def decode_fn(_q=q, _k=k, _v=v):
                    return F.scaled_dot_product_attention(_q, _k, _v, is_causal=False)

                # FLOPs: 2 * batch * num_heads * 1 * kv_cache_size * head_dim (QK^T + AV)
                flops = 2 * 2 * batch_size * nh * 1 * kv_cache_size * hd
                # Bytes: reading KV cache
                bytes_accessed = 2 * batch_size * nh * kv_cache_size * hd * 2  # fp16

                measurement = harness.measure(
                    decode_fn,
                    operator_name="attention_decode",
                    category=OperatorCategory.ATTENTION_DECODE,
                    batch_size=batch_size,
                    seq_len=kv_cache_size,
                    flops=flops,
                    bytes_accessed=bytes_accessed,
                )
                measurement.metadata["kv_cache_size"] = kv_cache_size
                measurements.append(measurement)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                continue

        return measurements

    def _profile_kv_cache_ops(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile KV cache append and evict operations."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        nkv = model_spec.num_kv_heads
        hd = model_spec.head_dim
        measurements = []

        for point in sweep_config.get_sweep_points(["batch_sizes", "kv_cache_sizes"]):
            batch_size = point["batch_size"]
            cache_len = point["kv_cache_size"]

            try:
                # --- kv_cache_append ---
                max_cache = cache_len + 1
                buffer = torch.zeros(batch_size, nkv, max_cache, hd, device=device, dtype=dtype)
                new_kv = torch.randn(batch_size, nkv, 1, hd, device=device, dtype=dtype)
                insert_idx = cache_len

                def append_fn(_buf=buffer, _new=new_kv, _idx=insert_idx):
                    _buf[:, :, _idx:_idx + 1, :] = _new
                    return _buf

                bytes_accessed_append = batch_size * nkv * 1 * hd * 2  # fp16

                measurement = harness.measure(
                    append_fn,
                    operator_name="kv_cache_append",
                    category=OperatorCategory.KV_CACHE,
                    batch_size=batch_size,
                    seq_len=cache_len,
                    flops=0,
                    bytes_accessed=bytes_accessed_append,
                )
                measurement.metadata["kv_cache_size"] = cache_len
                measurements.append(measurement)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                continue

            try:
                # --- kv_cache_evict ---
                cache = torch.randn(batch_size, nkv, cache_len, hd, device=device, dtype=dtype)
                evict_n = max(1, cache_len // 8)

                def evict_fn(_cache=cache, _n=evict_n):
                    return _cache[:, :, _n:, :].contiguous()

                bytes_accessed_evict = batch_size * nkv * cache_len * hd * 2

                measurement = harness.measure(
                    evict_fn,
                    operator_name="kv_cache_evict",
                    category=OperatorCategory.KV_CACHE,
                    batch_size=batch_size,
                    seq_len=cache_len,
                    flops=0,
                    bytes_accessed=bytes_accessed_evict,
                )
                measurement.metadata["kv_cache_size"] = cache_len
                measurement.metadata["evict_count"] = evict_n
                measurements.append(measurement)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                continue

        return measurements

    def _profile_attention_variants(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile sliding window attention and MQA/GQA head expansion."""
        import torch
        import torch.nn.functional as F

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        nh = model_spec.num_attention_heads
        nkv = model_spec.num_kv_heads
        hd = model_spec.head_dim
        measurements = []

        for point in sweep_config.get_sweep_points(["batch_sizes", "prefill_seq_lengths"]):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]

            try:
                # --- sliding_window_attention ---
                window_size = min(4096, seq_len)
                q = torch.randn(batch_size, nh, seq_len, hd, device=device, dtype=dtype)
                k = torch.randn(batch_size, nh, seq_len, hd, device=device, dtype=dtype)
                v = torch.randn(batch_size, nh, seq_len, hd, device=device, dtype=dtype)

                # Create sliding window mask
                row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
                mask = ((row_idx - col_idx).abs() <= window_size) & (col_idx <= row_idx)
                attn_mask = mask.float().unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(attn_mask == 1, 0.0)
                attn_mask = attn_mask.to(dtype=dtype)

                def swa_fn(_q=q, _k=k, _v=v, _mask=attn_mask):
                    return F.scaled_dot_product_attention(_q, _k, _v, attn_mask=_mask)

                flops_swa = 2 * 2 * batch_size * nh * seq_len * seq_len * hd

                measurement = harness.measure(
                    swa_fn,
                    operator_name="sliding_window_attention",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    flops=flops_swa,
                )
                measurement.metadata["window_size"] = window_size
                measurements.append(measurement)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                continue

            try:
                # --- mqa_gqa_expansion ---
                k_unexpanded = torch.randn(batch_size, nkv, seq_len, hd, device=device, dtype=dtype)
                repeats = nh // nkv if nkv != nh else 1

                def expand_fn(_k=k_unexpanded, _r=repeats):
                    return _k.repeat_interleave(_r, dim=1)

                flops_expand = batch_size * nh * seq_len * hd  # memory copy

                measurement = harness.measure(
                    expand_fn,
                    operator_name="mqa_gqa_expansion",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    flops=flops_expand,
                )
                measurement.metadata["num_kv_heads"] = nkv
                measurement.metadata["num_q_heads"] = nh
                measurement.metadata["repeat_factor"] = repeats
                measurements.append(measurement)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                continue

        return measurements
