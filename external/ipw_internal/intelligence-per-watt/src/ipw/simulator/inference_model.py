"""Roofline-based single-inference latency and energy predictor.

Implements the hybrid analytical model from the design plan:

  Prefill (compute-bound):
    FLOPs = 2 * active_params * input_tokens
    time  = FLOPs / (peak_tflops * eta_prefill)

  Decode (memory-bandwidth-bound):
    bytes_read = 2 * active_params * bytes_per_param  (per token)
    time_per_token = bytes_read / (mem_bw * eta_decode)
    total_decode_time = output_tokens * time_per_token

  Power:
    P = alpha * TDP
    E = P * time

When calibration factors are provided, eta_prefill, eta_decode, and alpha
are fitted values from E2E traces. Otherwise, conservative defaults are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ipw.simulator.hardware_specs import HardwareSpecs
from ipw.simulator.types import (
    CalibrationFactors,
    PhaseResult,
    SingleInferenceResult,
)

# Default efficiency assumptions when no calibration data is available.
# These are conservative lower bounds from the literature.
DEFAULT_ETA_PREFILL = 0.4   # 40% of peak TFLOPS utilization during prefill
DEFAULT_ETA_DECODE = 0.5    # 50% of peak memory bandwidth utilization during decode
DEFAULT_ALPHA = 0.65        # 65% of TDP as average power draw

# Log-linear fits from A100 SXM4, Qwen3 (0.6B/4B/14B), B=1, FP16:
#   η ≈ 0.161 × ln(params_b) + 0.266
#   α ≈ 0.148 × ln(params_b) + 0.562
_ETA_LOG_SLOPE = 0.161
_ETA_LOG_INTERCEPT = 0.266
_ALPHA_LOG_SLOPE = 0.148
_ALPHA_LOG_INTERCEPT = 0.562

# Bottom-up pJ energy constants
PJ_PER_FLOP_MATMUL_FP16 = 0.70      # pJ per tensor-core FLOP at FP16
PJ_PER_BIT_HBM = 13.11              # pJ per bit transferred over HBM
SYSTEM_OVERHEAD_MULTIPLIER = 2.0     # fitted from A100 empirical data

# Batch-dependent overhead constants (Model F).
# overhead(B) = OVERHEAD_BASE + OVERHEAD_DECAY / B^OVERHEAD_BETA
# Fitted from 180 A100 empirical configs (3 models × 60 configs) via MAPE minimization.
# At B=1: overhead ≈ 2.18, at B>1: rapidly converges to ~1.28.
OVERHEAD_BASE = 1.2834   # irreducible overhead at large batch
OVERHEAD_DECAY = 0.8927  # amortizable scheduling overhead magnitude
OVERHEAD_BETA = 16.5690  # decay exponent (sharp step-like transition)


def estimate_eta_for_model_size(active_params_b: float) -> float:
    """Estimate MFU (η) from model size using a log-linear fit.

    Fitted from A100 SXM4, Qwen3 (0.6B/4B/14B), B=1, FP16.
    Larger models achieve higher compute utilization because they
    have more FLOPs per kernel launch overhead.

    Args:
        active_params_b: Active parameters in billions.

    Returns:
        Estimated η, clamped to [0.05, 0.95].
    """
    if active_params_b <= 0:
        return DEFAULT_ETA_PREFILL
    eta = _ETA_LOG_SLOPE * math.log(active_params_b) + _ETA_LOG_INTERCEPT
    return max(0.05, min(0.95, eta))


def estimate_alpha_for_model_size(active_params_b: float) -> float:
    """Estimate power fraction (α) from model size using a log-linear fit.

    Fitted from A100 SXM4, Qwen3 (0.6B/4B/14B), B=1, FP16.
    Larger models tend to sustain higher power draw because they
    keep more compute units active.

    Args:
        active_params_b: Active parameters in billions.

    Returns:
        Estimated α, clamped to [0.1, 1.0].
    """
    if active_params_b <= 0:
        return DEFAULT_ALPHA
    alpha = _ALPHA_LOG_SLOPE * math.log(active_params_b) + _ALPHA_LOG_INTERCEPT
    return max(0.1, min(1.0, alpha))


@dataclass(slots=True)
class BottomUpEnergyBreakdown:
    """Breakdown of prefill energy from a bottom-up pJ model.

    Five energy components plus a system overhead multiplier.
    """

    e_matmul_j: float
    e_attention_j: float
    e_weight_memory_j: float
    e_activation_memory_j: float
    e_kv_cache_j: float
    e_raw_total_j: float
    overhead_multiplier: float
    e_total_j: float


def batch_overhead_multiplier(
    batch_size: int,
    *,
    c_base: float = OVERHEAD_BASE,
    c_decay: float = OVERHEAD_DECAY,
    beta: float = OVERHEAD_BETA,
) -> float:
    """Compute batch-dependent overhead multiplier.

    overhead(B) = c_base + c_decay / B^beta

    At B=1, this equals c_base + c_decay (close to the static 2.0× for
    default constants). At large B, it converges to c_base.

    Args:
        batch_size: Batch size (clamped to >= 1).
        c_base: Irreducible system overhead at large batch.
        c_decay: Amortizable scheduling overhead magnitude.
        beta: Decay exponent.

    Returns:
        Overhead multiplier (always positive).
    """
    b = max(batch_size, 1)
    return c_base + c_decay / (b ** beta)


def estimate_prefill_batch_aware(
    active_params_b: float,
    input_tokens: int,
    batch_size: int = 1,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    hidden_size: int = 4096,
    *,
    c_base: float = OVERHEAD_BASE,
    c_decay: float = OVERHEAD_DECAY,
    beta: float = OVERHEAD_BETA,
) -> BottomUpEnergyBreakdown:
    """Estimate per-query prefill energy with batch-dependent overhead (Model F).

    Same five energy components as estimate_prefill_bottomup(), but:
      - Weight memory is amortized across the batch (divided by batch_size).
      - Overhead multiplier depends on batch size via batch_overhead_multiplier().

    Args:
        active_params_b: Active parameters in billions.
        input_tokens: Number of input tokens (sequence length).
        batch_size: Batch size (>= 1).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA).
        head_dim: Dimension per attention head.
        hidden_size: Hidden dimension of the model.
        c_base: Irreducible overhead at large batch.
        c_decay: Amortizable overhead magnitude.
        beta: Decay exponent.

    Returns:
        BottomUpEnergyBreakdown with batch-aware overhead.
    """
    batch_size = max(batch_size, 1)
    overhead = batch_overhead_multiplier(batch_size, c_base=c_base, c_decay=c_decay, beta=beta)

    if input_tokens <= 0 or active_params_b <= 0:
        return BottomUpEnergyBreakdown(
            e_matmul_j=0.0,
            e_attention_j=0.0,
            e_weight_memory_j=0.0,
            e_activation_memory_j=0.0,
            e_kv_cache_j=0.0,
            e_raw_total_j=0.0,
            overhead_multiplier=overhead,
            e_total_j=0.0,
        )

    params = active_params_b * 1e9
    sin = input_tokens
    pj_flop = PJ_PER_FLOP_MATMUL_FP16 * 1e-12   # joules per FLOP
    pj_bit = PJ_PER_BIT_HBM * 1e-12              # joules per bit

    # 1. Matmul FLOPs: 2 × params × sin (per query)
    e_matmul = 2.0 * params * sin * pj_flop

    # 2. Attention FLOPs: L × n_heads × sin² × head_dim × 2 (per query)
    e_attention = num_layers * num_heads * (sin ** 2) * head_dim * 2.0 * pj_flop

    # 3. Weight memory: one HBM read, amortized across batch
    e_weight_memory = params * 2.0 * 8.0 * pj_bit / batch_size

    # 4. Activation memory: L × sin × hidden × 2 bytes × 8 bits (per query)
    e_activation_memory = num_layers * sin * hidden_size * 2.0 * 8.0 * pj_bit

    # 5. KV cache writes: L × sin × 2 (K+V) × n_kv × head_dim × 2 bytes × 8 bits
    e_kv_cache = num_layers * sin * 2.0 * num_kv_heads * head_dim * 2.0 * 8.0 * pj_bit

    e_raw = e_matmul + e_attention + e_weight_memory + e_activation_memory + e_kv_cache
    e_total = e_raw * overhead

    return BottomUpEnergyBreakdown(
        e_matmul_j=e_matmul,
        e_attention_j=e_attention,
        e_weight_memory_j=e_weight_memory,
        e_activation_memory_j=e_activation_memory,
        e_kv_cache_j=e_kv_cache,
        e_raw_total_j=e_raw,
        overhead_multiplier=overhead,
        e_total_j=e_total,
    )


def estimate_prefill_bottomup(
    active_params_b: float,
    input_tokens: int,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    hidden_size: int = 4096,
    overhead_multiplier: float = SYSTEM_OVERHEAD_MULTIPLIER,
) -> BottomUpEnergyBreakdown:
    """Estimate prefill energy using a bottom-up pJ component model.

    Accounts for five energy sources:
      1. Matmul FLOPs (2 × params × sin)
      2. Attention FLOPs (O(sin²) self-attention)
      3. Weight memory (one HBM read of all parameters)
      4. Activation memory (per-layer intermediate activations)
      5. KV cache writes (keys + values for each layer)

    The raw total is multiplied by an overhead factor to account for
    scheduling, memory management, and other system-level costs not
    captured by the component model.

    Args:
        active_params_b: Active parameters in billions.
        input_tokens: Number of input tokens (sequence length).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA).
        head_dim: Dimension per attention head.
        hidden_size: Hidden dimension of the model.
        overhead_multiplier: System overhead factor (default 2.0).

    Returns:
        BottomUpEnergyBreakdown with all components.
    """
    if input_tokens <= 0 or active_params_b <= 0:
        return BottomUpEnergyBreakdown(
            e_matmul_j=0.0,
            e_attention_j=0.0,
            e_weight_memory_j=0.0,
            e_activation_memory_j=0.0,
            e_kv_cache_j=0.0,
            e_raw_total_j=0.0,
            overhead_multiplier=overhead_multiplier,
            e_total_j=0.0,
        )

    params = active_params_b * 1e9
    sin = input_tokens
    pj_flop = PJ_PER_FLOP_MATMUL_FP16 * 1e-12   # joules per FLOP
    pj_bit = PJ_PER_BIT_HBM * 1e-12              # joules per bit

    # 1. Matmul FLOPs: 2 × params × sin
    e_matmul = 2.0 * params * sin * pj_flop

    # 2. Attention FLOPs: L × n_heads × sin² × head_dim × 2 (QK + AV)
    e_attention = num_layers * num_heads * (sin ** 2) * head_dim * 2.0 * pj_flop

    # 3. Weight memory: one HBM read of all params (FP16 = 2 bytes)
    e_weight_memory = params * 2.0 * 8.0 * pj_bit

    # 4. Activation memory: L × sin × hidden × 2 bytes × 8 bits
    e_activation_memory = num_layers * sin * hidden_size * 2.0 * 8.0 * pj_bit

    # 5. KV cache writes: L × sin × 2 (K+V) × n_kv × head_dim × 2 bytes × 8 bits
    e_kv_cache = num_layers * sin * 2.0 * num_kv_heads * head_dim * 2.0 * 8.0 * pj_bit

    e_raw = e_matmul + e_attention + e_weight_memory + e_activation_memory + e_kv_cache
    e_total = e_raw * overhead_multiplier

    return BottomUpEnergyBreakdown(
        e_matmul_j=e_matmul,
        e_attention_j=e_attention,
        e_weight_memory_j=e_weight_memory,
        e_activation_memory_j=e_activation_memory,
        e_kv_cache_j=e_kv_cache,
        e_raw_total_j=e_raw,
        overhead_multiplier=overhead_multiplier,
        e_total_j=e_total,
    )


def estimate_prefill(
    active_params_b: float,
    input_tokens: int,
    peak_tflops: float,
    eta: float = DEFAULT_ETA_PREFILL,
    power_watts: float = 0.0,
) -> PhaseResult:
    """Estimate prefill phase latency and energy.

    Prefill processes the entire input context in one forward pass.
    It is compute-bound: FLOPs = 2 * P * S_in.

    Args:
        active_params_b: Active parameters in billions (for MoE, this is the
            number of parameters activated per token, not total).
        input_tokens: Number of input tokens to process.
        peak_tflops: Peak tensor-core TFLOPS of the hardware.
        eta: Compute efficiency factor (actual MFU / peak).
        power_watts: Average power draw during prefill.

    Returns:
        PhaseResult with time and energy estimates.
    """
    if input_tokens <= 0 or active_params_b <= 0 or peak_tflops <= 0:
        return PhaseResult()

    # FLOPs for a single forward pass over input_tokens
    # Each token requires ~2 FLOPs per parameter (multiply-accumulate)
    flops = 2.0 * active_params_b * 1e9 * input_tokens

    # Effective compute throughput (FLOP/s)
    effective_tflops = peak_tflops * max(eta, 0.01)
    effective_flops_per_s = effective_tflops * 1e12

    time_s = flops / effective_flops_per_s
    energy_j = power_watts * time_s if power_watts > 0 else 0.0

    return PhaseResult(
        time_seconds=time_s,
        energy_joules=energy_j,
        flops=flops,
        bytes_transferred=0.0,
    )


def estimate_decode(
    active_params_b: float,
    output_tokens: int,
    bytes_per_param: float,
    mem_bw_gb_s: float,
    eta: float = DEFAULT_ETA_DECODE,
    power_watts: float = 0.0,
) -> PhaseResult:
    """Estimate decode phase latency and energy.

    Decode generates one token at a time. Each token requires reading
    all model weights from memory, making it memory-bandwidth-bound.

    Args:
        active_params_b: Active parameters in billions.
        output_tokens: Number of tokens to generate.
        bytes_per_param: Bytes per parameter (2 for FP16, 1 for FP8).
        mem_bw_gb_s: Peak HBM bandwidth in GB/s.
        eta: Memory bandwidth efficiency factor (actual MBU / peak).
        power_watts: Average power draw during decode.

    Returns:
        PhaseResult with time and energy estimates.
    """
    if output_tokens <= 0 or active_params_b <= 0 or mem_bw_gb_s <= 0:
        return PhaseResult()

    # Bytes read per token = all model weights
    weight_bytes = active_params_b * 1e9 * bytes_per_param

    # Effective bandwidth (bytes/s)
    effective_bw_gb_s = mem_bw_gb_s * max(eta, 0.01)
    effective_bw_bytes_s = effective_bw_gb_s * 1e9

    # Time per token
    time_per_token_s = weight_bytes / effective_bw_bytes_s
    total_time_s = output_tokens * time_per_token_s

    # Total bytes transferred across all decode steps
    total_bytes = weight_bytes * output_tokens
    energy_j = power_watts * total_time_s if power_watts > 0 else 0.0

    return PhaseResult(
        time_seconds=total_time_s,
        energy_joules=energy_j,
        flops=0.0,
        bytes_transferred=total_bytes,
    )


def estimate_power(hw: HardwareSpecs, alpha: float = DEFAULT_ALPHA) -> float:
    """Estimate average power draw as a fraction of TDP.

    This is a simplified power model: P = alpha * TDP.
    When calibration data is available, alpha is fitted from E2E traces.

    Args:
        hw: Hardware specifications.
        alpha: Power fraction (0 < alpha <= 1).

    Returns:
        Estimated average power in watts.
    """
    return hw.tdp_watts * max(min(alpha, 1.0), 0.0)


def predict(
    hw: HardwareSpecs,
    active_params_b: float,
    input_tokens: int,
    output_tokens: int,
    bytes_per_param: float = 1.0,
    calibration: CalibrationFactors | None = None,
    num_gpus: int = 1,
) -> SingleInferenceResult:
    """Predict latency and energy for a single LLM inference call.

    Uses the roofline model as the structural backbone. When calibration
    factors are provided, they replace the conservative defaults.

    If calibration includes fitted energy-per-token regression slopes,
    those are used directly instead of the analytical model (higher accuracy).

    Args:
        hw: Hardware specifications for the target accelerator.
        active_params_b: Active parameters in billions.
        input_tokens: Number of input tokens (context length).
        output_tokens: Number of output tokens to generate.
        bytes_per_param: Bytes per parameter (1 for FP8, 2 for FP16).
        calibration: Optional fitted calibration factors.
        num_gpus: Number of GPUs (for tensor-parallel scaling).

    Returns:
        SingleInferenceResult with prefill/decode breakdown.
    """
    eta_prefill = estimate_eta_for_model_size(active_params_b)
    eta_decode = DEFAULT_ETA_DECODE
    alpha = estimate_alpha_for_model_size(active_params_b)

    if calibration is not None:
        eta_prefill = calibration.eta_prefill
        eta_decode = calibration.eta_decode
        alpha = calibration.alpha

    # Scale for multi-GPU tensor parallelism:
    # Compute TFLOPS scales linearly, bandwidth scales linearly
    effective_tflops = hw.peak_tflops * num_gpus
    effective_bw = hw.hbm_bandwidth_gb_s * num_gpus
    # TDP also scales (each GPU draws its own power)
    effective_tdp_watts = hw.tdp_watts * num_gpus

    power_watts = effective_tdp_watts * max(min(alpha, 1.0), 0.0)

    # If calibration has regression-based energy-per-token, use it directly
    if (
        calibration is not None
        and calibration.energy_per_input_token_j is not None
        and calibration.energy_per_output_token_j is not None
    ):
        intercept = calibration.intercept_j or 0.0
        total_energy = (
            calibration.energy_per_input_token_j * input_tokens
            + calibration.energy_per_output_token_j * output_tokens
            + intercept
        )
        total_energy = max(total_energy, 0.0)

        # Still compute analytical time for the breakdown
        prefill = estimate_prefill(
            active_params_b, input_tokens, effective_tflops, eta_prefill, power_watts,
        )
        decode = estimate_decode(
            active_params_b, output_tokens, bytes_per_param, effective_bw, eta_decode, power_watts,
        )

        total_time = prefill.time_seconds + decode.time_seconds

        # Distribute regression energy proportionally to analytical time
        if total_time > 0:
            prefill_frac = prefill.time_seconds / total_time
            prefill.energy_joules = total_energy * prefill_frac
            decode.energy_joules = total_energy * (1.0 - prefill_frac)
        else:
            prefill.energy_joules = total_energy * 0.3  # reasonable default split
            decode.energy_joules = total_energy * 0.7

        return SingleInferenceResult(
            prefill=prefill,
            decode=decode,
            total_time_seconds=total_time,
            total_energy_joules=total_energy,
            avg_power_watts=total_energy / total_time if total_time > 0 else power_watts,
            tokens_per_second=output_tokens / total_time if total_time > 0 else 0.0,
        )

    # Analytical roofline model
    prefill = estimate_prefill(
        active_params_b, input_tokens, effective_tflops, eta_prefill, power_watts,
    )
    decode = estimate_decode(
        active_params_b, output_tokens, bytes_per_param, effective_bw, eta_decode, power_watts,
    )

    total_time = prefill.time_seconds + decode.time_seconds
    total_energy = prefill.energy_joules + decode.energy_joules

    return SingleInferenceResult(
        prefill=prefill,
        decode=decode,
        total_time_seconds=total_time,
        total_energy_joules=total_energy,
        avg_power_watts=total_energy / total_time if total_time > 0 else 0.0,
        tokens_per_second=output_tokens / total_time if total_time > 0 else 0.0,
    )


__all__ = [
    "BottomUpEnergyBreakdown",
    "DEFAULT_ALPHA",
    "DEFAULT_ETA_DECODE",
    "DEFAULT_ETA_PREFILL",
    "OVERHEAD_BASE",
    "OVERHEAD_BETA",
    "OVERHEAD_DECAY",
    "PJ_PER_BIT_HBM",
    "PJ_PER_FLOP_MATMUL_FP16",
    "SYSTEM_OVERHEAD_MULTIPLIER",
    "batch_overhead_multiplier",
    "estimate_alpha_for_model_size",
    "estimate_decode",
    "estimate_eta_for_model_size",
    "estimate_power",
    "estimate_prefill",
    "estimate_prefill_batch_aware",
    "estimate_prefill_bottomup",
    "predict",
]
