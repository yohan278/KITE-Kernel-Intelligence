"""Centralized configuration for the event-driven simulator.

Collects all tunable constants that were previously hardcoded across
the simulator pipeline.  Every field carries a measured default (see
``data/micro_experiments/`` for grounding methodology) and can be
overridden via JSON or programmatically.

Usage::

    config = SimulatorConfig()                 # conservative defaults
    config = SimulatorConfig.from_json(path)   # measured overrides
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# Precision → bytes-per-element lookup.  Centralised here so that every
# consumer agrees on the mapping (fixes the ``"fp16" else 1.0`` bug).
PRECISION_BYTES: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
}


@dataclass(frozen=True)
class SimulatorConfig:
    """All tunable constants for the inference simulator pipeline.

    Attributes whose defaults were formerly hardcoded are annotated with
    the original source location for traceability.
    """

    # --- Batching overheads (simulator.py) ---
    # GQA batching overhead applied when num_kv_heads < num_attention_heads.
    # Original: simulator.py:130  (was 0.1)
    gqa_batching_overhead: float = 0.1

    # Per-request overhead for multi-request prefill batches.
    # Original: simulator.py:563  (was 0.05)
    prefill_batch_overhead: float = 0.05

    # --- Event-loop safety limits ---
    # Floor on any single GPU operation time (prevents event storms).
    # Original: simulator.py:203  (was 1e-4)
    min_time_s: float = 1e-4

    # KV-cache block size for the cache manager.
    # Original: simulator.py:354  (was 16)
    kv_cache_block_size: int = 16

    # Maximum events before the simulation is force-stopped.
    # Original: simulator.py:390  (was 500_000)
    max_events: int = 500_000

    # --- Roofline calibration (inference_model.py:33-35) ---
    # Compute-efficiency factor during prefill (fraction of peak TFLOPS).
    eta_prefill: float = 0.4

    # Memory-bandwidth efficiency factor during decode.
    # Fitted from A100-80GB GT benchmarks (Qwen3 0.6B-14B, chat qps=2,
    # 1gpu-fp16).  Per-model optimal etas range from 0.30 (0.6B, kernel-
    # launch dominated) to 0.73 (8B, fully bandwidth-bound); 0.58 is the
    # brute-force minimum of median |%-error| across five model sizes
    # (median APE ~11%, vs ~23% with the prior 0.50 default).
    eta_decode: float = 0.58

    # Power fraction: P = alpha * TDP.
    alpha_power: float = 0.65

    # --- TP scaling (lut_generator.py:71) ---
    # "linear" = divide time by tp  (current).
    # "measured" = divide time by tp PLUS comm_overhead_s.
    tp_scaling_mode: str = "measured"

    # Per-AllReduce-per-hop communication overhead (seconds).
    # Each decode step has num_layers*2 AllReduce ops (attn + FFN per layer);
    # in a ring topology each traverses (tp-1) hops.
    # 8μs accounts for NCCL kernel launch + NVLink sync.
    tp_comm_overhead_s: float = 8e-6

    # CPU scheduling overhead per step (nanoseconds).
    # Accounts for Python scheduler + CUDA kernel launch in vLLM.
    cpu_overhead_ns: int = 50_000  # 50μs

    # --- Power model (utilization-scaled fallback) ---
    # GPU idle power as fraction of TDP (no active compute).
    power_idle_fraction: float = 0.30

    # GPU active power as fraction of TDP (fully utilized).
    power_active_fraction: float = 0.85

    # --- vLLM fused-kernel decomposition (vllm_engine.py:185,199) ---
    # Fraction of fused forward-pass time attributed to attention.
    fused_attention_fraction: float = 0.4

    # Fraction of fused forward-pass time attributed to MLP.
    fused_mlp_fraction: float = 0.55

    # --- Per-layer operator counts (for LUT-based forward pass summation) ---
    # These counts define how many invocations of each operator category
    # occur per transformer layer.  Defaults match standard dense Qwen3/Llama
    # architecture (SwiGLU MLP, RMSNorm, GQA attention).
    ops_linear_per_layer: int = 5      # QKV, O, up, gate, down projections
    ops_norm_per_layer: int = 2        # pre-attention norm, pre-MLP norm
    ops_activation_per_layer: int = 4  # silu/gelu, softmax, dropout, residual_add
    ops_embedding_per_layer: int = 0   # rotary embedding is fused inside attention kernel

    # --- Overlap correction (Phase 6) ---
    # 1.0 = no correction (conservative: assumes no compute/memory overlap).
    # <1.0 = discount total time to model compute/memory overlap, kernel
    # fusion, and category-averaging artifacts in the token-ops LUT.
    overlap_correction: float = 1.0

    # --- Learned composition weights (IrEne-inspired) ---
    # Path to a JSON file with CompositionWeights (from composition_model.py).
    # When set, overrides the integer ops_*_per_layer counts with learned
    # real-valued weights.
    composition_weights_path: Optional[str] = None

    # --- Per-layer event emission toggle ---
    # When True, the simulator emits per-layer-per-category OperatorEvents
    # for fine-grained energy breakdown.  Set to False to revert to
    # aggregate-only events (lower overhead).
    emit_per_layer_events: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulatorConfig":
        """Load config from a JSON file, using defaults for missing keys."""
        with open(path) as f:
            data = json.load(f)
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_json(self, path: str | Path) -> None:
        """Persist config to a JSON file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def precision_bytes_for(precision: str) -> float:
    """Return the number of bytes per element for *precision*.

    Falls back to 2.0 (fp16) for unrecognised strings.
    """
    return PRECISION_BYTES.get(precision, 2.0)
