"""LUT (Lookup Table) bundle types for dense operator performance tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LUTBundle:
    """Reference to all lookup tables for a (model, hardware, quantization) triple.

    A LUT bundle is the output of Pipeline #1b (Runtime Estimator). It contains
    dense .npz files covering the full operating range of each operator, plus
    fitted tool latency distributions for agentic workloads.

    Attributes:
        base_dir: Root directory containing all bundle files.
        model_id: Model identifier (e.g., "Qwen/Qwen3-8B").
        hardware_id: Hardware identifier (e.g., "a100_80gb").
        quantization: Precision/quantization (e.g., "bf16", "fp16").
        gpu_token_ops_lut: Path to token-level ops .npz (linear, norm, activation).
        gpu_attention_prefill_lut: Path to attention prefill .npz.
        gpu_attention_decode_lut: Path to attention decode .npz.
        gpu_moe_lut: Path to MoE ops .npz (None for dense models).
        fused_ops_lut: Path to fused kernel ops .npz (None if no fused profiling).
        network_lut: Path to communication ops .npz (None for single-GPU).
        cpu_ops_lut: Path to CPU-side ops .npz (None if not profiled).
        energy_lut: Path to energy consumption .npz (None if not profiled).
        tool_distributions: Path to fitted tool distributions .pkl file.
        metadata: Estimator accuracy scores, coverage ranges, etc.
    """

    base_dir: Path
    model_id: str
    hardware_id: str
    quantization: str
    gpu_token_ops_lut: Path
    gpu_attention_prefill_lut: Path
    gpu_attention_decode_lut: Path
    gpu_moe_lut: Optional[Path] = None
    gpu_ssm_lut: Optional[Path] = None
    fused_ops_lut: Optional[Path] = None
    network_lut: Optional[Path] = None
    cpu_ops_lut: Optional[Path] = None
    energy_lut: Optional[Path] = None
    tool_distributions: Optional[Path] = None
    composition_weights: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def exists(self) -> bool:
        """Check that all required LUT files exist on disk."""
        required = [
            self.gpu_token_ops_lut,
            self.gpu_attention_prefill_lut,
            self.gpu_attention_decode_lut,
        ]
        return all(p.exists() for p in required)

    def optional_paths(self) -> Dict[str, Optional[Path]]:
        """Return a dict of optional LUT paths and their availability."""
        return {
            "gpu_moe_lut": self.gpu_moe_lut,
            "gpu_ssm_lut": self.gpu_ssm_lut,
            "fused_ops_lut": self.fused_ops_lut,
            "network_lut": self.network_lut,
            "cpu_ops_lut": self.cpu_ops_lut,
            "energy_lut": self.energy_lut,
            "tool_distributions": self.tool_distributions,
            "composition_weights": self.composition_weights,
        }


__all__ = ["LUTBundle"]
