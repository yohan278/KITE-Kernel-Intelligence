"""Model specification types for describing LLM architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ArchitectureType(str, Enum):
    """High-level model architecture categories."""

    DENSE_TRANSFORMER = "dense_transformer"
    MOE_TRANSFORMER = "moe_transformer"
    SSM_HYBRID = "ssm_hybrid"
    LINEAR_ATTENTION = "linear_attention"


class AttentionType(str, Enum):
    """Attention head configuration variants."""

    MHA = "mha"   # Multi-Head Attention (num_kv_heads == num_heads)
    MQA = "mqa"   # Multi-Query Attention (num_kv_heads == 1)
    GQA = "gqa"   # Grouped-Query Attention (1 < num_kv_heads < num_heads)


@dataclass(frozen=True)
class LayerConfig:
    """Per-layer configuration for hybrid architectures.

    Allows specifying different layer types (attention, SSM, MoE, MTP)
    within a single model, e.g. Jamba-style interleaved attention/SSM.
    """

    layer_type: str  # "attention", "ssm", "moe_attention", "mtp"
    hidden_dim: int
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_dim: int = 0
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    ssm_state_size: Optional[int] = None


@dataclass(frozen=True)
class ModelSpec:
    """Frozen specification of an LLM architecture for profiling.

    All dimensions are per-layer unless stated otherwise.

    Attributes:
        model_id: HuggingFace-style model identifier (e.g., "Qwen/Qwen3-8B").
        architecture_type: High-level architecture category.
        attention_type: Attention head configuration.
        num_layers: Number of transformer layers.
        hidden_dim: Model hidden dimension (d_model).
        num_attention_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        head_dim: Per-head dimension.
        intermediate_dim: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length supported.
        num_experts: Number of MoE experts (None for dense models).
        experts_per_token: Experts activated per token (None for dense models).
        tie_word_embeddings: Whether input/output embeddings are tied.
        metadata: Additional architecture-specific metadata.
    """

    model_id: str
    architecture_type: ArchitectureType
    attention_type: AttentionType
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int = 131072
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    tie_word_embeddings: bool = False
    layer_configs: Optional[Tuple[LayerConfig, ...]] = None
    ssm_state_size: Optional[int] = None
    ssm_conv_width: Optional[int] = None
    ssm_n_heads: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_layer_config(self, layer_idx: int) -> LayerConfig:
        """Return per-layer config, falling back to uniform config from top-level fields.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            LayerConfig for the given layer.
        """
        if self.layer_configs is not None:
            if 0 <= layer_idx < len(self.layer_configs):
                return self.layer_configs[layer_idx]
            raise IndexError(
                f"layer_idx {layer_idx} out of range for {len(self.layer_configs)} layers"
            )
        # Fallback: build uniform config from top-level fields
        layer_type = "attention"
        if self.architecture_type == ArchitectureType.SSM_HYBRID:
            layer_type = "ssm"
        elif self.num_experts is not None:
            layer_type = "moe_attention"
        return LayerConfig(
            layer_type=layer_type,
            hidden_dim=self.hidden_dim,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
        )

    @property
    def total_params_billion(self) -> float:
        """Rough parameter count in billions (dense model estimate)."""
        # Embedding: vocab_size * hidden_dim
        embed_params = self.vocab_size * self.hidden_dim
        # Per-layer: QKV proj + O proj + MLP (up + gate + down) + 2x LayerNorm
        qkv_params = self.hidden_dim * (self.num_attention_heads + 2 * self.num_kv_heads) * self.head_dim
        o_params = self.num_attention_heads * self.head_dim * self.hidden_dim
        mlp_params = 3 * self.hidden_dim * self.intermediate_dim  # up + gate + down
        norm_params = 2 * self.hidden_dim
        per_layer = qkv_params + o_params + mlp_params + norm_params
        total_layer = per_layer * self.num_layers
        # LM head (if not tied)
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_dim
        return (embed_params + total_layer + lm_head) / 1e9

    @property
    def attention_layer_count(self) -> int:
        """Count of attention layers (from layer_configs or architecture default)."""
        if self.layer_configs is not None:
            return sum(1 for lc in self.layer_configs if "attention" in lc.layer_type)
        if self.architecture_type == ArchitectureType.SSM_HYBRID:
            return 0
        return self.num_layers

    @property
    def ssm_layer_count(self) -> int:
        """Count of SSM layers (from layer_configs or architecture default)."""
        if self.layer_configs is not None:
            return sum(1 for lc in self.layer_configs if "ssm" in lc.layer_type)
        if self.architecture_type == ArchitectureType.SSM_HYBRID:
            return self.num_layers
        return 0

    @property
    def kv_head_ratio(self) -> float:
        """Ratio of KV heads to query heads (1.0 for MHA)."""
        return self.num_kv_heads / self.num_attention_heads
