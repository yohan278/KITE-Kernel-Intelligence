"""Qwen3-specific model loader with SwiGLU and GQA handling."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.model_spec import ModelSpec
from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader


class Qwen3ModelLoader(HuggingFaceModelLoader):
    """Model loader with Qwen3-specific handling.

    Handles:
    - SwiGLU FFN intermediate dimension
    - GQA verification (Qwen3-8B: 32 Q heads, 8 KV heads)
    """

    @property
    def supported_architectures(self) -> List[str]:
        return ["qwen3", "qwen2"]

    def _parse_config(self, model_id: str, config: Dict[str, Any]) -> ModelSpec:
        """Parse with Qwen3-specific adjustments."""
        # Qwen3 uses SwiGLU: intermediate_size in config is the actual SwiGLU dim
        # The config already specifies the correct intermediate_size
        spec = super()._parse_config(model_id, config)

        # Qwen3 metadata
        metadata = dict(spec.metadata)
        metadata["family"] = "qwen3"
        metadata["activation"] = "swiglu"

        # Reconstruct with updated metadata (frozen dataclass)
        return ModelSpec(
            model_id=spec.model_id,
            architecture_type=spec.architecture_type,
            attention_type=spec.attention_type,
            num_layers=spec.num_layers,
            hidden_dim=spec.hidden_dim,
            num_attention_heads=spec.num_attention_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            intermediate_dim=spec.intermediate_dim,
            vocab_size=spec.vocab_size,
            max_seq_len=spec.max_seq_len,
            num_experts=spec.num_experts,
            experts_per_token=spec.experts_per_token,
            tie_word_embeddings=spec.tie_word_embeddings,
            metadata=metadata,
        )
