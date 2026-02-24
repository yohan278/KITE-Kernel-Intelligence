"""Kimi model loader for Moonshot AI models."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader


class KimiModelLoader(HuggingFaceModelLoader):
    """Loader for Kimi/Moonshot models.

    Handles Kimi-specific config fields and MoE variants (Kimi-K2).
    """

    @property
    def supported_architectures(self) -> List[str]:
        return ["kimi", "moonshot"]

    def _parse_config(self, model_id: str, config: Dict[str, Any]) -> ModelSpec:
        """Parse Kimi config with family-specific field mapping."""
        mapped = dict(config)

        # Kimi may use different field names for MoE
        if "num_local_experts" not in mapped and "num_experts" in config:
            mapped["num_local_experts"] = config["num_experts"]

        if "num_experts_per_tok" not in mapped and "num_selected_experts" in config:
            mapped["num_experts_per_tok"] = config["num_selected_experts"]

        spec = super()._parse_config(model_id, mapped)

        # Kimi metadata
        metadata = dict(spec.metadata)
        metadata["family"] = "kimi"
        metadata["activation"] = config.get("hidden_act", "silu")
        if spec.num_experts is not None:
            metadata["variant"] = "kimi_moe"
        else:
            metadata["variant"] = "kimi_dense"

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
