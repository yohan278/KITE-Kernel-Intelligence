"""GLM model loader for ChatGLM family models."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader


class GLMModelLoader(HuggingFaceModelLoader):
    """Loader for GLM family models (ChatGLM, GLM-4, etc.).

    Handles GLM-specific config fields:
    - padded_vocab_size instead of vocab_size
    - multi_query_group_num for GQA
    - ffn_hidden_size for intermediate dimension
    """

    @property
    def supported_architectures(self) -> List[str]:
        return ["glm", "chatglm"]

    def _parse_config(self, model_id: str, config: Dict[str, Any]) -> ModelSpec:
        """Parse GLM config with family-specific field mapping."""
        mapped = dict(config)

        # GLM uses padded_vocab_size
        if "vocab_size" not in mapped:
            mapped["vocab_size"] = config.get("padded_vocab_size", config.get("vocab_size", 65024))

        # GLM uses multi_query_group_num for KV heads (GQA)
        if "num_key_value_heads" not in mapped:
            mapped["num_key_value_heads"] = config.get(
                "multi_query_group_num",
                config.get("num_key_value_heads", config.get("num_attention_heads", 32)),
            )

        # GLM uses ffn_hidden_size
        if "intermediate_size" not in mapped:
            mapped["intermediate_size"] = config.get(
                "ffn_hidden_size",
                config.get("intermediate_size", config.get("hidden_size", 4096) * 4),
            )

        # GLM uses num_layers
        if "num_hidden_layers" not in mapped:
            mapped["num_hidden_layers"] = config.get(
                "num_layers", config.get("num_hidden_layers", 28)
            )

        spec = super()._parse_config(model_id, mapped)

        # GLM metadata
        metadata = dict(spec.metadata)
        metadata["family"] = "glm"
        if "chatglm" in config.get("model_type", "").lower():
            metadata["variant"] = "chatglm"
        metadata["activation"] = config.get("hidden_act", "swiglu")

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
            tie_word_embeddings=spec.tie_word_embeddings,
            metadata=metadata,
        )
