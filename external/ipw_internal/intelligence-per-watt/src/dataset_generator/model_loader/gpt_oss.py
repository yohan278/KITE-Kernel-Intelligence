"""GPT-OSS model loader for open-source GPT variants (GPT-2, GPT-J, GPT-NeoX)."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader


class GPTOSSModelLoader(HuggingFaceModelLoader):
    """Loader for open-source GPT variants.

    Handles GPT-2, GPT-J, GPT-NeoX, and similar architectures that use
    different config.json field names than the standard HuggingFace format.
    """

    @property
    def supported_architectures(self) -> List[str]:
        return ["gpt_oss", "gpt2", "gptj", "gpt_neox"]

    def _parse_config(self, model_id: str, config: Dict[str, Any]) -> ModelSpec:
        """Parse GPT-OSS config with family-specific field mapping.

        GPT variants may use:
        - n_embd / d_model instead of hidden_size
        - n_head instead of num_attention_heads
        - n_layer instead of num_hidden_layers
        - n_inner / d_ff instead of intermediate_size
        """
        # Map GPT-specific fields to standard HuggingFace field names
        mapped = dict(config)

        # Hidden dimension
        if "hidden_size" not in mapped:
            mapped["hidden_size"] = (
                config.get("n_embd")
                or config.get("d_model")
                or config.get("hidden_size", 768)
            )

        # Number of layers
        if "num_hidden_layers" not in mapped:
            mapped["num_hidden_layers"] = (
                config.get("n_layer")
                or config.get("num_layers")
                or config.get("num_hidden_layers", 12)
            )

        # Number of attention heads
        if "num_attention_heads" not in mapped:
            mapped["num_attention_heads"] = (
                config.get("n_head")
                or config.get("num_heads")
                or config.get("num_attention_heads", 12)
            )

        # Intermediate dimension (FFN)
        if "intermediate_size" not in mapped:
            hidden = mapped["hidden_size"]
            mapped["intermediate_size"] = (
                config.get("n_inner")
                or config.get("d_ff")
                or config.get("intermediate_size")
                or hidden * 4
            )

        # Vocab size
        if "vocab_size" not in mapped:
            mapped["vocab_size"] = config.get("vocab_size", 50257)

        # MoE fields: GPT-OSS variants may use non-standard field names
        if "num_local_experts" not in mapped and "num_experts" not in mapped:
            num_moe = config.get("num_moe_experts") or config.get("n_experts")
            if num_moe is not None:
                mapped["num_local_experts"] = num_moe
        if "num_experts_per_tok" not in mapped and "num_selected_experts" not in mapped:
            top_k = (
                config.get("top_k_experts")
                or config.get("experts_per_token")
                or config.get("moe_top_k")
            )
            if top_k is not None:
                mapped["num_experts_per_tok"] = top_k

        spec = super()._parse_config(model_id, mapped)

        # GPT-specific metadata
        metadata = dict(spec.metadata)
        metadata["family"] = "gpt_oss"
        model_type = config.get("model_type", "").lower()
        if "gpt2" in model_type:
            metadata["variant"] = "gpt2"
        elif "gptj" in model_type or "gpt-j" in model_type:
            metadata["variant"] = "gptj"
        elif "neox" in model_type:
            metadata["variant"] = "gpt_neox"
        metadata["activation"] = config.get("activation_function", "gelu")

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
            layer_configs=spec.layer_configs,
            ssm_state_size=spec.ssm_state_size,
            ssm_conv_width=spec.ssm_conv_width,
            ssm_n_heads=spec.ssm_n_heads,
            metadata=metadata,
        )
