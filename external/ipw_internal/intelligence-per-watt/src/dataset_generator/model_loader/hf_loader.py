"""Generic HuggingFace model loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    LayerConfig,
    ModelSpec,
)
from dataset_generator.model_loader.base import BaseModelLoader


class HuggingFaceModelLoader(BaseModelLoader):
    """Load model specs from HuggingFace Hub config.json files."""

    @property
    def supported_architectures(self) -> List[str]:
        return ["generic_hf"]

    def load(self, model_id: str) -> ModelSpec:
        """Download config.json from HuggingFace and parse into ModelSpec.

        Args:
            model_id: HuggingFace repo ID (e.g., "Qwen/Qwen3-8B").

        Returns:
            ModelSpec with architecture parameters.
        """
        config = self._load_config(model_id)
        return self._parse_config(model_id, config)

    def _load_config(self, model_id: str) -> Dict[str, Any]:
        """Download and parse config.json from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
        )
        with open(config_path, "r") as f:
            return json.load(f)

    def _parse_config(self, model_id: str, config: Dict[str, Any]) -> ModelSpec:
        """Parse HuggingFace config dict into ModelSpec."""
        hidden_dim = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        num_attention_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_attention_heads)
        intermediate_dim = config.get("intermediate_size", hidden_dim * 4)
        vocab_size = config["vocab_size"]
        max_seq_len = config.get(
            "max_position_embeddings",
            config.get("max_sequence_length", 131072),
        )
        head_dim = config.get("head_dim", hidden_dim // num_attention_heads)
        tie_word_embeddings = config.get("tie_word_embeddings", False)

        attention_type = self._detect_attention_type(num_attention_heads, num_kv_heads)
        architecture_type = self._detect_architecture_type(config)

        num_experts = config.get("num_local_experts", config.get("num_experts"))
        experts_per_token = config.get(
            "num_experts_per_tok",
            config.get("num_selected_experts"),
        )

        metadata = {
            "model_type": config.get("model_type", "unknown"),
            "torch_dtype": config.get("torch_dtype", "float16"),
        }

        # SSM-specific fields
        ssm_state_size = None
        ssm_conv_width = None
        ssm_n_heads = None
        model_type_lower = config.get("model_type", "").lower()
        if model_type_lower in ("falcon_h1",) or "mamba" in model_type_lower:
            ssm_state_size = config.get("mamba_d_state", config.get("ssm_state_size"))
            ssm_conv_width = config.get("mamba_d_conv")
            ssm_n_heads = config.get("mamba_n_heads", config.get("n_mamba_heads"))
            metadata["mamba_expand"] = config.get("mamba_expand")
            metadata["mamba_chunk_size"] = config.get("mamba_chunk_size")
        elif model_type_lower == "lfm2":
            metadata["full_attn_idxs"] = config.get("full_attn_idxs", [])
            metadata["conv_L_cache"] = config.get("conv_L_cache")
            metadata["conv_dim"] = config.get("conv_dim")

        # Detect per-layer configurations for hybrid models
        layer_configs = self._detect_per_layer_configs(
            config, hidden_dim, num_attention_heads, num_kv_heads,
            head_dim, intermediate_dim, num_experts, experts_per_token,
        )

        return ModelSpec(
            model_id=model_id,
            architecture_type=architecture_type,
            attention_type=attention_type,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=intermediate_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            tie_word_embeddings=tie_word_embeddings,
            layer_configs=layer_configs,
            ssm_state_size=ssm_state_size,
            ssm_conv_width=ssm_conv_width,
            ssm_n_heads=ssm_n_heads,
            metadata=metadata,
        )

    def _detect_attention_type(
        self, num_heads: int, num_kv_heads: int
    ) -> AttentionType:
        """Detect attention type from head counts."""
        if num_kv_heads == num_heads:
            return AttentionType.MHA
        if num_kv_heads == 1:
            return AttentionType.MQA
        return AttentionType.GQA

    def _detect_per_layer_configs(
        self,
        config: Dict[str, Any],
        hidden_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_dim: int,
        num_experts: Optional[int],
        experts_per_token: Optional[int],
    ) -> Optional[Tuple[LayerConfig, ...]]:
        """Detect per-layer config arrays in HuggingFace config.json.

        Some hybrid models (e.g. Jamba) specify per-layer types via
        ``layers`` or ``decoder_layers`` arrays in the config.
        """
        # Falcon-H1: Use attn_layer_indices to mark attention vs SSM layers
        attn_layer_indices = config.get("attn_layer_indices", config.get("attn_layer_idx"))
        if attn_layer_indices is not None:
            num_layers_cfg = config.get("num_hidden_layers", 0)
            attn_set = set(attn_layer_indices)
            configs: List[LayerConfig] = []
            for i in range(num_layers_cfg):
                if i in attn_set:
                    configs.append(LayerConfig(
                        layer_type="attention",
                        hidden_dim=hidden_dim,
                        num_attention_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        intermediate_dim=intermediate_dim,
                    ))
                else:
                    configs.append(LayerConfig(
                        layer_type="ssm",
                        hidden_dim=hidden_dim,
                        intermediate_dim=intermediate_dim,
                        ssm_state_size=config.get("mamba_d_state", config.get("ssm_state_size")),
                    ))
            return tuple(configs) if configs else None

        # LFM2: Use full_attn_idxs
        full_attn_idxs = config.get("full_attn_idxs")
        if full_attn_idxs is not None:
            num_layers_cfg = config.get("num_hidden_layers", 0)
            attn_set = set(full_attn_idxs)
            configs_lfm: List[LayerConfig] = []
            for i in range(num_layers_cfg):
                if i in attn_set:
                    configs_lfm.append(LayerConfig(
                        layer_type="attention",
                        hidden_dim=hidden_dim,
                        num_attention_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        intermediate_dim=intermediate_dim,
                    ))
                else:
                    configs_lfm.append(LayerConfig(
                        layer_type="ssm",
                        hidden_dim=hidden_dim,
                        intermediate_dim=intermediate_dim,
                    ))
            return tuple(configs_lfm) if configs_lfm else None

        layer_defs = config.get("layers", config.get("decoder_layers"))
        if not isinstance(layer_defs, list) or len(layer_defs) == 0:
            return None

        configs: List[LayerConfig] = []
        for layer_def in layer_defs:
            if isinstance(layer_def, str):
                # Simple string like "attention", "ssm", "moe_attention"
                layer_type = layer_def
                lc = LayerConfig(
                    layer_type=layer_type,
                    hidden_dim=hidden_dim,
                    num_attention_heads=num_attention_heads if "attention" in layer_type else 0,
                    num_kv_heads=num_kv_heads if "attention" in layer_type else 0,
                    head_dim=head_dim if "attention" in layer_type else 0,
                    intermediate_dim=intermediate_dim,
                    num_experts=num_experts if "moe" in layer_type else None,
                    experts_per_token=experts_per_token if "moe" in layer_type else None,
                    ssm_state_size=config.get("ssm_state_size") if "ssm" in layer_type else None,
                )
            elif isinstance(layer_def, dict):
                layer_type = layer_def.get("type", layer_def.get("layer_type", "attention"))
                lc = LayerConfig(
                    layer_type=layer_type,
                    hidden_dim=layer_def.get("hidden_size", hidden_dim),
                    num_attention_heads=layer_def.get("num_attention_heads", num_attention_heads if "attention" in layer_type else 0),
                    num_kv_heads=layer_def.get("num_key_value_heads", num_kv_heads if "attention" in layer_type else 0),
                    head_dim=layer_def.get("head_dim", head_dim if "attention" in layer_type else 0),
                    intermediate_dim=layer_def.get("intermediate_size", intermediate_dim),
                    num_experts=layer_def.get("num_experts", num_experts if "moe" in layer_type else None),
                    experts_per_token=layer_def.get("num_experts_per_tok", experts_per_token if "moe" in layer_type else None),
                    ssm_state_size=layer_def.get("ssm_state_size") if "ssm" in layer_type else None,
                )
            else:
                continue
            configs.append(lc)

        return tuple(configs) if configs else None

    def _detect_architecture_type(self, config: Dict[str, Any]) -> ArchitectureType:
        """Detect architecture type from config fields."""
        model_type = config.get("model_type", "").lower()
        # Check MoE first (some SSM-MoE hybrids should be treated as MoE)
        if config.get("num_local_experts") or config.get("num_experts"):
            return ArchitectureType.MOE_TRANSFORMER
        if "mamba" in model_type or "ssm" in model_type or model_type in ("falcon_h1", "lfm2"):
            return ArchitectureType.SSM_HYBRID
        return ArchitectureType.DENSE_TRANSFORMER
