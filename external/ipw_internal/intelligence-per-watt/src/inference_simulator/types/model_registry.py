"""Canonical model registry with verified HuggingFace repo IDs and architecture metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec


MODEL_FAMILIES: Dict[str, Dict[str, Any]] = {
    "qwen3": {
        "dense": {
            "qwen3-0.6b": {
                "hf_id": "Qwen/Qwen3-0.6B",
                "total_params_b": 0.6,
                "active_params_b": 0.6,
                "architecture_type": ArchitectureType.DENSE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 28,
                "hidden_dim": 896,
                "num_attention_heads": 14,
                "num_kv_heads": 2,
                "head_dim": 64,
                "intermediate_dim": 4864,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "license": "Apache 2.0",
            },
            "qwen3-1.7b": {
                "hf_id": "Qwen/Qwen3-1.7B",
                "total_params_b": 1.7,
                "active_params_b": 1.7,
                "architecture_type": ArchitectureType.DENSE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 28,
                "hidden_dim": 2048,
                "num_attention_heads": 16,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 6144,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "license": "Apache 2.0",
            },
            "qwen3-4b": {
                "hf_id": "Qwen/Qwen3-4B",
                "total_params_b": 4.0,
                "active_params_b": 4.0,
                "architecture_type": ArchitectureType.DENSE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 36,
                "hidden_dim": 2560,
                "num_attention_heads": 32,
                "num_kv_heads": 4,
                "head_dim": 80,
                "intermediate_dim": 9216,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "license": "Apache 2.0",
            },
            "qwen3-8b": {
                "hf_id": "Qwen/Qwen3-8B",
                "total_params_b": 8.2,
                "active_params_b": 8.2,
                "architecture_type": ArchitectureType.DENSE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 36,
                "hidden_dim": 4096,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "intermediate_dim": 12288,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "license": "Apache 2.0",
            },
            "qwen3-14b": {
                "hf_id": "Qwen/Qwen3-14B",
                "total_params_b": 14.0,
                "active_params_b": 14.0,
                "architecture_type": ArchitectureType.DENSE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 40,
                "hidden_dim": 5120,
                "num_attention_heads": 40,
                "num_kv_heads": 8,
                "head_dim": 128,
                "intermediate_dim": 13824,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "license": "Apache 2.0",
            },
            "qwen3-32b": {
                "hf_id": "Qwen/Qwen3-32B",
                "total_params_b": 33.0,
                "active_params_b": 33.0,
                "architecture_type": ArchitectureType.DENSE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 64,
                "hidden_dim": 5120,
                "num_attention_heads": 40,
                "num_kv_heads": 8,
                "head_dim": 128,
                "intermediate_dim": 17408,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "license": "Apache 2.0",
            },
        },
        "moe": {
            "qwen3-30b-a3b": {
                "hf_id": "Qwen/Qwen3-30B-A3B",
                "total_params_b": 30.5,
                "active_params_b": 3.3,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 48,
                "hidden_dim": 2048,
                "num_attention_heads": 16,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 2560,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "num_experts": 128,
                "experts_per_token": 8,
                "license": "Apache 2.0",
            },
            "qwen3-235b-a22b": {
                "hf_id": "Qwen/Qwen3-235B-A22B",
                "total_params_b": 235.0,
                "active_params_b": 22.0,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 94,
                "hidden_dim": 4096,
                "num_attention_heads": 64,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 3072,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "num_experts": 128,
                "experts_per_token": 8,
                "license": "Apache 2.0",
            },
        },
        "next": {
            "qwen3-next-80b-a3b": {
                "hf_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
                "total_params_b": 80.0,
                "active_params_b": 3.0,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 62,
                "hidden_dim": 2560,
                "num_attention_heads": 20,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 1536,
                "vocab_size": 151936,
                "max_seq_len": 131072,
                "num_experts": 512,
                "experts_per_token": 10,
                "license": "Apache 2.0",
                "mtp": True,
                "shared_experts": 1,
            },
        },
    },
    "gpt-oss": {
        "models": {
            "gpt-oss-20b": {
                "hf_id": "openai/gpt-oss-20b",
                "total_params_b": 21.0,
                "active_params_b": 3.6,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 52,
                "hidden_dim": 2560,
                "num_attention_heads": 20,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 5120,
                "vocab_size": 200064,
                "max_seq_len": 131072,
                "num_experts": 32,
                "experts_per_token": 4,
                "license": "MIT",
                "native_precision": "MXFP4",
            },
            "gpt-oss-120b": {
                "hf_id": "openai/gpt-oss-120b",
                "total_params_b": 117.0,
                "active_params_b": 5.1,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 80,
                "hidden_dim": 4096,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "intermediate_dim": 3584,
                "vocab_size": 200064,
                "max_seq_len": 131072,
                "num_experts": 128,
                "experts_per_token": 4,
                "license": "MIT",
                "native_precision": "MXFP4",
            },
        },
    },
    "glm": {
        "models": {
            "glm-4.7": {
                "hf_id": "zai-org/GLM-4.7",
                "total_params_b": 358.0,
                "active_params_b": 32.0,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 62,
                "hidden_dim": 6656,
                "num_attention_heads": 52,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 4096,
                "vocab_size": 152064,
                "max_seq_len": 131072,
                "num_experts": 256,
                "experts_per_token": 8,
                "license": "MIT",
            },
            "glm-4.7-flash": {
                "hf_id": "zai-org/GLM-4.7-Flash",
                "total_params_b": 31.0,
                "active_params_b": 3.0,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 40,
                "hidden_dim": 3584,
                "num_attention_heads": 28,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 2816,
                "vocab_size": 152064,
                "max_seq_len": 131072,
                "num_experts": 64,
                "experts_per_token": 6,
                "license": "MIT",
            },
        },
    },
    "kimi": {
        "models": {
            "kimi-k2.5": {
                "hf_id": "moonshotai/Kimi-K2.5",
                "total_params_b": 1000.0,
                "active_params_b": 32.0,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 61,
                "hidden_dim": 7168,
                "num_attention_heads": 56,
                "num_kv_heads": 8,
                "head_dim": 128,
                "intermediate_dim": 2048,
                "vocab_size": 128256,
                "max_seq_len": 131072,
                "num_experts": 384,
                "experts_per_token": 8,
                "license": "MIT",
                "shared_experts": 1,
                "uses_mla": True,
            },
        },
    },
    "kimi-linear": {
        "models": {
            "kimi-linear-48b-a3b": {
                "hf_id": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
                "total_params_b": 48.0,
                "active_params_b": 3.0,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 62,
                "hidden_dim": 2560,
                "num_attention_heads": 20,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 1536,
                "vocab_size": 122880,
                "max_seq_len": 131072,
                "num_experts": 128,
                "experts_per_token": 8,
                "license": "MIT",
                "hybrid_linear_attention": True,
            },
        },
    },
    "moonlight": {
        "models": {
            "moonlight-16b-a3b": {
                "hf_id": "moonshotai/Moonlight-16B-A3B-Instruct",
                "total_params_b": 16.0,
                "active_params_b": 2.24,
                "architecture_type": ArchitectureType.MOE_TRANSFORMER,
                "attention_type": AttentionType.GQA,
                "num_layers": 27,
                "hidden_dim": 4096,
                "num_attention_heads": 32,
                "num_kv_heads": 4,
                "head_dim": 128,
                "intermediate_dim": 2048,
                "vocab_size": 102400,
                "max_seq_len": 131072,
                "num_experts": 64,
                "experts_per_token": 8,
                "license": "MIT",
                "base_architecture": "deepseek-v3",
                "optimizer": "muon",
            },
        },
    },
    "falcon-h1": {
        "models": {
            "falcon-h1-0.5b": {
                "hf_id": "tiiuae/Falcon-H1-0.5B-Base",
                "total_params_b": 0.5,
                "active_params_b": 0.5,
                "architecture_type": ArchitectureType.SSM_HYBRID,
                "attention_type": AttentionType.GQA,
                "num_layers": 36,
                "hidden_dim": 1024,
                "num_attention_heads": 8,
                "num_kv_heads": 2,
                "head_dim": 128,
                "intermediate_dim": 2816,
                "vocab_size": 130048,
                "max_seq_len": 262144,
                "ssm_state_size": 256,
                "ssm_conv_width": 4,
                "ssm_n_heads": 8,
                "license": "TII Falcon License",
                "ssm_type": "mamba2",
            },
            "falcon-h1-7b": {
                "hf_id": "tiiuae/Falcon-H1-7B-Base",
                "total_params_b": 7.0,
                "active_params_b": 7.0,
                "architecture_type": ArchitectureType.SSM_HYBRID,
                "attention_type": AttentionType.GQA,
                "num_layers": 44,
                "hidden_dim": 3072,
                "num_attention_heads": 12,
                "num_kv_heads": 2,
                "head_dim": 128,
                "intermediate_dim": 12288,
                "vocab_size": 130048,
                "max_seq_len": 262144,
                "ssm_state_size": 256,
                "ssm_conv_width": 4,
                "ssm_n_heads": 24,
                "license": "TII Falcon License",
                "ssm_type": "mamba2",
            },
            "falcon-h1-34b": {
                "hf_id": "tiiuae/Falcon-H1-34B-Base",
                "total_params_b": 34.0,
                "active_params_b": 34.0,
                "architecture_type": ArchitectureType.SSM_HYBRID,
                "attention_type": AttentionType.GQA,
                "num_layers": 60,
                "hidden_dim": 5120,
                "num_attention_heads": 40,
                "num_kv_heads": 8,
                "head_dim": 128,
                "intermediate_dim": 16384,
                "vocab_size": 130048,
                "max_seq_len": 262144,
                "ssm_state_size": 256,
                "ssm_conv_width": 4,
                "ssm_n_heads": 40,
                "license": "TII Falcon License",
                "ssm_type": "mamba2",
            },
        },
    },
    "lfm2": {
        "models": {
            "lfm2-1.2b": {
                "hf_id": "LiquidAI/LFM2-1.2B",
                "total_params_b": 1.2,
                "active_params_b": 1.2,
                "architecture_type": ArchitectureType.SSM_HYBRID,
                "attention_type": AttentionType.GQA,
                "num_layers": 16,
                "hidden_dim": 2048,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 64,
                "intermediate_dim": 12288,
                "vocab_size": 65536,
                "max_seq_len": 128000,
                "license": "LiquidAI License",
                "ssm_type": "conv_ssm",
                "full_attn_idxs": [2, 5, 8, 10, 12, 14],
                "conv_L_cache": 3,
            },
            "lfm2-2.6b": {
                "hf_id": "LiquidAI/LFM2-2.6B",
                "total_params_b": 2.6,
                "active_params_b": 2.6,
                "architecture_type": ArchitectureType.SSM_HYBRID,
                "attention_type": AttentionType.GQA,
                "num_layers": 28,
                "hidden_dim": 2560,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 80,
                "intermediate_dim": 13824,
                "vocab_size": 65536,
                "max_seq_len": 128000,
                "license": "LiquidAI License",
                "ssm_type": "conv_ssm",
                "full_attn_idxs": [3, 7, 11, 15, 19, 23, 27],
                "conv_L_cache": 3,
            },
        },
    },
}


def _build_hybrid_layer_configs(
    d: Dict[str, Any],
    num_layers: int,
    hidden_dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_dim: int,
) -> Optional[Tuple]:
    """Build LayerConfig tuples for hybrid SSM/attention models."""
    full_attn_idxs = d.get("full_attn_idxs")
    if full_attn_idxs is None:
        return None
    from inference_simulator.types.model_spec import LayerConfig
    configs = []
    attn_indices = set(full_attn_idxs)
    for i in range(num_layers):
        if i in attn_indices:
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
                ssm_state_size=d.get("ssm_state_size"),
            ))
    return tuple(configs)


def _flatten_registry() -> Dict[str, Dict[str, Any]]:
    """Flatten nested MODEL_FAMILIES into {model_key: spec_dict} with family metadata."""
    flat: Dict[str, Dict[str, Any]] = {}
    for family_name, family in MODEL_FAMILIES.items():
        for sub_name, sub in family.items():
            if isinstance(sub, dict) and "hf_id" in sub:
                entry = dict(sub)
                entry["_family"] = family_name
                entry["_subfamily"] = ""
                flat[sub_name] = entry
            elif isinstance(sub, dict):
                for model_key, model_data in sub.items():
                    if isinstance(model_data, dict) and "hf_id" in model_data:
                        entry = dict(model_data)
                        entry["_family"] = family_name
                        entry["_subfamily"] = sub_name
                        flat[model_key] = entry
    return flat


_FLAT_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _get_flat() -> Dict[str, Dict[str, Any]]:
    global _FLAT_REGISTRY
    if not _FLAT_REGISTRY:
        _FLAT_REGISTRY = _flatten_registry()
    return _FLAT_REGISTRY


def get_model_spec(model_key: str) -> ModelSpec:
    """Look up a model by key and return a ModelSpec.

    Args:
        model_key: Case-insensitive model key (e.g., "qwen3-8b", "gpt-oss-120b").

    Returns:
        A frozen ModelSpec dataclass for the requested model.

    Raises:
        KeyError: If model_key is not in the registry.
    """
    flat = _get_flat()
    key = model_key.lower()
    if key not in flat:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {sorted(flat.keys())}")
    d = flat[key]

    # Build metadata from extra keys not consumed by ModelSpec fields
    metadata: Dict[str, Any] = {
        "total_params_b": d.get("total_params_b"),
        "active_params_b": d.get("active_params_b"),
        "license": d.get("license", ""),
        "family": d["_family"],
        "subfamily": d["_subfamily"],
    }
    # Carry through any additional keys as metadata
    _spec_keys = {
        "hf_id", "total_params_b", "active_params_b", "architecture_type",
        "attention_type", "num_layers", "hidden_dim", "num_attention_heads",
        "num_kv_heads", "head_dim", "intermediate_dim", "vocab_size",
        "max_seq_len", "num_experts", "experts_per_token", "tie_word_embeddings",
        "ssm_state_size", "ssm_conv_width", "ssm_n_heads",
        "license", "_family", "_subfamily",
    }
    for k, v in d.items():
        if k not in _spec_keys:
            metadata[k] = v

    # Build layer_configs for hybrid models with full_attn_idxs
    layer_configs = _build_hybrid_layer_configs(
        d,
        num_layers=d["num_layers"],
        hidden_dim=d["hidden_dim"],
        num_attention_heads=d["num_attention_heads"],
        num_kv_heads=d["num_kv_heads"],
        head_dim=d["head_dim"],
        intermediate_dim=d["intermediate_dim"],
    )

    return ModelSpec(
        model_id=d["hf_id"],
        architecture_type=d["architecture_type"],
        attention_type=d["attention_type"],
        num_layers=d["num_layers"],
        hidden_dim=d["hidden_dim"],
        num_attention_heads=d["num_attention_heads"],
        num_kv_heads=d["num_kv_heads"],
        head_dim=d["head_dim"],
        intermediate_dim=d["intermediate_dim"],
        vocab_size=d["vocab_size"],
        max_seq_len=d.get("max_seq_len", 131072),
        num_experts=d.get("num_experts"),
        experts_per_token=d.get("experts_per_token"),
        tie_word_embeddings=d.get("tie_word_embeddings", False),
        layer_configs=layer_configs,
        ssm_state_size=d.get("ssm_state_size"),
        ssm_conv_width=d.get("ssm_conv_width"),
        ssm_n_heads=d.get("ssm_n_heads"),
        metadata=metadata,
    )


def list_models(
    family: Optional[str] = None,
    arch: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available models, optionally filtered by family or architecture type.

    Args:
        family: Filter by family name (e.g., "qwen3", "gpt-oss"). Case-insensitive.
        arch: Filter by ArchitectureType value (e.g., "moe_transformer"). Case-insensitive.

    Returns:
        List of dicts with keys: model_key, hf_id, family, architecture_type,
        total_params_b, active_params_b.
    """
    flat = _get_flat()
    results: List[Dict[str, Any]] = []
    for model_key, d in sorted(flat.items()):
        if family is not None and d["_family"] != family.lower():
            continue
        arch_type: ArchitectureType = d["architecture_type"]
        if arch is not None and arch_type.value != arch.lower():
            continue
        results.append({
            "model_key": model_key,
            "hf_id": d["hf_id"],
            "family": d["_family"],
            "architecture_type": arch_type.value,
            "total_params_b": d.get("total_params_b"),
            "active_params_b": d.get("active_params_b"),
        })
    return results
