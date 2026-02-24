"""Tests for SSM model registry and loader support."""
from __future__ import annotations

import pytest

from inference_simulator.types.model_registry import get_model_spec, list_models
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)


class TestFalconH1Registry:
    def test_falcon_h1_7b_lookup(self):
        spec = get_model_spec("falcon-h1-7b")
        assert spec.model_id == "tiiuae/Falcon-H1-7B-Base"
        assert spec.architecture_type == ArchitectureType.SSM_HYBRID
        assert spec.attention_type == AttentionType.GQA
        assert spec.num_layers == 44
        assert spec.hidden_dim == 3072
        assert spec.num_attention_heads == 12
        assert spec.num_kv_heads == 2
        assert spec.head_dim == 128
        assert spec.intermediate_dim == 12288
        assert spec.vocab_size == 130048
        assert spec.max_seq_len == 262144
        assert spec.ssm_state_size == 256
        assert spec.ssm_conv_width == 4
        assert spec.ssm_n_heads == 24

    def test_falcon_h1_0_5b_lookup(self):
        spec = get_model_spec("falcon-h1-0.5b")
        assert spec.architecture_type == ArchitectureType.SSM_HYBRID
        assert spec.ssm_state_size == 256

    def test_falcon_h1_34b_lookup(self):
        spec = get_model_spec("falcon-h1-34b")
        assert spec.architecture_type == ArchitectureType.SSM_HYBRID
        assert spec.num_layers == 60

    def test_falcon_h1_metadata(self):
        spec = get_model_spec("falcon-h1-7b")
        assert spec.metadata.get("ssm_type") == "mamba2"
        assert spec.metadata.get("family") == "falcon-h1"


class TestLFM2Registry:
    def test_lfm2_1_2b_lookup(self):
        spec = get_model_spec("lfm2-1.2b")
        assert spec.model_id == "LiquidAI/LFM2-1.2B"
        assert spec.architecture_type == ArchitectureType.SSM_HYBRID
        assert spec.attention_type == AttentionType.GQA
        assert spec.num_layers == 16
        assert spec.hidden_dim == 2048
        assert spec.num_attention_heads == 32
        assert spec.num_kv_heads == 8
        assert spec.intermediate_dim == 12288
        assert spec.vocab_size == 65536
        assert spec.max_seq_len == 128000

    def test_lfm2_2_6b_lookup(self):
        spec = get_model_spec("lfm2-2.6b")
        assert spec.architecture_type == ArchitectureType.SSM_HYBRID
        assert spec.num_layers == 28

    def test_lfm2_layer_configs(self):
        spec = get_model_spec("lfm2-1.2b")
        assert spec.layer_configs is not None
        assert len(spec.layer_configs) == 16
        # Attention layers at indices 2, 5, 8, 10, 12, 14
        attn_indices = {2, 5, 8, 10, 12, 14}
        for i, lc in enumerate(spec.layer_configs):
            if i in attn_indices:
                assert lc.layer_type == "attention", f"Layer {i} should be attention"
            else:
                assert lc.layer_type == "ssm", f"Layer {i} should be SSM"

    def test_lfm2_metadata(self):
        spec = get_model_spec("lfm2-1.2b")
        assert spec.metadata.get("ssm_type") == "conv_ssm"


class TestSSMModelSpecProperties:
    def test_attention_layer_count_hybrid(self):
        spec = get_model_spec("lfm2-1.2b")
        assert spec.attention_layer_count == 6  # 6 attention layers
        assert spec.ssm_layer_count == 10  # 10 SSM layers

    def test_attention_layer_count_dense(self):
        spec = get_model_spec("qwen3-8b")
        assert spec.attention_layer_count == 36  # all layers are attention
        assert spec.ssm_layer_count == 0

    def test_ssm_fields_none_for_dense(self):
        spec = get_model_spec("qwen3-8b")
        assert spec.ssm_state_size is None
        assert spec.ssm_conv_width is None
        assert spec.ssm_n_heads is None

    def test_total_params_property(self):
        spec = get_model_spec("falcon-h1-7b")
        # Should compute something reasonable
        assert spec.total_params_billion > 0


class TestSSMListModels:
    def test_list_ssm_hybrid_models(self):
        models = list_models(arch="ssm_hybrid")
        assert len(models) >= 5  # 3 falcon-h1 + 2 lfm2
        model_keys = {m["model_key"] for m in models}
        assert "falcon-h1-7b" in model_keys
        assert "lfm2-1.2b" in model_keys

    def test_list_falcon_h1_family(self):
        models = list_models(family="falcon-h1")
        assert len(models) == 3


class TestHFLoaderSSM:
    """Test HF loader detection for SSM architectures (using mock configs)."""

    def test_detect_falcon_h1_architecture(self):
        from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
        loader = HuggingFaceModelLoader()
        config = {
            "model_type": "falcon_h1",
            "hidden_size": 3072,
            "num_hidden_layers": 44,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 12288,
            "vocab_size": 130048,
            "mamba_d_state": 256,
            "mamba_n_heads": 24,
            "mamba_d_conv": 4,
        }
        arch = loader._detect_architecture_type(config)
        assert arch == ArchitectureType.SSM_HYBRID

    def test_detect_lfm2_architecture(self):
        from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
        loader = HuggingFaceModelLoader()
        config = {
            "model_type": "lfm2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "vocab_size": 65536,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
        }
        arch = loader._detect_architecture_type(config)
        assert arch == ArchitectureType.SSM_HYBRID

    def test_detect_lfm2_layer_configs(self):
        from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
        loader = HuggingFaceModelLoader()
        config = {
            "model_type": "lfm2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "intermediate_size": 12288,
            "vocab_size": 65536,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
        }
        layer_configs = loader._detect_per_layer_configs(
            config, 2048, 32, 8, 64, 12288, None, None
        )
        assert layer_configs is not None
        assert len(layer_configs) == 16
        assert layer_configs[2].layer_type == "attention"
        assert layer_configs[0].layer_type == "ssm"

    def test_detect_falcon_h1_with_attn_indices(self):
        from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
        loader = HuggingFaceModelLoader()
        config = {
            "model_type": "falcon_h1",
            "hidden_size": 3072,
            "num_hidden_layers": 8,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 12288,
            "vocab_size": 130048,
            "mamba_d_state": 256,
            "attn_layer_indices": [1, 4, 7],
        }
        layer_configs = loader._detect_per_layer_configs(
            config, 3072, 12, 2, 128, 12288, None, None
        )
        assert layer_configs is not None
        assert len(layer_configs) == 8
        assert layer_configs[1].layer_type == "attention"
        assert layer_configs[0].layer_type == "ssm"
        assert layer_configs[4].layer_type == "attention"
        assert layer_configs[3].layer_type == "ssm"

    def test_moe_ssm_hybrid_detected_as_moe(self):
        """MoE + SSM hybrid should be treated as MoE."""
        from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
        loader = HuggingFaceModelLoader()
        config = {
            "model_type": "falcon_h1",
            "num_local_experts": 8,
            "hidden_size": 3072,
            "num_hidden_layers": 44,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
        }
        arch = loader._detect_architecture_type(config)
        assert arch == ArchitectureType.MOE_TRANSFORMER
