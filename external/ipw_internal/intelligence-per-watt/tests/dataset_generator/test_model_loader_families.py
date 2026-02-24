"""Tests for family-specific model loaders (GPT-OSS, GLM, Kimi)."""

from __future__ import annotations

import pytest

from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)
from dataset_generator.model_loader.gpt_oss import GPTOSSModelLoader
from dataset_generator.model_loader.glm import GLMModelLoader
from dataset_generator.model_loader.kimi import KimiModelLoader


class TestGPTOSSModelLoader:
    def test_supported_architectures(self):
        loader = GPTOSSModelLoader()
        assert "gpt_oss" in loader.supported_architectures

    def test_parse_gpt2_config(self):
        """Test parsing GPT-2 style config with n_embd, n_head, n_layer."""
        loader = GPTOSSModelLoader()
        config = {
            "model_type": "gpt2",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_inner": 3072,
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "activation_function": "gelu_new",
        }
        spec = loader._parse_config("openai-community/gpt2", config)
        assert spec.hidden_dim == 768
        assert spec.num_attention_heads == 12
        assert spec.num_layers == 12
        assert spec.intermediate_dim == 3072
        assert spec.vocab_size == 50257
        assert spec.metadata["family"] == "gpt_oss"
        assert spec.metadata["variant"] == "gpt2"

    def test_parse_gpt_neox_config(self):
        """Test parsing GPT-NeoX style config."""
        loader = GPTOSSModelLoader()
        config = {
            "model_type": "gpt_neox",
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_hidden_layers": 44,
            "intermediate_size": 24576,
            "vocab_size": 50432,
            "max_position_embeddings": 2048,
        }
        spec = loader._parse_config("EleutherAI/gpt-neox-20b", config)
        assert spec.hidden_dim == 6144
        assert spec.num_layers == 44
        assert spec.metadata["variant"] == "gpt_neox"

    def test_default_intermediate_dim(self):
        """When intermediate_size is missing, default to hidden * 4."""
        loader = GPTOSSModelLoader()
        config = {
            "model_type": "gpt2",
            "n_embd": 1024,
            "n_head": 16,
            "n_layer": 24,
            "vocab_size": 50257,
        }
        spec = loader._parse_config("test/gpt2-medium", config)
        assert spec.intermediate_dim == 1024 * 4


class TestGLMModelLoader:
    def test_supported_architectures(self):
        loader = GLMModelLoader()
        assert "glm" in loader.supported_architectures
        assert "chatglm" in loader.supported_architectures

    def test_parse_chatglm_config(self):
        """Test parsing ChatGLM-style config with padded_vocab_size and ffn_hidden_size."""
        loader = GLMModelLoader()
        config = {
            "model_type": "chatglm",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "multi_query_group_num": 2,
            "num_layers": 28,
            "ffn_hidden_size": 13696,
            "padded_vocab_size": 65024,
            "max_position_embeddings": 8192,
        }
        spec = loader._parse_config("THUDM/chatglm3-6b", config)
        assert spec.hidden_dim == 4096
        assert spec.num_kv_heads == 2  # GQA via multi_query_group_num
        assert spec.intermediate_dim == 13696
        assert spec.vocab_size == 65024
        assert spec.num_layers == 28
        assert spec.attention_type == AttentionType.GQA
        assert spec.metadata["family"] == "glm"
        assert spec.metadata["variant"] == "chatglm"

    def test_parse_glm4_config(self):
        """Test GLM-4 style config."""
        loader = GLMModelLoader()
        config = {
            "model_type": "chatglm",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "multi_query_group_num": 4,
            "num_layers": 40,
            "ffn_hidden_size": 13696,
            "padded_vocab_size": 151552,
            "max_position_embeddings": 131072,
        }
        spec = loader._parse_config("THUDM/glm-4-9b", config)
        assert spec.hidden_dim == 4096
        assert spec.num_kv_heads == 4
        assert spec.num_layers == 40


class TestKimiModelLoader:
    def test_supported_architectures(self):
        loader = KimiModelLoader()
        assert "kimi" in loader.supported_architectures

    def test_parse_dense_kimi(self):
        """Test parsing a dense Kimi model config."""
        loader = KimiModelLoader()
        config = {
            "model_type": "kimi",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 64000,
            "max_position_embeddings": 131072,
            "hidden_act": "silu",
        }
        spec = loader._parse_config("moonshot-ai/kimi-v1", config)
        assert spec.hidden_dim == 4096
        assert spec.num_kv_heads == 8
        assert spec.metadata["family"] == "kimi"
        assert spec.metadata["variant"] == "kimi_dense"
        assert spec.num_experts is None

    def test_parse_moe_kimi(self):
        """Test parsing a Kimi MoE variant (K2-style)."""
        loader = KimiModelLoader()
        config = {
            "model_type": "kimi",
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "num_hidden_layers": 48,
            "intermediate_size": 13824,
            "vocab_size": 64000,
            "num_experts": 64,
            "num_selected_experts": 8,
            "max_position_embeddings": 131072,
        }
        spec = loader._parse_config("moonshot-ai/kimi-k2", config)
        assert spec.architecture_type == ArchitectureType.MOE_TRANSFORMER
        assert spec.num_experts == 64
        assert spec.experts_per_token == 8
        assert spec.metadata["variant"] == "kimi_moe"
