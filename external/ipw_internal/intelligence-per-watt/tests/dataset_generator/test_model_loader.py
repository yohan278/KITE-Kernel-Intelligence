"""Tests for dataset_generator model loaders."""

import json
import pytest
from unittest.mock import patch, MagicMock

from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from dataset_generator.model_loader.base import BaseModelLoader
from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
from dataset_generator.model_loader.qwen3 import Qwen3ModelLoader


# Qwen3-8B config.json (key fields)
QWEN3_8B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "max_position_embeddings": 131072,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


class TestBaseModelLoader:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseModelLoader()


class TestHuggingFaceModelLoader:
    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_load_generic(self, mock_load_config):
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 32000,
            "intermediate_size": 3072,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "torch_dtype": "float16",
        }
        mock_load_config.return_value = config

        loader = HuggingFaceModelLoader()
        spec = loader.load("test/model")

        assert spec.model_id == "test/model"
        assert spec.hidden_dim == 768
        assert spec.num_layers == 12
        assert spec.num_attention_heads == 12
        assert spec.num_kv_heads == 12  # defaults to num_heads (MHA)
        assert spec.head_dim == 64  # 768 // 12
        assert spec.intermediate_dim == 3072
        assert spec.vocab_size == 32000
        assert spec.attention_type == AttentionType.MHA
        assert spec.architecture_type == ArchitectureType.DENSE_TRANSFORMER

    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_detect_gqa(self, mock_load_config):
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "model_type": "llama",
        }
        mock_load_config.return_value = config
        spec = HuggingFaceModelLoader().load("test/gqa")
        assert spec.attention_type == AttentionType.GQA

    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_detect_mqa(self, mock_load_config):
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
            "vocab_size": 32000,
            "model_type": "falcon",
        }
        mock_load_config.return_value = config
        spec = HuggingFaceModelLoader().load("test/mqa")
        assert spec.attention_type == AttentionType.MQA

    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_detect_moe(self, mock_load_config):
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "model_type": "mixtral",
        }
        mock_load_config.return_value = config
        spec = HuggingFaceModelLoader().load("test/moe")
        assert spec.architecture_type == ArchitectureType.MOE_TRANSFORMER
        assert spec.num_experts == 8
        assert spec.experts_per_token == 2

    def test_supported_architectures(self):
        loader = HuggingFaceModelLoader()
        assert "generic_hf" in loader.supported_architectures


class TestQwen3ModelLoader:
    @patch.object(Qwen3ModelLoader, "_load_config")
    def test_qwen3_8b(self, mock_load_config):
        mock_load_config.return_value = QWEN3_8B_CONFIG

        loader = Qwen3ModelLoader()
        spec = loader.load("Qwen/Qwen3-8B")

        # Verify all expected Qwen3-8B dimensions
        assert spec.model_id == "Qwen/Qwen3-8B"
        assert spec.num_layers == 36
        assert spec.hidden_dim == 4096
        assert spec.num_attention_heads == 32
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128
        assert spec.intermediate_dim == 11008
        assert spec.vocab_size == 151936
        assert spec.max_seq_len == 131072
        assert spec.attention_type == AttentionType.GQA
        assert spec.architecture_type == ArchitectureType.DENSE_TRANSFORMER
        assert spec.tie_word_embeddings is False

    @patch.object(Qwen3ModelLoader, "_load_config")
    def test_qwen3_metadata(self, mock_load_config):
        mock_load_config.return_value = QWEN3_8B_CONFIG

        spec = Qwen3ModelLoader().load("Qwen/Qwen3-8B")
        assert spec.metadata["family"] == "qwen3"
        assert spec.metadata["activation"] == "swiglu"

    def test_supported_architectures(self):
        loader = Qwen3ModelLoader()
        assert "qwen3" in loader.supported_architectures
        assert "qwen2" in loader.supported_architectures


class TestFamilyLoaders:
    """Tests that family-specific loaders correctly parse configs."""

    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_gpt_oss_loader(self, mock_load_config):
        from dataset_generator.model_loader.gpt_oss import GPTOSSModelLoader
        mock_load_config.return_value = {
            "model_type": "gpt2",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "vocab_size": 50257,
        }
        spec = GPTOSSModelLoader().load("openai-community/gpt2")
        assert spec.hidden_dim == 768
        assert spec.metadata["family"] == "gpt_oss"

    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_glm_loader(self, mock_load_config):
        from dataset_generator.model_loader.glm import GLMModelLoader
        mock_load_config.return_value = {
            "model_type": "chatglm",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "multi_query_group_num": 2,
            "num_layers": 28,
            "ffn_hidden_size": 13696,
            "padded_vocab_size": 65024,
        }
        spec = GLMModelLoader().load("THUDM/chatglm3-6b")
        assert spec.hidden_dim == 4096
        assert spec.metadata["family"] == "glm"

    @patch.object(HuggingFaceModelLoader, "_load_config")
    def test_kimi_loader(self, mock_load_config):
        from dataset_generator.model_loader.kimi import KimiModelLoader
        mock_load_config.return_value = {
            "model_type": "kimi",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 64000,
        }
        spec = KimiModelLoader().load("moonshot-ai/kimi-v1")
        assert spec.hidden_dim == 4096
        assert spec.metadata["family"] == "kimi"
