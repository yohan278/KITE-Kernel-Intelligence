"""Tests for the model registry."""

import pytest

from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from inference_simulator.types.model_registry import (
    MODEL_FAMILIES,
    get_model_spec,
    list_models,
)


ALL_MODEL_KEYS = [
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "qwen3-next-80b-a3b",
    "gpt-oss-20b", "gpt-oss-120b",
    "glm-4.7", "glm-4.7-flash",
    "kimi-k2.5",
    "kimi-linear-48b-a3b",
    "moonlight-16b-a3b",
    "falcon-h1-0.5b", "falcon-h1-7b", "falcon-h1-34b",
    "lfm2-1.2b", "lfm2-2.6b",
]


class TestGetModelSpec:
    def test_qwen3_8b_fields(self):
        spec = get_model_spec("qwen3-8b")
        assert isinstance(spec, ModelSpec)
        assert spec.model_id == "Qwen/Qwen3-8B"
        assert spec.architecture_type == ArchitectureType.DENSE_TRANSFORMER
        assert spec.attention_type == AttentionType.GQA
        assert spec.num_layers == 36
        assert spec.hidden_dim == 4096
        assert spec.num_attention_heads == 32
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128
        assert spec.intermediate_dim == 12288
        assert spec.vocab_size == 151936
        assert spec.max_seq_len == 131072
        assert spec.num_experts is None
        assert spec.experts_per_token is None

    def test_qwen3_8b_metadata(self):
        spec = get_model_spec("qwen3-8b")
        assert spec.metadata["total_params_b"] == 8.2
        assert spec.metadata["active_params_b"] == 8.2
        assert spec.metadata["family"] == "qwen3"
        assert spec.metadata["license"] == "Apache 2.0"

    def test_moe_model_has_experts(self):
        spec = get_model_spec("qwen3-30b-a3b")
        assert spec.architecture_type == ArchitectureType.MOE_TRANSFORMER
        assert spec.num_experts == 128
        assert spec.experts_per_token == 8
        assert spec.metadata["total_params_b"] == 30.5
        assert spec.metadata["active_params_b"] == 3.3

    def test_gpt_oss_model(self):
        spec = get_model_spec("gpt-oss-120b")
        assert spec.model_id == "openai/gpt-oss-120b"
        assert spec.num_experts == 128
        assert spec.experts_per_token == 4
        assert spec.metadata["native_precision"] == "MXFP4"

    def test_kimi_k2_5(self):
        spec = get_model_spec("kimi-k2.5")
        assert spec.model_id == "moonshotai/Kimi-K2.5"
        assert spec.metadata["total_params_b"] == 1000.0
        assert spec.metadata["active_params_b"] == 32.0
        assert spec.num_experts == 384
        assert spec.metadata["shared_experts"] == 1

    def test_case_insensitive(self):
        spec_lower = get_model_spec("qwen3-8b")
        spec_upper = get_model_spec("Qwen3-8B")
        assert spec_lower.model_id == spec_upper.model_id

    @pytest.mark.parametrize("model_key", ALL_MODEL_KEYS)
    def test_all_models_loadable(self, model_key):
        spec = get_model_spec(model_key)
        assert isinstance(spec, ModelSpec)
        assert spec.model_id  # non-empty
        assert spec.num_layers > 0
        assert spec.hidden_dim > 0

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown model key"):
            get_model_spec("nonexistent-model")

    def test_returns_frozen_modelspec(self):
        spec = get_model_spec("qwen3-8b")
        with pytest.raises(AttributeError):
            spec.hidden_dim = 999


class TestListModels:
    def test_list_all_returns_21(self):
        models = list_models()
        assert len(models) == 21

    def test_list_all_keys_present(self):
        models = list_models()
        keys = {m["model_key"] for m in models}
        assert keys == set(ALL_MODEL_KEYS)

    def test_filter_by_family_qwen3(self):
        models = list_models(family="qwen3")
        assert len(models) == 9  # 6 dense + 2 MoE + 1 Next

    def test_filter_by_family_gpt_oss(self):
        models = list_models(family="gpt-oss")
        assert len(models) == 2

    def test_filter_by_family_glm(self):
        models = list_models(family="glm")
        assert len(models) == 2

    def test_filter_by_arch_dense(self):
        models = list_models(arch="dense_transformer")
        assert len(models) == 6
        for m in models:
            assert m["architecture_type"] == "dense_transformer"

    def test_filter_by_arch_moe(self):
        models = list_models(arch="moe_transformer")
        assert len(models) == 10
        for m in models:
            assert m["architecture_type"] == "moe_transformer"

    def test_filter_by_family_and_arch(self):
        models = list_models(family="qwen3", arch="dense_transformer")
        assert len(models) == 6

    def test_filter_no_results(self):
        models = list_models(family="nonexistent")
        assert models == []

    def test_result_has_expected_keys(self):
        models = list_models()
        for m in models:
            assert "model_key" in m
            assert "hf_id" in m
            assert "family" in m
            assert "architecture_type" in m
            assert "total_params_b" in m
            assert "active_params_b" in m


class TestModelFamiliesIntegrity:
    def test_21_unique_models(self):
        """Verify exactly 21 unique model entries exist in MODEL_FAMILIES."""
        count = 0
        for family_name, family in MODEL_FAMILIES.items():
            for sub_name, sub in family.items():
                if isinstance(sub, dict) and "hf_id" in sub:
                    count += 1
                elif isinstance(sub, dict):
                    for model_key, model_data in sub.items():
                        if isinstance(model_data, dict) and "hf_id" in model_data:
                            count += 1
        assert count == 21

    def test_all_dense_models_have_no_experts(self):
        for model_key in ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b",
                          "qwen3-8b", "qwen3-14b", "qwen3-32b"]:
            spec = get_model_spec(model_key)
            assert spec.num_experts is None
            assert spec.experts_per_token is None
            assert spec.architecture_type == ArchitectureType.DENSE_TRANSFORMER

    def test_all_moe_models_have_experts(self):
        moe_models = list_models(arch="moe_transformer")
        for m in moe_models:
            spec = get_model_spec(m["model_key"])
            assert spec.num_experts is not None and spec.num_experts > 0
            assert spec.experts_per_token is not None and spec.experts_per_token > 0
