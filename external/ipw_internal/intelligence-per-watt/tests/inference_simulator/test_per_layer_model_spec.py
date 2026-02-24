"""Tests for per-layer ModelSpec (LayerConfig and get_layer_config)."""

import pytest

from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    LayerConfig,
    ModelSpec,
)


def _make_model_spec(**kwargs):
    defaults = dict(
        model_id="test/test-7b",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=11008,
        vocab_size=32000,
    )
    defaults.update(kwargs)
    return ModelSpec(**defaults)


class TestLayerConfig:
    def test_creation(self):
        cfg = LayerConfig(
            layer_type="attention",
            hidden_dim=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=11008,
        )
        assert cfg.layer_type == "attention"
        assert cfg.hidden_dim == 4096
        assert cfg.num_experts is None

    def test_frozen(self):
        cfg = LayerConfig(layer_type="ssm", hidden_dim=4096)
        with pytest.raises(AttributeError):
            cfg.layer_type = "attention"

    def test_ssm_layer(self):
        cfg = LayerConfig(
            layer_type="ssm",
            hidden_dim=4096,
            ssm_state_size=16,
        )
        assert cfg.ssm_state_size == 16
        assert cfg.num_attention_heads == 0

    def test_moe_layer(self):
        cfg = LayerConfig(
            layer_type="moe_attention",
            hidden_dim=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=11008,
            num_experts=8,
            experts_per_token=2,
        )
        assert cfg.num_experts == 8
        assert cfg.experts_per_token == 2


class TestModelSpecGetLayerConfig:
    def test_fallback_uniform_dense(self):
        spec = _make_model_spec()
        cfg = spec.get_layer_config(0)
        assert cfg.layer_type == "attention"
        assert cfg.hidden_dim == 4096
        assert cfg.num_attention_heads == 32
        assert cfg.num_kv_heads == 8

    def test_fallback_uniform_ssm(self):
        spec = _make_model_spec(architecture_type=ArchitectureType.SSM_HYBRID)
        cfg = spec.get_layer_config(0)
        assert cfg.layer_type == "ssm"

    def test_fallback_uniform_moe(self):
        spec = _make_model_spec(num_experts=8, experts_per_token=2)
        cfg = spec.get_layer_config(0)
        assert cfg.layer_type == "moe_attention"
        assert cfg.num_experts == 8

    def test_per_layer_configs(self):
        """Hybrid model with alternating attention/SSM layers."""
        layers = tuple(
            LayerConfig(
                layer_type="attention" if i % 2 == 0 else "ssm",
                hidden_dim=4096,
                num_attention_heads=32 if i % 2 == 0 else 0,
                num_kv_heads=8 if i % 2 == 0 else 0,
                head_dim=128 if i % 2 == 0 else 0,
                intermediate_dim=11008,
                ssm_state_size=16 if i % 2 == 1 else None,
            )
            for i in range(4)
        )
        spec = _make_model_spec(num_layers=4, layer_configs=layers)
        assert spec.get_layer_config(0).layer_type == "attention"
        assert spec.get_layer_config(1).layer_type == "ssm"
        assert spec.get_layer_config(2).layer_type == "attention"
        assert spec.get_layer_config(3).layer_type == "ssm"

    def test_out_of_range_index(self):
        layers = (
            LayerConfig(layer_type="attention", hidden_dim=4096),
            LayerConfig(layer_type="ssm", hidden_dim=4096),
        )
        spec = _make_model_spec(num_layers=2, layer_configs=layers)
        with pytest.raises(IndexError):
            spec.get_layer_config(5)

    def test_negative_index_raises(self):
        layers = (
            LayerConfig(layer_type="attention", hidden_dim=4096),
        )
        spec = _make_model_spec(num_layers=1, layer_configs=layers)
        with pytest.raises(IndexError):
            spec.get_layer_config(-1)

    def test_any_layer_index_without_configs(self):
        """Without per-layer configs, any valid layer index works."""
        spec = _make_model_spec(num_layers=32)
        cfg0 = spec.get_layer_config(0)
        cfg31 = spec.get_layer_config(31)
        assert cfg0.hidden_dim == cfg31.hidden_dim
