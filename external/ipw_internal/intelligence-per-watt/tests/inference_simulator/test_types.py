"""Tests for inference_simulator shared types."""

import pytest
from dataclasses import FrozenInstanceError

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.types.results import ProfilingResult
from inference_simulator.types.inference_spec import InferenceSpec
from inference_simulator.types.workload_spec import WorkloadSpec
from inference_simulator.types.sla import SLASpec


class TestOperatorCategory:
    def test_enum_values(self):
        assert OperatorCategory.LINEAR == "linear"
        assert OperatorCategory.ATTENTION_PREFILL == "attention_prefill"
        assert OperatorCategory.ATTENTION_DECODE == "attention_decode"
        assert OperatorCategory.EMBEDDING == "embedding"
        assert OperatorCategory.NORMALIZATION == "normalization"
        assert OperatorCategory.ACTIVATION == "activation"
        assert OperatorCategory.MOE_ROUTING == "moe_routing"
        assert OperatorCategory.AGENTIC_TOOL == "agentic_tool"

    def test_all_categories_present(self):
        expected = {
            "linear", "lm_head", "attention_prefill", "attention_decode",
            "embedding", "normalization", "activation", "moe_routing",
            "moe_expert", "ssm_scan", "communication", "agentic_tool",
            "sampling", "mtp", "cpu_host", "kv_cache",
            "fused_prefill", "fused_decode_step", "fused_attention",
            "fused_mlp", "fused_norm_attn",
        }
        actual = {c.value for c in OperatorCategory}
        assert actual == expected


class TestOperatorMeasurement:
    def test_create_minimal(self):
        m = OperatorMeasurement(
            operator_name="linear_qkv",
            category=OperatorCategory.LINEAR,
            batch_size=8,
            seq_len=256,
            time_s=0.001,
        )
        assert m.operator_name == "linear_qkv"
        assert m.category == OperatorCategory.LINEAR
        assert m.batch_size == 8
        assert m.seq_len == 256
        assert m.time_s == 0.001
        assert m.energy_j is None
        assert m.power_w is None
        assert m.flops is None
        assert m.metadata == {}

    def test_create_full(self):
        m = OperatorMeasurement(
            operator_name="attention_prefill",
            category=OperatorCategory.ATTENTION_PREFILL,
            batch_size=4,
            seq_len=1024,
            time_s=0.005,
            energy_j=1.5,
            power_w=300.0,
            flops=1_000_000_000,
            bytes_accessed=500_000_000,
            bandwidth_gb_s=100.0,
            metadata={"variant": "flash"},
        )
        assert m.energy_j == 1.5
        assert m.power_w == 300.0
        assert m.metadata["variant"] == "flash"

    def test_tflops_property(self):
        m = OperatorMeasurement(
            operator_name="test",
            category=OperatorCategory.LINEAR,
            batch_size=1,
            seq_len=1,
            time_s=0.001,
            flops=1_000_000_000_000,  # 1 TFLOP
        )
        assert m.tflops == pytest.approx(1000.0)

    def test_tflops_none_when_no_flops(self):
        m = OperatorMeasurement(
            operator_name="test",
            category=OperatorCategory.LINEAR,
            batch_size=1,
            seq_len=1,
            time_s=0.001,
        )
        assert m.tflops is None

    def test_arithmetic_intensity(self):
        m = OperatorMeasurement(
            operator_name="test",
            category=OperatorCategory.LINEAR,
            batch_size=1,
            seq_len=1,
            time_s=0.001,
            flops=1000,
            bytes_accessed=100,
        )
        assert m.arithmetic_intensity == pytest.approx(10.0)


class TestModelSpec:
    @pytest.fixture
    def qwen3_spec(self):
        return ModelSpec(
            model_id="Qwen/Qwen3-8B",
            architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.GQA,
            num_layers=36,
            hidden_dim=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=11008,
            vocab_size=151936,
        )

    def test_frozen_immutable(self, qwen3_spec):
        with pytest.raises(FrozenInstanceError):
            qwen3_spec.hidden_dim = 8192

    def test_fields(self, qwen3_spec):
        assert qwen3_spec.model_id == "Qwen/Qwen3-8B"
        assert qwen3_spec.architecture_type == ArchitectureType.DENSE_TRANSFORMER
        assert qwen3_spec.attention_type == AttentionType.GQA
        assert qwen3_spec.num_layers == 36
        assert qwen3_spec.hidden_dim == 4096
        assert qwen3_spec.num_attention_heads == 32
        assert qwen3_spec.num_kv_heads == 8
        assert qwen3_spec.head_dim == 128
        assert qwen3_spec.intermediate_dim == 11008
        assert qwen3_spec.vocab_size == 151936

    def test_defaults(self, qwen3_spec):
        assert qwen3_spec.max_seq_len == 131072
        assert qwen3_spec.num_experts is None
        assert qwen3_spec.experts_per_token is None
        assert qwen3_spec.tie_word_embeddings is False
        assert qwen3_spec.metadata == {}

    def test_kv_head_ratio(self, qwen3_spec):
        assert qwen3_spec.kv_head_ratio == pytest.approx(0.25)

    def test_total_params(self, qwen3_spec):
        # Should be roughly 8B params
        params = qwen3_spec.total_params_billion
        assert 6.0 < params < 10.0


class TestArchitectureType:
    def test_enum_values(self):
        assert ArchitectureType.DENSE_TRANSFORMER == "dense_transformer"
        assert ArchitectureType.MOE_TRANSFORMER == "moe_transformer"
        assert ArchitectureType.SSM_HYBRID == "ssm_hybrid"
        assert ArchitectureType.LINEAR_ATTENTION == "linear_attention"


class TestAttentionType:
    def test_enum_values(self):
        assert AttentionType.MHA == "mha"
        assert AttentionType.MQA == "mqa"
        assert AttentionType.GQA == "gqa"


class TestHardwareSpec:
    def test_create(self):
        hw = HardwareSpec(
            name="NVIDIA A100 80GB SXM",
            vendor="nvidia",
            memory_gb=80,
            tdp_watts=400,
            peak_fp16_tflops=312.0,
        )
        assert hw.name == "NVIDIA A100 80GB SXM"
        assert hw.peak_tflops == 312.0  # FP8 is 0, falls back to FP16

    def test_from_registry(self):
        hw = HardwareSpec.from_registry("a100_80gb")
        assert hw.name == "NVIDIA A100 80GB SXM"
        assert hw.vendor == "nvidia"
        assert hw.memory_gb == 80
        assert hw.tdp_watts == 400
        assert hw.peak_fp16_tflops == 312.0

    def test_from_ipw_specs(self):
        from ipw.simulator.hardware_specs import get_hardware_specs
        ipw_hw = get_hardware_specs("h100_80gb")
        hw = HardwareSpec.from_ipw_specs(ipw_hw, cpu_model="AMD EPYC")
        assert hw.name == "NVIDIA H100 80GB SXM"
        assert hw.peak_fp8_tflops == 1978.9
        assert hw.cpu_model == "AMD EPYC"

    def test_frozen_immutable(self):
        hw = HardwareSpec(
            name="test", vendor="test", memory_gb=80,
            tdp_watts=400, peak_fp16_tflops=312.0,
        )
        with pytest.raises(FrozenInstanceError):
            hw.memory_gb = 160

    def test_peak_tflops_prefers_fp8(self):
        hw = HardwareSpec(
            name="test", vendor="nvidia", memory_gb=80,
            tdp_watts=700, peak_fp16_tflops=989.4,
            peak_fp8_tflops=1978.9,
        )
        assert hw.peak_tflops == 1978.9
        assert hw.bytes_per_param == 1.0


class TestProfilingResult:
    def test_create(self):
        spec = ModelSpec(
            model_id="test", architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.MHA, num_layers=12, hidden_dim=768,
            num_attention_heads=12, num_kv_heads=12, head_dim=64,
            intermediate_dim=3072, vocab_size=32000,
        )
        hw = HardwareSpec(
            name="test", vendor="nvidia", memory_gb=80,
            tdp_watts=400, peak_fp16_tflops=312.0,
        )
        result = ProfilingResult(
            model_spec=spec, hardware_spec=hw,
            precision="fp16", timestamp="2024-01-01T00:00:00Z",
        )
        assert result.num_measurements == 0

    def test_filter_by_category(self):
        spec = ModelSpec(
            model_id="test", architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.MHA, num_layers=12, hidden_dim=768,
            num_attention_heads=12, num_kv_heads=12, head_dim=64,
            intermediate_dim=3072, vocab_size=32000,
        )
        hw = HardwareSpec(
            name="test", vendor="nvidia", memory_gb=80,
            tdp_watts=400, peak_fp16_tflops=312.0,
        )
        m1 = OperatorMeasurement(
            operator_name="linear", category=OperatorCategory.LINEAR,
            batch_size=1, seq_len=1, time_s=0.001,
        )
        m2 = OperatorMeasurement(
            operator_name="attn", category=OperatorCategory.ATTENTION_PREFILL,
            batch_size=1, seq_len=1, time_s=0.002,
        )
        result = ProfilingResult(
            model_spec=spec, hardware_spec=hw,
            precision="fp16", timestamp="2024-01-01T00:00:00Z",
            measurements=[m1, m2],
        )
        linear_only = result.filter_by_category(OperatorCategory.LINEAR)
        assert len(linear_only) == 1
        assert linear_only[0].operator_name == "linear"


class TestStubs:
    def test_inference_spec_exists(self):
        spec = InferenceSpec()
        assert spec.num_gpus == 1
        assert spec.precision == "fp16"

    def test_workload_spec_exists(self):
        spec = WorkloadSpec()
        assert spec.qps == 1.0
        assert spec.avg_input_tokens == 500

    def test_sla_spec_exists(self):
        spec = SLASpec()
        assert spec.max_ttft_s == 1.0
        assert spec.min_throughput_tps == 10.0
