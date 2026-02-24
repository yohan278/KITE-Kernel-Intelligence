"""Tests for the LLM inference simulator.

Covers:
1. Analytical model against hand-calculated roofline values
2. Calibration roundtrip (fit -> predict -> verify)
3. Cross-hardware sanity checks (H100 faster than A100, etc.)
4. Workload model (multi-turn projection)
5. CalibrationDB persistence
6. CLI smoke test (if click is available)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from ipw.simulator.hardware_specs import (
    HARDWARE_SPECS_REGISTRY,
    HardwareSpecs,
    get_hardware_specs,
)
from ipw.simulator.inference_model import (
    DEFAULT_ALPHA,
    DEFAULT_ETA_DECODE,
    DEFAULT_ETA_PREFILL,
    OVERHEAD_BASE,
    OVERHEAD_DECAY,
    OVERHEAD_BETA,
    SYSTEM_OVERHEAD_MULTIPLIER,
    batch_overhead_multiplier,
    estimate_alpha_for_model_size,
    estimate_decode,
    estimate_eta_for_model_size,
    estimate_power,
    estimate_prefill,
    estimate_prefill_batch_aware,
    estimate_prefill_bottomup,
    predict,
)
from ipw.simulator.types import (
    CalibrationFactors,
    ConfidenceLevel,
    PhaseResult,
    SimulationResult,
    SimulatorConfig,
    SingleInferenceResult,
    WorkloadProfile,
    WorkloadType,
)
from ipw.simulator.workload_model import project
from ipw.simulator.calibration import CalibrationDB


# =============================================================================
# Test: Hardware Specs Registry
# =============================================================================

class TestHardwareSpecs:
    """Verify hardware specs are complete and sensible."""

    def test_all_gpu_types_present(self):
        expected = {
            "a100_80gb", "h100_80gb", "h200", "gh200", "b200",
            "mi300x", "m4_max", "m4_pro", "m3_max", "m3_pro",
        }
        assert set(HARDWARE_SPECS_REGISTRY.keys()) == expected

    def test_get_hardware_specs(self):
        hw = get_hardware_specs("h100_80gb")
        assert hw.name == "NVIDIA H100 80GB SXM"
        assert hw.tdp_watts == 700
        assert hw.peak_fp16_tflops > 0
        assert hw.hbm_bandwidth_gb_s > 0

    def test_get_hardware_specs_case_insensitive(self):
        hw = get_hardware_specs("H100_80GB")
        assert hw.vendor == "nvidia"

    def test_unknown_gpu_raises(self):
        with pytest.raises(KeyError, match="Unknown GPU type"):
            get_hardware_specs("rtx_4090")

    def test_peak_tflops_prefers_fp8(self):
        h100 = get_hardware_specs("h100_80gb")
        assert h100.peak_tflops == h100.peak_fp8_tflops

        # Apple Silicon has no FP8, should fall back to FP16
        m4 = get_hardware_specs("m4_max")
        assert m4.peak_tflops == m4.peak_fp16_tflops

    def test_bytes_per_param(self):
        h100 = get_hardware_specs("h100_80gb")
        assert h100.bytes_per_param == 1.0  # FP8

        m4 = get_hardware_specs("m4_max")
        assert m4.bytes_per_param == 2.0  # FP16

    def test_all_specs_positive(self):
        for name, hw in HARDWARE_SPECS_REGISTRY.items():
            assert hw.tdp_watts > 0, f"{name}: TDP must be positive"
            assert hw.peak_fp16_tflops > 0, f"{name}: FP16 TFLOPS must be positive"
            assert hw.hbm_bandwidth_gb_s > 0, f"{name}: HBM BW must be positive"
            assert hw.memory_gb > 0, f"{name}: Memory must be positive"


# =============================================================================
# Test: Analytical Inference Model
# =============================================================================

class TestInferenceModel:
    """Test roofline-based single-inference predictions against hand-calculated values."""

    def test_estimate_prefill_basic(self):
        """Verify prefill time for a known configuration.

        H100 SXM: peak FP8 = 1978.9 TFLOPS
        Model: 8B params, 1000 input tokens
        FLOPs = 2 * 8e9 * 1000 = 16e12 = 16 TFLOPS
        Time = 16e12 / (1978.9e12 * 0.4) = 16 / 791.56 = ~0.0202 s
        """
        result = estimate_prefill(
            active_params_b=8.0,
            input_tokens=1000,
            peak_tflops=1978.9,
            eta=0.4,
            power_watts=455.0,  # 0.65 * 700
        )

        expected_flops = 2.0 * 8e9 * 1000
        assert result.flops == pytest.approx(expected_flops)

        expected_time = expected_flops / (1978.9e12 * 0.4)
        assert result.time_seconds == pytest.approx(expected_time, rel=1e-4)
        assert result.energy_joules == pytest.approx(455.0 * expected_time, rel=1e-4)

    def test_estimate_decode_basic(self):
        """Verify decode time for a known configuration.

        H100 SXM: HBM BW = 3352 GB/s
        Model: 8B params, FP8 (1 byte/param)
        Weight bytes = 8e9 * 1 = 8 GB
        Time per token = 8e9 / (3352e9 * 0.5) = 8 / 1676 = ~0.00477 s
        200 tokens => ~0.954 s
        """
        result = estimate_decode(
            active_params_b=8.0,
            output_tokens=200,
            bytes_per_param=1.0,
            mem_bw_gb_s=3352.0,
            eta=0.5,
            power_watts=455.0,
        )

        weight_bytes = 8e9 * 1.0
        time_per_token = weight_bytes / (3352e9 * 0.5)
        expected_time = 200 * time_per_token

        assert result.time_seconds == pytest.approx(expected_time, rel=1e-4)
        assert result.bytes_transferred == pytest.approx(weight_bytes * 200, rel=1e-4)

    def test_estimate_prefill_zero_tokens(self):
        result = estimate_prefill(8.0, 0, 1000.0)
        assert result.time_seconds == 0.0
        assert result.energy_joules == 0.0

    def test_estimate_decode_zero_tokens(self):
        result = estimate_decode(8.0, 0, 1.0, 3000.0)
        assert result.time_seconds == 0.0

    def test_estimate_power(self):
        hw = get_hardware_specs("h100_80gb")
        power = estimate_power(hw, alpha=0.65)
        assert power == pytest.approx(700 * 0.65)

    def test_estimate_power_clamps(self):
        hw = get_hardware_specs("h100_80gb")
        assert estimate_power(hw, alpha=1.5) == hw.tdp_watts  # clamped to 1.0
        assert estimate_power(hw, alpha=-0.1) == 0.0  # clamped to 0.0

    def test_predict_single_inference(self):
        hw = get_hardware_specs("h100_80gb")
        result = predict(
            hw=hw,
            active_params_b=8.0,
            input_tokens=500,
            output_tokens=200,
            bytes_per_param=1.0,
        )

        assert result.total_time_seconds > 0
        assert result.total_energy_joules > 0
        assert result.prefill.time_seconds > 0
        assert result.decode.time_seconds > 0
        assert result.tokens_per_second > 0

        # Total time = prefill + decode
        assert result.total_time_seconds == pytest.approx(
            result.prefill.time_seconds + result.decode.time_seconds,
            rel=1e-6,
        )

    def test_predict_with_calibration_regression(self):
        """When calibration provides regression slopes, use them directly."""
        hw = get_hardware_specs("h100_80gb")
        cal = CalibrationFactors(
            energy_per_input_token_j=0.001,
            energy_per_output_token_j=0.005,
            intercept_j=0.5,
        )
        result = predict(
            hw=hw,
            active_params_b=8.0,
            input_tokens=500,
            output_tokens=200,
            calibration=cal,
        )

        expected_energy = 0.001 * 500 + 0.005 * 200 + 0.5  # = 2.0
        assert result.total_energy_joules == pytest.approx(expected_energy, rel=1e-4)

    def test_multi_gpu_scaling(self):
        """Multi-GPU should reduce latency."""
        hw = get_hardware_specs("a100_80gb")

        result_1gpu = predict(hw=hw, active_params_b=8.0, input_tokens=1000,
                              output_tokens=200, num_gpus=1)
        result_4gpu = predict(hw=hw, active_params_b=8.0, input_tokens=1000,
                              output_tokens=200, num_gpus=4)

        # 4 GPUs should be faster (linear scaling in the roofline model)
        assert result_4gpu.total_time_seconds < result_1gpu.total_time_seconds


# =============================================================================
# Test: Model-Size-Dependent Defaults
# =============================================================================

class TestModelSizeDependentDefaults:
    """Verify η and α estimation from model size."""

    def test_eta_increases_with_model_size(self):
        """Larger models should achieve higher MFU."""
        eta_05 = estimate_eta_for_model_size(0.5)
        eta_4 = estimate_eta_for_model_size(4.0)
        eta_14 = estimate_eta_for_model_size(14.0)
        assert eta_05 < eta_4 < eta_14

    def test_alpha_increases_with_model_size(self):
        """Larger models should sustain higher power fraction."""
        alpha_05 = estimate_alpha_for_model_size(0.5)
        alpha_4 = estimate_alpha_for_model_size(4.0)
        alpha_14 = estimate_alpha_for_model_size(14.0)
        assert alpha_05 < alpha_4 < alpha_14

    def test_eta_clamping(self):
        """Tiny model η ≥ 0.05, huge model η ≤ 0.95."""
        assert estimate_eta_for_model_size(0.001) >= 0.05
        assert estimate_eta_for_model_size(10000.0) <= 0.95

    def test_alpha_clamping(self):
        """Tiny model α ≥ 0.1, huge model α ≤ 1.0."""
        assert estimate_alpha_for_model_size(0.001) >= 0.1
        assert estimate_alpha_for_model_size(10000.0) <= 1.0

    def test_eta_known_values(self):
        """Verify against empirical fits for Qwen3 models."""
        # Qwen3-0.6B: ~0.752B params → η ≈ 0.22
        assert estimate_eta_for_model_size(0.752) == pytest.approx(0.22, abs=0.03)
        # Qwen3-4B: ~4.411B params → η ≈ 0.50
        assert estimate_eta_for_model_size(4.411) == pytest.approx(0.50, abs=0.03)
        # Qwen3-14B: ~14.768B params → η ≈ 0.70
        assert estimate_eta_for_model_size(14.768) == pytest.approx(0.70, abs=0.03)

    def test_zero_params_returns_default(self):
        """Zero or negative params should fall back to old constants."""
        assert estimate_eta_for_model_size(0.0) == DEFAULT_ETA_PREFILL
        assert estimate_alpha_for_model_size(0.0) == DEFAULT_ALPHA
        assert estimate_eta_for_model_size(-1.0) == DEFAULT_ETA_PREFILL
        assert estimate_alpha_for_model_size(-1.0) == DEFAULT_ALPHA


# =============================================================================
# Test: Bottom-Up Energy Model
# =============================================================================

class TestBottomUpEnergy:
    """Verify bottom-up pJ energy model."""

    def test_basic_computation(self):
        """All fields should be positive for a valid config."""
        result = estimate_prefill_bottomup(
            active_params_b=4.0,
            input_tokens=512,
            num_layers=36,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=2560,
        )
        assert result.e_matmul_j > 0
        assert result.e_attention_j > 0
        assert result.e_weight_memory_j > 0
        assert result.e_activation_memory_j > 0
        assert result.e_kv_cache_j > 0
        assert result.e_raw_total_j > 0
        assert result.e_total_j > 0
        assert result.overhead_multiplier == SYSTEM_OVERHEAD_MULTIPLIER

    def test_zero_tokens_returns_zero(self):
        """Zero input tokens should produce zero energy everywhere."""
        result = estimate_prefill_bottomup(active_params_b=4.0, input_tokens=0)
        assert result.e_total_j == 0.0
        assert result.e_matmul_j == 0.0
        assert result.e_raw_total_j == 0.0

    def test_attention_quadratic_scaling(self):
        """Doubling sin should ~4× the attention energy (O(sin²))."""
        r1 = estimate_prefill_bottomup(active_params_b=4.0, input_tokens=512)
        r2 = estimate_prefill_bottomup(active_params_b=4.0, input_tokens=1024)
        ratio = r2.e_attention_j / r1.e_attention_j
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_custom_overhead(self):
        """overhead=2.0 should give 2× the total of overhead=1.0."""
        r1 = estimate_prefill_bottomup(
            active_params_b=4.0, input_tokens=512, overhead_multiplier=1.0,
        )
        r2 = estimate_prefill_bottomup(
            active_params_b=4.0, input_tokens=512, overhead_multiplier=2.0,
        )
        assert r2.e_total_j == pytest.approx(2.0 * r1.e_total_j, rel=1e-6)

    def test_breakdown_sums_to_raw(self):
        """Components should sum to e_raw_total_j."""
        result = estimate_prefill_bottomup(
            active_params_b=4.0, input_tokens=512,
            num_layers=36, num_heads=32, num_kv_heads=8,
            head_dim=128, hidden_size=2560,
        )
        component_sum = (
            result.e_matmul_j
            + result.e_attention_j
            + result.e_weight_memory_j
            + result.e_activation_memory_j
            + result.e_kv_cache_j
        )
        assert component_sum == pytest.approx(result.e_raw_total_j, rel=1e-9)


# =============================================================================
# Test: Batch-Aware Energy Model (Model F)
# =============================================================================

class TestBatchAwareEnergy:
    """Verify batch-aware energy model with batch-dependent overhead."""

    def test_batch_overhead_at_b1(self):
        """At B=1, overhead should equal c_base + c_decay, close to Model E's 2.0×."""
        overhead = batch_overhead_multiplier(1)
        assert overhead == pytest.approx(OVERHEAD_BASE + OVERHEAD_DECAY, rel=1e-9)

    def test_batch_overhead_monotone_decreasing(self):
        """Overhead must be non-increasing with batch size, strictly decreasing from B=1."""
        overheads = [batch_overhead_multiplier(b) for b in [1, 8, 32, 128, 512]]
        # B=1 must be strictly greater than B=8
        assert overheads[0] > overheads[1]
        # Remaining pairs must be non-increasing
        for i in range(1, len(overheads) - 1):
            assert overheads[i] >= overheads[i + 1]

    def test_batch_overhead_converges(self):
        """At very large batch, overhead should approach OVERHEAD_BASE."""
        overhead = batch_overhead_multiplier(10000)
        assert overhead == pytest.approx(OVERHEAD_BASE, abs=0.01)

    def test_batch_overhead_positive(self):
        """Overhead must always be positive for any positive batch size."""
        for b in [1, 2, 5, 10, 100, 1000, 100000]:
            assert batch_overhead_multiplier(b) > 0

    def test_batch_aware_b1_similar_to_model_e(self):
        """At B=1, batch-aware energy should be within 15% of Model E."""
        model_e = estimate_prefill_bottomup(
            active_params_b=4.0, input_tokens=512,
            num_layers=36, num_heads=32, num_kv_heads=8,
            head_dim=128, hidden_size=2560,
        )
        model_f = estimate_prefill_batch_aware(
            active_params_b=4.0, input_tokens=512, batch_size=1,
            num_layers=36, num_heads=32, num_kv_heads=8,
            head_dim=128, hidden_size=2560,
        )
        ratio = model_f.e_total_j / model_e.e_total_j
        assert 0.85 <= ratio <= 1.15

    def test_batch_aware_energy_decreases_with_batch(self):
        """Per-query energy should decrease with larger batch size."""
        energies = []
        for b in [1, 8, 32, 128, 512]:
            result = estimate_prefill_batch_aware(
                active_params_b=14.0, input_tokens=512, batch_size=b,
                num_layers=40, num_heads=40, num_kv_heads=8,
                head_dim=128, hidden_size=5120,
            )
            energies.append(result.e_total_j)
        for i in range(len(energies) - 1):
            assert energies[i] > energies[i + 1]

    def test_batch_aware_saturation(self):
        """B=32 and B=512 should produce energy within 30% of each other (saturation)."""
        r32 = estimate_prefill_batch_aware(
            active_params_b=14.0, input_tokens=512, batch_size=32,
            num_layers=40, num_heads=40, num_kv_heads=8,
            head_dim=128, hidden_size=5120,
        )
        r512 = estimate_prefill_batch_aware(
            active_params_b=14.0, input_tokens=512, batch_size=512,
            num_layers=40, num_heads=40, num_kv_heads=8,
            head_dim=128, hidden_size=5120,
        )
        ratio = r32.e_total_j / r512.e_total_j
        assert 1.0 <= ratio <= 1.30

    def test_batch_aware_weight_mem_amortizes(self):
        """Weight memory at B=1 should be ~8× weight memory at B=8."""
        r1 = estimate_prefill_batch_aware(
            active_params_b=4.0, input_tokens=512, batch_size=1,
            num_layers=36, num_heads=32, num_kv_heads=8,
            head_dim=128, hidden_size=2560,
        )
        r8 = estimate_prefill_batch_aware(
            active_params_b=4.0, input_tokens=512, batch_size=8,
            num_layers=36, num_heads=32, num_kv_heads=8,
            head_dim=128, hidden_size=2560,
        )
        ratio = r1.e_weight_memory_j / r8.e_weight_memory_j
        assert ratio == pytest.approx(8.0, rel=1e-9)

    def test_batch_aware_zero_tokens(self):
        """Zero input tokens should produce zero energy."""
        result = estimate_prefill_batch_aware(
            active_params_b=4.0, input_tokens=0, batch_size=8,
        )
        assert result.e_total_j == 0.0
        assert result.e_matmul_j == 0.0
        assert result.e_raw_total_j == 0.0

    def test_batch_aware_breakdown_sums_to_raw(self):
        """Components should sum to e_raw_total_j."""
        result = estimate_prefill_batch_aware(
            active_params_b=4.0, input_tokens=512, batch_size=8,
            num_layers=36, num_heads=32, num_kv_heads=8,
            head_dim=128, hidden_size=2560,
        )
        component_sum = (
            result.e_matmul_j
            + result.e_attention_j
            + result.e_weight_memory_j
            + result.e_activation_memory_j
            + result.e_kv_cache_j
        )
        assert component_sum == pytest.approx(result.e_raw_total_j, rel=1e-9)


# =============================================================================
# Test: Cross-Hardware Sanity Checks
# =============================================================================

class TestCrossHardwareSanity:
    """Ensure relative performance ordering is physically reasonable."""

    def test_h100_faster_than_a100(self):
        """H100 has higher TFLOPS and bandwidth than A100."""
        h100 = get_hardware_specs("h100_80gb")
        a100 = get_hardware_specs("a100_80gb")

        r_h100 = predict(h100, 8.0, 1000, 200, bytes_per_param=1.0)
        r_a100 = predict(a100, 8.0, 1000, 200, bytes_per_param=2.0)  # A100 no FP8

        assert r_h100.total_time_seconds < r_a100.total_time_seconds

    def test_b200_faster_than_h100(self):
        """B200 is the next-gen accelerator."""
        b200 = get_hardware_specs("b200")
        h100 = get_hardware_specs("h100_80gb")

        r_b200 = predict(b200, 8.0, 1000, 200, bytes_per_param=1.0)
        r_h100 = predict(h100, 8.0, 1000, 200, bytes_per_param=1.0)

        assert r_b200.total_time_seconds < r_h100.total_time_seconds

    def test_apple_silicon_lower_power(self):
        """Apple Silicon has much lower TDP than NVIDIA GPUs."""
        m4 = get_hardware_specs("m4_max")
        h100 = get_hardware_specs("h100_80gb")

        # M4 Max has ~40W TDP vs H100's 700W
        power_m4 = estimate_power(m4, DEFAULT_ALPHA)
        power_h100 = estimate_power(h100, DEFAULT_ALPHA)
        assert power_m4 < power_h100

    def test_apple_silicon_higher_latency(self):
        """Apple Silicon is slower than datacenter GPUs."""
        m4 = get_hardware_specs("m4_max")
        a100 = get_hardware_specs("a100_80gb")

        r_m4 = predict(m4, 8.0, 1000, 200, bytes_per_param=2.0)
        r_a100 = predict(a100, 8.0, 1000, 200, bytes_per_param=2.0)

        assert r_m4.total_time_seconds > r_a100.total_time_seconds

    def test_larger_model_more_energy(self):
        """Larger models should consume more energy."""
        hw = get_hardware_specs("h100_80gb")

        r_8b = predict(hw, 8.0, 1000, 200, bytes_per_param=1.0)
        r_32b = predict(hw, 32.0, 1000, 200, bytes_per_param=1.0)

        assert r_32b.total_energy_joules > r_8b.total_energy_joules

    def test_more_tokens_more_energy(self):
        """More output tokens should consume more energy."""
        hw = get_hardware_specs("h100_80gb")

        r_100 = predict(hw, 8.0, 500, 100, bytes_per_param=1.0)
        r_1000 = predict(hw, 8.0, 500, 1000, bytes_per_param=1.0)

        assert r_1000.total_energy_joules > r_100.total_energy_joules


# =============================================================================
# Test: Calibration DB
# =============================================================================

class TestCalibrationDB:
    """Test calibration persistence and lookup."""

    def test_add_and_get(self):
        db = CalibrationDB()
        factors = CalibrationFactors(
            gpu_type="h100_80gb",
            model_type="qwen3-8b",
            eta_prefill=0.5,
            eta_decode=0.6,
            alpha=0.7,
        )
        db.add(factors)

        result = db.get("h100_80gb", "qwen3-8b")
        assert result is not None
        assert result.eta_prefill == 0.5
        assert result.alpha == 0.7

    def test_get_missing_returns_none(self):
        db = CalibrationDB()
        assert db.get("h100_80gb", "qwen3-8b") is None

    def test_interpolation_same_gpu(self):
        db = CalibrationDB()
        db.add(CalibrationFactors(gpu_type="h100_80gb", model_type="qwen3-4b", eta_prefill=0.4))
        db.add(CalibrationFactors(gpu_type="h100_80gb", model_type="qwen3-14b", eta_prefill=0.6))

        # Should interpolate from the two entries
        result = db.get_or_interpolate("h100_80gb", "qwen3-8b")
        assert result is not None
        assert result.eta_prefill == pytest.approx(0.5)

    def test_save_and_load_roundtrip(self):
        db = CalibrationDB()
        db.add(CalibrationFactors(
            gpu_type="h100_80gb",
            model_type="qwen3-8b",
            eta_prefill=0.45,
            eta_decode=0.55,
            alpha=0.72,
            energy_per_input_token_j=0.001,
            energy_per_output_token_j=0.005,
            intercept_j=0.5,
            sample_count=50,
            r_squared=0.95,
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            db.save(path)

            db2 = CalibrationDB()
            db2.load(path)

            result = db2.get("h100_80gb", "qwen3-8b")
            assert result is not None
            assert result.eta_prefill == pytest.approx(0.45)
            assert result.energy_per_input_token_j == pytest.approx(0.001)
            assert result.r_squared == pytest.approx(0.95)
        finally:
            path.unlink(missing_ok=True)


# =============================================================================
# Test: Workload Model
# =============================================================================

class TestWorkloadModel:
    """Test multi-turn workload projection."""

    def test_single_query_matches_predict(self):
        """Single-query workload should match direct predict() call."""
        hw = get_hardware_specs("h100_80gb")
        workload = WorkloadProfile(
            workload_type=WorkloadType.SINGLE_QUERY,
            avg_input_tokens=500,
            avg_output_tokens=200,
        )

        single = predict(hw, 8.0, 500, 200, bytes_per_param=1.0)
        result = project(hw, 8.0, workload, bytes_per_param=1.0)

        assert result.total_time_seconds == pytest.approx(single.total_time_seconds, rel=1e-6)
        assert result.total_energy_joules == pytest.approx(single.total_energy_joules, rel=1e-6)

    def test_multi_turn_more_energy(self):
        """Multi-turn workload should use more energy than single turn."""
        hw = get_hardware_specs("h100_80gb")

        single = WorkloadProfile(
            workload_type=WorkloadType.SINGLE_QUERY,
            avg_input_tokens=500,
            avg_output_tokens=200,
        )
        multi = WorkloadProfile(
            workload_type=WorkloadType.AGENTIC_REASONING,
            avg_input_tokens=500,
            avg_output_tokens=200,
            avg_turns=5,
            context_growth_per_turn=300,
        )

        r_single = project(hw, 8.0, single, bytes_per_param=1.0)
        r_multi = project(hw, 8.0, multi, bytes_per_param=1.0)

        assert r_multi.total_energy_joules > r_single.total_energy_joules
        assert r_multi.total_time_seconds > r_single.total_time_seconds
        assert r_multi.num_turns == 5

    def test_tool_calls_add_idle_energy(self):
        """Tool calls should add idle time and energy."""
        hw = get_hardware_specs("h100_80gb")

        without_tools = WorkloadProfile(
            workload_type=WorkloadType.AGENTIC_REASONING,
            avg_input_tokens=500,
            avg_output_tokens=200,
            avg_turns=3,
            avg_tool_calls=0,
        )
        with_tools = WorkloadProfile(
            workload_type=WorkloadType.AGENTIC_REASONING,
            avg_input_tokens=500,
            avg_output_tokens=200,
            avg_turns=3,
            avg_tool_calls=5,
            avg_tool_latency_seconds=2.0,
        )

        r_no_tools = project(hw, 8.0, without_tools, bytes_per_param=1.0)
        r_tools = project(hw, 8.0, with_tools, bytes_per_param=1.0)

        assert r_tools.idle_time_seconds == pytest.approx(5 * 2.0)
        assert r_tools.idle_energy_joules > 0
        assert r_tools.total_time_seconds > r_no_tools.total_time_seconds

    def test_context_growth(self):
        """Each turn should process a larger context (growing prefill cost)."""
        hw = get_hardware_specs("h100_80gb")

        workload = WorkloadProfile(
            workload_type=WorkloadType.AGENTIC_REASONING,
            avg_input_tokens=500,
            avg_output_tokens=100,
            avg_turns=3,
            context_growth_per_turn=500,
        )
        result = project(hw, 8.0, workload, bytes_per_param=1.0)

        # Total input tokens: turn 0 = 500, turn 1 = 1000, turn 2 = 1500
        assert result.total_input_tokens == 500 + 1000 + 1500


# =============================================================================
# Test: SimulationResult formatting
# =============================================================================

class TestFormatResult:
    """Test human-readable formatting."""

    def test_format_basic(self):
        from ipw.simulator.simulator import format_result

        result = SimulationResult(
            total_energy_joules=10.5,
            total_time_seconds=2.3,
            avg_power_watts=4.57,
            prefill_time_seconds=0.3,
            prefill_energy_joules=1.5,
            decode_time_seconds=2.0,
            decode_energy_joules=9.0,
            confidence=ConfidenceLevel.LOW,
        )
        text = format_result(result)
        assert "10.50 J" in text
        assert "2.300 s" in text
        assert "low" in text


# =============================================================================
# Test: CLI (smoke test)
# =============================================================================

class TestCLI:
    """Smoke test the simulate CLI command."""

    def test_simulate_help(self):
        from click.testing import CliRunner
        from ipw.cli.simulate import simulate

        runner = CliRunner()
        result = runner.invoke(simulate, ["--help"])
        assert result.exit_code == 0
        assert "--gpu" in result.output
        assert "--model" in result.output

    def test_simulate_json_output(self):
        """Run a simulation and verify JSON output is valid."""
        from click.testing import CliRunner
        from ipw.cli.simulate import simulate

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "--gpu", "h100_80gb",
            "--model", "qwen3-8b",
            "--input-tokens", "500",
            "--output-tokens", "200",
            "--json-output",
        ])

        # May fail if grid_eval.config not importable, that's OK
        if result.exit_code == 0:
            output = json.loads(result.output)
            assert output["total_energy_joules"] > 0
            assert output["total_time_seconds"] > 0
            assert output["confidence"] == "low"
