"""Tests for inference-level energy microbenchmarks.

Covers:
1. WorkloadType has new inference enum values
2. EnergyParameters serialization roundtrip with inference fields
3. Backward compatibility: from_dict() without "inference" key -> defaults
4. Extraction functions with synthetic BenchmarkResult data
5. InferenceCharacterizationConfig defaults
6. Abstract base class hierarchy
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ipw.benchmarks.types import (
    BenchmarkResult,
    DataType,
    EnergyMeasurement,
    EnergyParameters,
    Platform,
    WorkloadConfig,
    WorkloadResult,
    WorkloadType,
)
from ipw.benchmarks.runner import InferenceCharacterizationConfig
from ipw.benchmarks.workloads.base import (
    AttentionWorkload,
    BatchedDecodeWorkload,
    InferenceGEMMWorkload,
    KVCacheWorkload,
    NCCLCollectiveWorkload,
)


# ---------------------------------------------------------------------------
# WorkloadType enum tests
# ---------------------------------------------------------------------------

class TestWorkloadType:
    """Test that new inference WorkloadType values exist."""

    def test_inference_gemm(self):
        assert WorkloadType.INFERENCE_GEMM.value == "inference_gemm"

    def test_attention(self):
        assert WorkloadType.ATTENTION.value == "attention"

    def test_kv_cache_io(self):
        assert WorkloadType.KV_CACHE_IO.value == "kv_cache_io"

    def test_nccl_collective(self):
        assert WorkloadType.NCCL_COLLECTIVE.value == "nccl_collective"

    def test_batched_decode(self):
        assert WorkloadType.BATCHED_DECODE.value == "batched_decode"

    def test_all_original_types_preserved(self):
        """Ensure original types still work."""
        assert WorkloadType.MEMORY_BANDWIDTH.value == "memory_bandwidth"
        assert WorkloadType.COMPUTE_BOUND.value == "compute_bound"
        assert WorkloadType.GEMM.value == "gemm"
        assert WorkloadType.CACHE_L1.value == "cache_l1"
        assert WorkloadType.CACHE_L2.value == "cache_l2"
        assert WorkloadType.HBM_BANDWIDTH.value == "hbm_bandwidth"


# ---------------------------------------------------------------------------
# EnergyParameters serialization tests
# ---------------------------------------------------------------------------

class TestEnergyParametersSerialization:
    """Test EnergyParameters roundtrip with inference fields."""

    def test_roundtrip_with_inference_fields(self):
        """Full roundtrip: create -> to_dict -> from_dict -> verify."""
        params = EnergyParameters(
            platform=Platform.NVIDIA,
            hardware_name="NVIDIA A100",
            e_inference_gemm_prefill_pj_per_flop={DataType.FP16: 1.5, DataType.FP32: 3.0},
            e_inference_gemm_decode_pj_per_flop={DataType.FP16: 2.0},
            e_attention_pj_per_flop=0.8,
            e_kv_read_pj_per_bit=12.0,
            e_kv_write_pj_per_bit=11.0,
            e_comm_pj_per_bit=5.0,
            e_decode_batch_exponent=0.7,
        )

        d = params.to_dict()

        # Verify inference section exists in serialized output
        assert "inference" in d
        inf = d["inference"]
        assert inf["attention_pj_per_flop"] == 0.8
        assert inf["kv_read_pj_per_bit"] == 12.0
        assert inf["kv_write_pj_per_bit"] == 11.0
        assert inf["comm_pj_per_bit"] == 5.0
        assert inf["decode_batch_exponent"] == 0.7
        assert inf["gemm_prefill"]["fp16"] == 1.5
        assert inf["gemm_decode"]["fp16"] == 2.0

        # Roundtrip
        restored = EnergyParameters.from_dict(d)
        assert restored.e_attention_pj_per_flop == 0.8
        assert restored.e_kv_read_pj_per_bit == 12.0
        assert restored.e_kv_write_pj_per_bit == 11.0
        assert restored.e_comm_pj_per_bit == 5.0
        assert restored.e_decode_batch_exponent == 0.7
        assert restored.e_inference_gemm_prefill_pj_per_flop[DataType.FP16] == 1.5
        assert restored.e_inference_gemm_prefill_pj_per_flop[DataType.FP32] == 3.0
        assert restored.e_inference_gemm_decode_pj_per_flop[DataType.FP16] == 2.0

    def test_backward_compat_no_inference_key(self):
        """from_dict without 'inference' section should use defaults."""
        data = {
            "platform": "nvidia",
            "hardware_name": "test GPU",
            "memory": {"total_pj_per_bit": 10.0},
            "idle_power": {"cpu_watts": 5.0, "gpu_watts": 20.0},
        }
        params = EnergyParameters.from_dict(data)

        # Inference fields should have zero/default values
        assert params.e_attention_pj_per_flop == 0.0
        assert params.e_kv_read_pj_per_bit == 0.0
        assert params.e_kv_write_pj_per_bit == 0.0
        assert params.e_comm_pj_per_bit == 0.0
        assert params.e_decode_batch_exponent == 1.0
        assert params.e_inference_gemm_prefill_pj_per_flop == {}
        assert params.e_inference_gemm_decode_pj_per_flop == {}

        # Original fields should still work
        assert params.e_memory_pj_per_bit == 10.0
        assert params.p_idle_cpu_watts == 5.0

    def test_json_roundtrip(self):
        """Serialize to JSON and back."""
        params = EnergyParameters(
            platform=Platform.MACOS,
            hardware_name="Apple M2",
            e_attention_pj_per_flop=1.5,
            e_decode_batch_exponent=0.85,
        )
        json_str = json.dumps(params.to_dict())
        data = json.loads(json_str)
        restored = EnergyParameters.from_dict(data)
        assert restored.e_attention_pj_per_flop == 1.5
        assert restored.e_decode_batch_exponent == 0.85


# ---------------------------------------------------------------------------
# InferenceCharacterizationConfig tests
# ---------------------------------------------------------------------------

class TestInferenceCharacterizationConfig:
    """Test InferenceCharacterizationConfig defaults and structure."""

    def test_defaults(self):
        config = InferenceCharacterizationConfig()
        assert config.gemm_batch_sizes == [1, 4, 16, 64]
        assert config.gemm_seq_lens == [128, 512, 2048, 8192]
        assert config.gemm_hidden_dim == 4096
        assert config.gemm_ff_dim == 11008
        assert config.attn_batch_sizes == [1, 4, 16]
        assert config.attn_num_heads == 32
        assert config.attn_head_dim == 128
        assert config.kv_cache_entries == [128, 512, 2048, 8192, 32768]
        assert config.nccl_message_sizes_mb == [1, 10, 100, 500]
        assert config.decode_batch_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
        assert config.duration_per_workload == 10.0

    def test_custom_values(self):
        config = InferenceCharacterizationConfig(
            gemm_batch_sizes=[1, 2],
            duration_per_workload=5.0,
        )
        assert config.gemm_batch_sizes == [1, 2]
        assert config.duration_per_workload == 5.0


# ---------------------------------------------------------------------------
# Abstract base class hierarchy tests
# ---------------------------------------------------------------------------

class TestAbstractBaseClasses:
    """Test that new abstract base classes inherit correctly."""

    def test_inference_gemm_is_gemm(self):
        from ipw.benchmarks.workloads.base import GEMMWorkload
        assert issubclass(InferenceGEMMWorkload, GEMMWorkload)

    def test_attention_is_workload(self):
        from ipw.benchmarks.workloads.base import Workload
        assert issubclass(AttentionWorkload, Workload)

    def test_kv_cache_is_memory(self):
        from ipw.benchmarks.workloads.base import MemoryWorkload
        assert issubclass(KVCacheWorkload, MemoryWorkload)

    def test_nccl_is_workload(self):
        from ipw.benchmarks.workloads.base import Workload
        assert issubclass(NCCLCollectiveWorkload, Workload)

    def test_batched_decode_is_workload(self):
        from ipw.benchmarks.workloads.base import Workload
        assert issubclass(BatchedDecodeWorkload, Workload)


# ---------------------------------------------------------------------------
# Extraction function tests with synthetic data
# ---------------------------------------------------------------------------

def _make_energy(total_joules: float, duration: float) -> EnergyMeasurement:
    """Helper to create a synthetic EnergyMeasurement."""
    power = total_joules / duration if duration > 0 else 0.0
    return EnergyMeasurement(
        cpu_energy_joules=0.0,
        gpu_energy_joules=total_joules,
        ane_energy_joules=0.0,
        avg_cpu_power_watts=0.0,
        avg_gpu_power_watts=power,
        avg_ane_power_watts=0.0,
        max_cpu_power_watts=0.0,
        max_gpu_power_watts=power,
        max_ane_power_watts=0.0,
        duration_seconds=duration,
        sample_count=10,
    )


def _make_inference_gemm_result(
    mode: str,
    batch_size: int,
    seq_len: int,
    tflops: float,
    total_joules: float,
    duration: float = 10.0,
    dtype: DataType = DataType.FP16,
) -> BenchmarkResult:
    """Create a synthetic inference GEMM benchmark result."""
    hidden_dim = 4096
    ff_dim = 11008
    M = batch_size * seq_len if mode == "prefill" else batch_size
    flops_per_mm = 2 * M * hidden_dim * ff_dim
    total_flops = int(tflops * 1e12 * duration)

    return BenchmarkResult(
        workload=WorkloadResult(
            workload_type=WorkloadType.INFERENCE_GEMM,
            config=WorkloadConfig(
                workload_type=WorkloadType.INFERENCE_GEMM,
                duration_seconds=duration,
                use_zeros=False,
                data_type=dtype,
                params={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "ff_dim": ff_dim,
                    "mode": mode,
                },
            ),
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=duration,
        ),
        energy=_make_energy(total_joules, duration),
    )


def _make_attention_result(
    batch_size: int,
    seq_len: int,
    tflops: float,
    total_joules: float,
    duration: float = 10.0,
) -> BenchmarkResult:
    """Create a synthetic attention benchmark result."""
    num_heads = 32
    head_dim = 128
    total_flops = int(tflops * 1e12 * duration)

    return BenchmarkResult(
        workload=WorkloadResult(
            workload_type=WorkloadType.ATTENTION,
            config=WorkloadConfig(
                workload_type=WorkloadType.ATTENTION,
                duration_seconds=duration,
                use_zeros=False,
                data_type=DataType.FP16,
                params={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                },
            ),
            throughput=tflops,
            throughput_unit="TFLOP/s",
            flops_executed=total_flops,
            duration_seconds=duration,
        ),
        energy=_make_energy(total_joules, duration),
    )


def _make_kv_cache_result(
    mode: str,
    cache_entries: int,
    bandwidth_gb_s: float,
    total_joules: float,
    duration: float = 10.0,
) -> BenchmarkResult:
    """Create a synthetic KV cache benchmark result."""
    bytes_transferred = int(bandwidth_gb_s * 1e9 * duration)

    return BenchmarkResult(
        workload=WorkloadResult(
            workload_type=WorkloadType.KV_CACHE_IO,
            config=WorkloadConfig(
                workload_type=WorkloadType.KV_CACHE_IO,
                duration_seconds=duration,
                use_zeros=False,
                data_type=DataType.FP16,
                params={
                    "cache_entries": cache_entries,
                    "num_heads": 32,
                    "head_dim": 128,
                    "batch_size": 1,
                    "mode": mode,
                },
            ),
            throughput=bandwidth_gb_s,
            throughput_unit="GB/s",
            bytes_transferred=bytes_transferred,
            duration_seconds=duration,
        ),
        energy=_make_energy(total_joules, duration),
    )


def _make_batched_decode_result(
    batch_size: int,
    tokens_per_s: float,
    total_joules: float,
    duration: float = 10.0,
) -> BenchmarkResult:
    """Create a synthetic batched decode benchmark result."""
    total_flops = int(4 * batch_size * 4096 * 11008 * tokens_per_s / batch_size * duration)

    return BenchmarkResult(
        workload=WorkloadResult(
            workload_type=WorkloadType.BATCHED_DECODE,
            config=WorkloadConfig(
                workload_type=WorkloadType.BATCHED_DECODE,
                duration_seconds=duration,
                use_zeros=False,
                data_type=DataType.FP16,
                params={
                    "batch_size": batch_size,
                    "hidden_dim": 4096,
                    "ff_dim": 11008,
                    "num_layers": 1,
                },
            ),
            throughput=tokens_per_s,
            throughput_unit="tokens/s",
            flops_executed=total_flops,
            duration_seconds=duration,
        ),
        energy=_make_energy(total_joules, duration),
    )


class TestExtractionFunctions:
    """Test inference parameter extraction with synthetic data."""

    def test_extract_inference_gemm_params(self):
        """Test extraction of inference GEMM params by mode and dtype."""
        from ipw.benchmarks.analysis import _extract_inference_gemm_params

        params = EnergyParameters(
            platform=Platform.NVIDIA, hardware_name="test"
        )

        results = [
            # Prefill at different TFLOP/s rates -> different power
            _make_inference_gemm_result("prefill", 1, 128, tflops=50.0, total_joules=150.0),
            _make_inference_gemm_result("prefill", 4, 512, tflops=100.0, total_joules=250.0),
            _make_inference_gemm_result("prefill", 16, 2048, tflops=200.0, total_joules=400.0),
            # Decode
            _make_inference_gemm_result("decode", 1, 128, tflops=10.0, total_joules=80.0),
            _make_inference_gemm_result("decode", 4, 512, tflops=30.0, total_joules=120.0),
            _make_inference_gemm_result("decode", 16, 2048, tflops=60.0, total_joules=200.0),
        ]

        params = _extract_inference_gemm_params(params, results, idle_power=0.0)

        # Should have FP16 entries for both modes
        assert DataType.FP16 in params.e_inference_gemm_prefill_pj_per_flop
        assert DataType.FP16 in params.e_inference_gemm_decode_pj_per_flop
        # Values should be positive
        assert params.e_inference_gemm_prefill_pj_per_flop[DataType.FP16] > 0
        assert params.e_inference_gemm_decode_pj_per_flop[DataType.FP16] > 0

    def test_extract_attention_params(self):
        """Test extraction of attention energy parameter."""
        from ipw.benchmarks.analysis import _extract_attention_params

        params = EnergyParameters(
            platform=Platform.NVIDIA, hardware_name="test"
        )

        results = [
            _make_attention_result(1, 512, tflops=50.0, total_joules=100.0),
            _make_attention_result(4, 1024, tflops=100.0, total_joules=180.0),
            _make_attention_result(16, 2048, tflops=200.0, total_joules=350.0),
        ]

        params = _extract_attention_params(params, results, idle_power=0.0)
        assert params.e_attention_pj_per_flop > 0

    def test_extract_kv_cache_params(self):
        """Test extraction of KV cache read/write energy."""
        from ipw.benchmarks.analysis import _extract_kv_cache_params

        params = EnergyParameters(
            platform=Platform.NVIDIA, hardware_name="test"
        )

        results = [
            # Read results at different bandwidths
            _make_kv_cache_result("read", 128, bandwidth_gb_s=50.0, total_joules=100.0),
            _make_kv_cache_result("read", 2048, bandwidth_gb_s=100.0, total_joules=180.0),
            _make_kv_cache_result("read", 8192, bandwidth_gb_s=200.0, total_joules=350.0),
            # Write results
            _make_kv_cache_result("write", 128, bandwidth_gb_s=40.0, total_joules=90.0),
            _make_kv_cache_result("write", 2048, bandwidth_gb_s=80.0, total_joules=160.0),
            _make_kv_cache_result("write", 8192, bandwidth_gb_s=160.0, total_joules=300.0),
        ]

        params = _extract_kv_cache_params(params, results, idle_power=0.0)
        assert params.e_kv_read_pj_per_bit > 0
        assert params.e_kv_write_pj_per_bit > 0

    def test_extract_batched_decode_params(self):
        """Test extraction of batched decode exponent β."""
        from ipw.benchmarks.analysis import _extract_batched_decode_params

        params = EnergyParameters(
            platform=Platform.NVIDIA, hardware_name="test"
        )

        # Simulate sub-linear scaling: E_per_token decreases with batch size
        # tokens/s scales super-linearly -> E/token drops
        results = [
            _make_batched_decode_result(1, tokens_per_s=100, total_joules=10.0),
            _make_batched_decode_result(4, tokens_per_s=380, total_joules=35.0),
            _make_batched_decode_result(16, tokens_per_s=1400, total_joules=120.0),
            _make_batched_decode_result(64, tokens_per_s=5000, total_joules=400.0),
        ]

        params = _extract_batched_decode_params(params, results, idle_power=0.0)
        # β should be extracted (between 0 and 2)
        assert 0.0 <= params.e_decode_batch_exponent <= 2.0

    def test_extract_parameters_integration(self):
        """Test that extract_parameters handles inference results correctly."""
        from ipw.benchmarks.analysis import extract_parameters

        results = [
            _make_inference_gemm_result("prefill", 1, 128, tflops=50.0, total_joules=150.0),
            _make_inference_gemm_result("prefill", 4, 512, tflops=100.0, total_joules=250.0),
            _make_inference_gemm_result("decode", 1, 128, tflops=10.0, total_joules=80.0),
            _make_inference_gemm_result("decode", 4, 512, tflops=30.0, total_joules=120.0),
            _make_attention_result(1, 512, tflops=50.0, total_joules=100.0),
            _make_attention_result(4, 1024, tflops=100.0, total_joules=180.0),
        ]

        params = extract_parameters(
            results,
            platform=Platform.NVIDIA,
            hardware_name="test",
            idle_cpu_watts=0.0,
            idle_gpu_watts=0.0,
        )

        # Inference params should be extracted
        assert len(params.e_inference_gemm_prefill_pj_per_flop) > 0
        assert len(params.e_inference_gemm_decode_pj_per_flop) > 0
        assert params.e_attention_pj_per_flop > 0

    def test_extract_empty_inference_results(self):
        """extract_parameters with no inference results should leave defaults."""
        from ipw.benchmarks.analysis import extract_parameters

        params = extract_parameters(
            [],
            platform=Platform.MACOS,
            hardware_name="test",
        )

        assert params.e_attention_pj_per_flop == 0.0
        assert params.e_kv_read_pj_per_bit == 0.0
        assert params.e_decode_batch_exponent == 1.0
        assert params.e_inference_gemm_prefill_pj_per_flop == {}


# ---------------------------------------------------------------------------
# Skip-if-no-CUDA tests for actual workload execution
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_mps():
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestNVIDIAWorkloads:
    """Integration tests for NVIDIA inference workloads (requires CUDA)."""

    def test_inference_gemm_prefill(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAInferenceGEMMWorkload

        wl = NVIDIAInferenceGEMMWorkload()
        assert wl.is_available()
        config = WorkloadConfig(
            workload_type=WorkloadType.INFERENCE_GEMM,
            duration_seconds=2.0,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 128,
                "hidden_dim": 4096,
                "ff_dim": 11008,
                "mode": "prefill",
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.flops_executed > 0

    def test_attention_workload(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAAttentionWorkload

        wl = NVIDIAAttentionWorkload()
        assert wl.is_available()
        config = WorkloadConfig(
            workload_type=WorkloadType.ATTENTION,
            duration_seconds=2.0,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 256,
                "num_heads": 32,
                "head_dim": 128,
            },
        )
        result = wl.run(config)
        assert result.throughput > 0

    def test_kv_cache_read(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAKVCacheWorkload

        wl = NVIDIAKVCacheWorkload()
        assert wl.is_available()
        config = WorkloadConfig(
            workload_type=WorkloadType.KV_CACHE_IO,
            duration_seconds=2.0,
            data_type=DataType.FP16,
            params={
                "cache_entries": 512,
                "num_heads": 32,
                "head_dim": 128,
                "batch_size": 1,
                "mode": "read",
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.bytes_transferred > 0

    def test_batched_decode(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIABatchedDecodeWorkload

        wl = NVIDIABatchedDecodeWorkload()
        assert wl.is_available()
        config = WorkloadConfig(
            workload_type=WorkloadType.BATCHED_DECODE,
            duration_seconds=2.0,
            data_type=DataType.FP16,
            params={
                "batch_size": 4,
                "hidden_dim": 4096,
                "ff_dim": 11008,
                "num_layers": 1,
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.flops_executed > 0

    def test_nccl_not_available_single_gpu(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIANCCLCollectiveWorkload

        wl = NVIDIANCCLCollectiveWorkload()
        # On single-GPU, should not be available
        assert not wl.is_available()


@pytest.mark.skipif(not _has_mps(), reason="MPS not available")
class TestMacOSWorkloads:
    """Integration tests for macOS inference workloads (requires MPS)."""

    def test_inference_gemm_prefill(self):
        from ipw.benchmarks.platforms.macos import MacOSInferenceGEMMWorkload

        wl = MacOSInferenceGEMMWorkload()
        assert wl.is_available()
        config = WorkloadConfig(
            workload_type=WorkloadType.INFERENCE_GEMM,
            duration_seconds=2.0,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 128,
                "hidden_dim": 4096,
                "ff_dim": 11008,
                "mode": "prefill",
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.flops_executed > 0

    def test_attention_workload(self):
        from ipw.benchmarks.platforms.macos import MacOSAttentionWorkload

        wl = MacOSAttentionWorkload()
        assert wl.is_available()
        config = WorkloadConfig(
            workload_type=WorkloadType.ATTENTION,
            duration_seconds=2.0,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 256,
                "num_heads": 32,
                "head_dim": 128,
            },
        )
        result = wl.run(config)
        assert result.throughput > 0


# ---------------------------------------------------------------------------
# CUDA Graph tests
# ---------------------------------------------------------------------------

class TestWorkloadConfigCUDAGraphs:
    """Test WorkloadConfig use_cuda_graphs field."""

    def test_default_false(self):
        config = WorkloadConfig(workload_type=WorkloadType.INFERENCE_GEMM)
        assert config.use_cuda_graphs is False

    def test_set_true(self):
        config = WorkloadConfig(
            workload_type=WorkloadType.INFERENCE_GEMM,
            use_cuda_graphs=True,
        )
        assert config.use_cuda_graphs is True


class TestEnergyParametersRawFields:
    """Test EnergyParameters raw (CUDA graph) fields and serialization."""

    def test_raw_fields_default_values(self):
        params = EnergyParameters(platform=Platform.NVIDIA, hardware_name="test")
        assert params.e_inference_gemm_prefill_raw_pj_per_flop == {}
        assert params.e_inference_gemm_decode_raw_pj_per_flop == {}
        assert params.e_attention_raw_pj_per_flop == 0.0
        assert params.e_kv_read_raw_pj_per_bit == 0.0
        assert params.e_kv_write_raw_pj_per_bit == 0.0
        assert params.e_decode_batch_raw_exponent == 1.0

    def test_roundtrip_with_raw_fields(self):
        params = EnergyParameters(
            platform=Platform.NVIDIA,
            hardware_name="NVIDIA A100",
            e_inference_gemm_prefill_raw_pj_per_flop={DataType.FP16: 0.5},
            e_inference_gemm_decode_raw_pj_per_flop={DataType.FP16: 1.2},
            e_attention_raw_pj_per_flop=0.3,
            e_kv_read_raw_pj_per_bit=8.0,
            e_kv_write_raw_pj_per_bit=7.5,
            e_decode_batch_raw_exponent=0.6,
        )

        d = params.to_dict()

        # Check inference_raw section exists
        assert "inference_raw" in d
        raw = d["inference_raw"]
        assert raw["attention_pj_per_flop"] == 0.3
        assert raw["kv_read_pj_per_bit"] == 8.0
        assert raw["kv_write_pj_per_bit"] == 7.5
        assert raw["decode_batch_exponent"] == 0.6
        assert raw["gemm_prefill"]["fp16"] == 0.5
        assert raw["gemm_decode"]["fp16"] == 1.2

        # Roundtrip
        restored = EnergyParameters.from_dict(d)
        assert restored.e_attention_raw_pj_per_flop == 0.3
        assert restored.e_kv_read_raw_pj_per_bit == 8.0
        assert restored.e_kv_write_raw_pj_per_bit == 7.5
        assert restored.e_decode_batch_raw_exponent == 0.6
        assert restored.e_inference_gemm_prefill_raw_pj_per_flop[DataType.FP16] == 0.5
        assert restored.e_inference_gemm_decode_raw_pj_per_flop[DataType.FP16] == 1.2

    def test_backward_compat_no_inference_raw_key(self):
        """from_dict without 'inference_raw' section should use defaults."""
        data = {
            "platform": "nvidia",
            "hardware_name": "test GPU",
            "memory": {"total_pj_per_bit": 10.0},
            "inference": {
                "attention_pj_per_flop": 0.8,
                "kv_read_pj_per_bit": 12.0,
            },
            "idle_power": {"cpu_watts": 5.0, "gpu_watts": 20.0},
        }
        params = EnergyParameters.from_dict(data)

        # Raw fields should have defaults
        assert params.e_attention_raw_pj_per_flop == 0.0
        assert params.e_kv_read_raw_pj_per_bit == 0.0
        assert params.e_decode_batch_raw_exponent == 1.0
        assert params.e_inference_gemm_prefill_raw_pj_per_flop == {}

        # Standard fields should be populated
        assert params.e_attention_pj_per_flop == 0.8
        assert params.e_kv_read_pj_per_bit == 12.0


class TestExtractionWithCUDAGraphFlag:
    """Test that extract_parameters separates standard vs CUDA graph results."""

    def test_standard_results_go_to_standard_fields(self):
        """Results without use_cuda_graphs go to standard inference fields."""
        from ipw.benchmarks.analysis import extract_parameters

        results = [
            _make_inference_gemm_result("prefill", 1, 128, tflops=50.0, total_joules=150.0),
            _make_inference_gemm_result("prefill", 4, 512, tflops=100.0, total_joules=250.0),
        ]

        params = extract_parameters(
            results,
            platform=Platform.NVIDIA,
            hardware_name="test",
        )

        assert len(params.e_inference_gemm_prefill_pj_per_flop) > 0
        # Raw fields should be empty (no CUDA graph results)
        assert params.e_inference_gemm_prefill_raw_pj_per_flop == {}

    def test_cuda_graph_results_go_to_raw_fields(self):
        """Results with use_cuda_graphs=True go to raw inference fields."""
        from ipw.benchmarks.analysis import extract_parameters

        # Create results with use_cuda_graphs=True
        raw_results = []
        for bs, seq, tflops, joules in [(1, 128, 55.0, 140.0), (4, 512, 110.0, 240.0)]:
            r = _make_inference_gemm_result("prefill", bs, seq, tflops=tflops, total_joules=joules)
            # Override the config to set use_cuda_graphs
            r.workload.config = WorkloadConfig(
                workload_type=r.workload.config.workload_type,
                duration_seconds=r.workload.config.duration_seconds,
                use_zeros=r.workload.config.use_zeros,
                use_cuda_graphs=True,
                data_type=r.workload.config.data_type,
                params=r.workload.config.params,
            )
            raw_results.append(r)

        params = extract_parameters(
            raw_results,
            platform=Platform.NVIDIA,
            hardware_name="test",
        )

        # Standard fields should be empty (no standard results)
        assert params.e_inference_gemm_prefill_pj_per_flop == {}
        # Raw fields should be populated
        assert len(params.e_inference_gemm_prefill_raw_pj_per_flop) > 0

    def test_mixed_standard_and_raw_results(self):
        """Both standard and CUDA graph results are extracted to their respective fields."""
        from ipw.benchmarks.analysis import extract_parameters

        # Standard results
        std_results = [
            _make_attention_result(1, 512, tflops=50.0, total_joules=100.0),
            _make_attention_result(4, 1024, tflops=100.0, total_joules=180.0),
        ]

        # CUDA graph results
        raw_results = []
        for bs, seq, tflops, joules in [(1, 512, 55.0, 95.0), (4, 1024, 110.0, 170.0)]:
            r = _make_attention_result(bs, seq, tflops=tflops, total_joules=joules)
            r.workload.config = WorkloadConfig(
                workload_type=r.workload.config.workload_type,
                duration_seconds=r.workload.config.duration_seconds,
                use_zeros=r.workload.config.use_zeros,
                use_cuda_graphs=True,
                data_type=r.workload.config.data_type,
                params=r.workload.config.params,
            )
            raw_results.append(r)

        all_results = std_results + raw_results

        params = extract_parameters(
            all_results,
            platform=Platform.NVIDIA,
            hardware_name="test",
        )

        # Both should be populated
        assert params.e_attention_pj_per_flop > 0
        assert params.e_attention_raw_pj_per_flop > 0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestNVIDIACUDAGraphWorkloads:
    """Integration tests for NVIDIA CUDA graph workloads (requires CUDA)."""

    def test_inference_gemm_cuda_graph(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAInferenceGEMMWorkload

        wl = NVIDIAInferenceGEMMWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.INFERENCE_GEMM,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 128,
                "hidden_dim": 4096,
                "ff_dim": 11008,
                "mode": "decode",
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.flops_executed > 0

    def test_attention_cuda_graph(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAAttentionWorkload

        wl = NVIDIAAttentionWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.ATTENTION,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 256,
                "num_heads": 32,
                "head_dim": 128,
            },
        )
        result = wl.run(config)
        assert result.throughput > 0

    def test_kv_cache_read_cuda_graph(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAKVCacheWorkload

        wl = NVIDIAKVCacheWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.KV_CACHE_IO,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "cache_entries": 512,
                "num_heads": 32,
                "head_dim": 128,
                "batch_size": 1,
                "mode": "read",
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.bytes_transferred > 0

    def test_kv_cache_write_cuda_graph(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIAKVCacheWorkload

        wl = NVIDIAKVCacheWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.KV_CACHE_IO,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "cache_entries": 512,
                "num_heads": 32,
                "head_dim": 128,
                "batch_size": 1,
                "mode": "write",
            },
        )
        result = wl.run(config)
        assert result.throughput > 0

    def test_batched_decode_cuda_graph(self):
        from ipw.benchmarks.platforms.nvidia import NVIDIABatchedDecodeWorkload

        wl = NVIDIABatchedDecodeWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.BATCHED_DECODE,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "batch_size": 4,
                "hidden_dim": 4096,
                "ff_dim": 11008,
                "num_layers": 1,
            },
        )
        result = wl.run(config)
        assert result.throughput > 0
        assert result.flops_executed > 0


@pytest.mark.skipif(not _has_mps(), reason="MPS not available")
class TestMacOSCUDAGraphWarning:
    """Test that macOS workloads warn when use_cuda_graphs=True."""

    def test_inference_gemm_warns(self):
        import warnings
        from ipw.benchmarks.platforms.macos import MacOSInferenceGEMMWorkload

        wl = MacOSInferenceGEMMWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.INFERENCE_GEMM,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 128,
                "hidden_dim": 4096,
                "ff_dim": 11008,
                "mode": "prefill",
            },
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wl.run(config)
            cuda_graph_warnings = [
                x for x in w if "CUDA graphs not supported on MPS" in str(x.message)
            ]
            assert len(cuda_graph_warnings) >= 1
        assert result.throughput > 0

    def test_attention_warns(self):
        import warnings
        from ipw.benchmarks.platforms.macos import MacOSAttentionWorkload

        wl = MacOSAttentionWorkload()
        config = WorkloadConfig(
            workload_type=WorkloadType.ATTENTION,
            duration_seconds=2.0,
            use_cuda_graphs=True,
            data_type=DataType.FP16,
            params={
                "batch_size": 1,
                "seq_len": 256,
                "num_heads": 32,
                "head_dim": 128,
            },
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wl.run(config)
            cuda_graph_warnings = [
                x for x in w if "CUDA graphs not supported on MPS" in str(x.message)
            ]
            assert len(cuda_graph_warnings) >= 1
        assert result.throughput > 0
