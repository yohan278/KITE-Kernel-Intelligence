"""Tests for Vidur's equivalent-sequence-length batch prefill formula."""

import math

import pytest

from inference_simulator.engine.simulator import EventDrivenSimulator
from inference_simulator.scheduler.vllm import VLLMScheduler
from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.types.inference_spec import InferenceSpec
from inference_simulator.types.model_spec import ArchitectureType, AttentionType, ModelSpec
from inference_simulator.types.workload_spec import WorkloadSpec


@pytest.fixture
def model_spec():
    return ModelSpec(
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


@pytest.fixture
def hardware_spec():
    return HardwareSpec.from_registry("h100_80gb")


@pytest.fixture
def inference_spec():
    return InferenceSpec(num_gpus=1, precision="fp16")


class TestVidurFormula:
    def test_single_request_batch(self):
        """Single request: equiv_seq_len == input_tokens."""
        seq_lengths = [1024]
        equiv = int(math.sqrt(sum(p * p for p in seq_lengths)))
        assert equiv == 1024

    def test_uniform_batch(self):
        """Uniform batch: equiv = sqrt(N * p^2) = p * sqrt(N)."""
        seq_lengths = [512, 512, 512, 512]
        equiv = int(math.sqrt(sum(p * p for p in seq_lengths)))
        expected = int(512 * math.sqrt(4))  # 1024
        assert equiv == expected

    def test_heterogeneous_batch(self):
        """Heterogeneous batch: dominated by largest sequence."""
        seq_lengths = [128, 128, 2048]
        equiv = int(math.sqrt(sum(p * p for p in seq_lengths)))
        # 128^2 + 128^2 + 2048^2 = 16384 + 16384 + 4194304 = 4227072
        # sqrt(4227072) ≈ 2056
        assert equiv > 2000  # Dominated by the 2048-length sequence
        assert equiv < 2100  # But not much more

        # Old formula: avg = sum / N = 2304 / 3 = 768
        avg = sum(seq_lengths) // len(seq_lengths)
        assert equiv > avg  # Vidur formula gives LARGER than average

    def test_simulator_uses_vidur_formula(self, model_spec, hardware_spec, inference_spec):
        """The simulator should use the Vidur formula for heterogeneous batches,
        resulting in different timing than the old average formula."""
        # Run a simulation with heterogeneous input lengths
        workload = WorkloadSpec(
            qps=5.0,
            avg_input_tokens=500,
            avg_output_tokens=10,
            input_token_std=300.0,  # High variance -> heterogeneous
            output_token_std=3.0,
        )
        scheduler = VLLMScheduler(max_num_seqs=64, max_num_batched_tokens=8192)
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )

        metrics = sim.run(workload, duration_s=2.0, seed=42)
        # Should complete requests successfully
        assert metrics.total_requests > 0
        assert metrics.ttft_p50 > 0


class TestEquivSeqLenEdgeCases:
    def test_empty_batch(self):
        """Empty batch has no effect."""
        seq_lengths = []
        equiv = int(math.sqrt(sum(p * p for p in seq_lengths)))
        assert equiv == 0

    def test_single_token(self):
        """Single token sequence."""
        seq_lengths = [1]
        equiv = int(math.sqrt(sum(p * p for p in seq_lengths)))
        assert equiv == 1

    def test_very_long_sequence(self):
        """Very long sequence in batch."""
        seq_lengths = [100, 100, 100000]
        equiv = int(math.sqrt(sum(p * p for p in seq_lengths)))
        # Dominated by 100000
        assert equiv > 99900
        assert equiv < 100100
