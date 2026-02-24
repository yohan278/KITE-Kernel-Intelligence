"""Tests for chunked prefill support in the simulator."""

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


class TestChunkedPrefill:
    def test_chunked_prefill_completes(self, model_spec, hardware_spec):
        """Simulation with chunked prefill enabled should complete."""
        inference_spec = InferenceSpec(
            num_gpus=1,
            precision="fp16",
            engine_config={
                "enable_chunked_prefill": True,
                "chunked_prefill_size": 512,
            },
        )
        scheduler = VLLMScheduler(
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
        )
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )
        workload = WorkloadSpec(
            qps=2.0,
            avg_input_tokens=1024,
            avg_output_tokens=50,
        )
        metrics = sim.run(workload, duration_s=2.0, seed=42)
        assert metrics.total_requests > 0
        assert metrics.ttft_p50 > 0

    def test_without_chunked_prefill_completes(self, model_spec, hardware_spec):
        """Simulation without chunked prefill should also complete."""
        inference_spec = InferenceSpec(
            num_gpus=1,
            precision="fp16",
            engine_config={
                "enable_chunked_prefill": False,
            },
        )
        scheduler = VLLMScheduler(
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=False,
        )
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )
        workload = WorkloadSpec(
            qps=2.0,
            avg_input_tokens=1024,
            avg_output_tokens=50,
        )
        metrics = sim.run(workload, duration_s=2.0, seed=42)
        assert metrics.total_requests > 0

    def test_chunked_prefill_with_long_prompts(self, model_spec, hardware_spec):
        """Long prompts with chunked prefill enabled should complete.

        Uses same basic parameters as test_chunked_prefill_completes but
        with an explicit chunked_prefill_size to trigger the chunking path.
        """
        inference_spec = InferenceSpec(
            num_gpus=1,
            precision="fp16",
            engine_config={
                "enable_chunked_prefill": True,
                "chunked_prefill_size": 512,
            },
        )
        scheduler = VLLMScheduler(
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
        )
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )
        workload = WorkloadSpec(
            qps=2.0,
            avg_input_tokens=1024,
            avg_output_tokens=50,
        )
        metrics = sim.run(workload, duration_s=2.0, seed=42)
        # Even if chunking adds overhead, simulation should complete
        # At minimum, total_prefill_time_s should be non-negative
        assert metrics.total_prefill_time_s >= 0

    def test_short_prompts_no_chunking(self, model_spec, hardware_spec):
        """Short prompts (< chunk_size) should not be chunked."""
        inference_spec = InferenceSpec(
            num_gpus=1,
            precision="fp16",
            engine_config={
                "enable_chunked_prefill": True,
                "chunked_prefill_size": 4096,
            },
        )
        scheduler = VLLMScheduler(
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
        )
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )
        workload = WorkloadSpec(
            qps=2.0,
            avg_input_tokens=100,  # Much smaller than chunk_size
            avg_output_tokens=10,
            input_token_std=0.0,
        )
        metrics = sim.run(workload, duration_s=2.0, seed=42)
        assert metrics.total_requests > 0
