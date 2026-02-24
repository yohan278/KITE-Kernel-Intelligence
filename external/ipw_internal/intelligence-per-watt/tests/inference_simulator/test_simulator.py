"""End-to-end tests for EventDrivenSimulator with roofline fallback."""

import pytest

from inference_simulator.engine.simulator import EventDrivenSimulator
from inference_simulator.scheduler.vllm import VLLMScheduler
from inference_simulator.scheduler.orca import OrcaScheduler
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


@pytest.fixture
def workload_spec():
    return WorkloadSpec(
        qps=10.0,
        avg_input_tokens=100,
        avg_output_tokens=20,
        input_token_std=20.0,
        output_token_std=5.0,
    )


class TestEventDrivenSimulatorVLLM:
    def test_basic_simulation(self, model_spec, hardware_spec, inference_spec, workload_spec):
        scheduler = VLLMScheduler(max_num_seqs=64, max_num_batched_tokens=4096)
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )

        metrics = sim.run(workload_spec, duration_s=2.0, seed=42)

        # Should process some requests
        assert metrics.total_requests > 0
        assert metrics.total_tokens_generated > 0

        # Throughput should be positive
        assert metrics.throughput_rps > 0
        assert metrics.throughput_tps > 0

        # Latencies should be positive
        assert metrics.ttft_p50 > 0
        assert metrics.e2e_p50 > 0

        # Energy should be positive
        assert metrics.total_energy_j > 0
        assert metrics.avg_power_w > 0

    def test_higher_qps(self, model_spec, hardware_spec, inference_spec):
        workload = WorkloadSpec(
            qps=10.0,
            avg_input_tokens=50,
            avg_output_tokens=10,
            input_token_std=10.0,
            output_token_std=3.0,
        )
        scheduler = VLLMScheduler(max_num_seqs=256, max_num_batched_tokens=8192)
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )

        metrics = sim.run(workload, duration_s=2.0, seed=42)
        assert metrics.total_requests > 5

    def test_deterministic(self, model_spec, hardware_spec, inference_spec, workload_spec):
        scheduler1 = VLLMScheduler()
        scheduler2 = VLLMScheduler()

        sim1 = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler1,
        )
        sim2 = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler2,
        )

        metrics1 = sim1.run(workload_spec, duration_s=2.0, seed=42)
        metrics2 = sim2.run(workload_spec, duration_s=2.0, seed=42)

        assert metrics1.total_requests == metrics2.total_requests
        assert metrics1.ttft_p50 == pytest.approx(metrics2.ttft_p50)
        assert metrics1.throughput_tps == pytest.approx(metrics2.throughput_tps)


class TestEventDrivenSimulatorOrca:
    def test_basic_simulation(self, model_spec, hardware_spec, inference_spec, workload_spec):
        scheduler = OrcaScheduler(max_batch_size=32)
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )

        metrics = sim.run(workload_spec, duration_s=2.0, seed=42)

        assert metrics.total_requests > 0
        assert metrics.total_tokens_generated > 0
        assert metrics.throughput_rps > 0
        assert metrics.ttft_p50 > 0
        assert metrics.e2e_p50 > 0


class TestSimulatorEdgeCases:
    def test_empty_workload(self, model_spec, hardware_spec, inference_spec):
        workload = WorkloadSpec(qps=0.0)
        scheduler = VLLMScheduler()
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )

        metrics = sim.run(workload, duration_s=1.0, seed=42)
        assert metrics.total_requests == 0

    def test_very_short_duration(self, model_spec, hardware_spec, inference_spec):
        workload = WorkloadSpec(qps=1.0)
        scheduler = VLLMScheduler()
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
        )

        metrics = sim.run(workload, duration_s=0.01, seed=42)
        # May or may not process requests in 10ms
        assert metrics.total_requests >= 0

    def test_multi_gpu(self, model_spec, hardware_spec):
        inference = InferenceSpec(num_gpus=4, precision="fp16")
        workload = WorkloadSpec(
            qps=10.0,
            avg_input_tokens=100,
            avg_output_tokens=20,
        )
        scheduler = VLLMScheduler()
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference,
            scheduler=scheduler,
        )

        metrics = sim.run(workload, duration_s=2.0, seed=42)
        assert metrics.total_requests > 0
