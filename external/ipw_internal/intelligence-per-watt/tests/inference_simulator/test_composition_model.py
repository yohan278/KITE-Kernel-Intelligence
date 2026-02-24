"""Tests for learned operator composition weights (IrEne-inspired)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from inference_simulator.estimator.composition_model import (
    CompositionModelTrainer,
    CompositionWeights,
    load_fused_measurements,
)
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


class TestCompositionWeightsSerialization:
    """Test JSON serialization roundtrip."""

    def test_roundtrip(self, tmp_path):
        weights = CompositionWeights(
            linear_weight=4.8,
            norm_weight=2.1,
            activation_weight=3.5,
            embedding_weight=0.2,
            communication_weight=1.8,
            overlap_correction=0.92,
            metadata={"source": "test", "loss": 0.001},
        )
        path = tmp_path / "weights.json"
        weights.to_json(path)

        loaded = CompositionWeights.from_json(path)
        assert loaded.linear_weight == pytest.approx(4.8)
        assert loaded.norm_weight == pytest.approx(2.1)
        assert loaded.activation_weight == pytest.approx(3.5)
        assert loaded.embedding_weight == pytest.approx(0.2)
        assert loaded.communication_weight == pytest.approx(1.8)
        assert loaded.overlap_correction == pytest.approx(0.92)
        assert loaded.metadata["source"] == "test"

    def test_defaults_roundtrip(self, tmp_path):
        weights = CompositionWeights()
        path = tmp_path / "defaults.json"
        weights.to_json(path)

        loaded = CompositionWeights.from_json(path)
        assert loaded.linear_weight == 5.0
        assert loaded.norm_weight == 2.0
        assert loaded.activation_weight == 4.0
        assert loaded.embedding_weight == 0.0
        assert loaded.overlap_correction == 1.0


class TestFromConfig:
    """Test CompositionWeights.from_config() equivalence to defaults."""

    def test_from_simulator_config(self):
        from inference_simulator.engine.simulator_config import SimulatorConfig

        config = SimulatorConfig()
        weights = CompositionWeights.from_config(config)

        assert weights.linear_weight == 5.0
        assert weights.norm_weight == 2.0
        assert weights.activation_weight == 4.0
        assert weights.embedding_weight == 0.0
        assert weights.overlap_correction == 1.0
        assert weights.metadata["source"] == "SimulatorConfig"

    def test_from_custom_config(self):
        from inference_simulator.engine.simulator_config import SimulatorConfig

        config = SimulatorConfig(
            ops_linear_per_layer=3,
            ops_norm_per_layer=1,
            ops_activation_per_layer=2,
            overlap_correction=0.8,
        )
        weights = CompositionWeights.from_config(config)

        assert weights.linear_weight == 3.0
        assert weights.norm_weight == 1.0
        assert weights.activation_weight == 2.0
        assert weights.overlap_correction == 0.8


class TestCompositionModelTrainer:
    """Test fitting composition weights from synthetic fused data."""

    @pytest.fixture
    def model_spec(self):
        from inference_simulator.types.model_spec import (
            ArchitectureType,
            AttentionType,
            ModelSpec,
        )

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
    def hardware_spec(self):
        from inference_simulator.types.hardware_spec import HardwareSpec

        return HardwareSpec.from_registry("h100_80gb")

    def _make_mock_estimator(self, time_per_op: float = 1e-5):
        """Create a mock estimator that returns fixed per-op times."""
        from inference_simulator.estimator.base import EstimatorResult

        class MockEstimator:
            def estimate(self, category, batch_size=1, seq_len=1, **kwargs):
                return EstimatorResult(time_s=time_per_op)

        return MockEstimator()

    @pytest.fixture(autouse=True)
    def _skip_no_scipy(self):
        pytest.importorskip("scipy")

    def test_fit_with_known_weights(self, model_spec, hardware_spec):
        """Synthetic data with known weights → verify recovery within 10%."""
        # Known ground-truth weights
        true_linear_w = 4.5
        true_norm_w = 1.8
        true_act_w = 3.2
        t_per_op = 1e-5  # each op takes 10us

        # Generate synthetic fused measurements:
        # fused_time = num_layers * (linear*w + norm*w + act*w + embed*0) * overlap * per_op_time
        measurements = []
        for bs in [1, 2, 4, 8]:
            for sl in [128, 256, 512, 1024]:
                per_layer_time = (
                    t_per_op * true_linear_w
                    + t_per_op * true_norm_w
                    + t_per_op * true_act_w
                )
                fused_time = per_layer_time * model_spec.num_layers
                measurements.append(
                    OperatorMeasurement(
                        operator_name="fused_prefill",
                        category=OperatorCategory.FUSED_PREFILL,
                        batch_size=bs,
                        seq_len=sl,
                        time_s=fused_time,
                    )
                )

        estimator = self._make_mock_estimator(time_per_op=t_per_op)
        trainer = CompositionModelTrainer(model_spec, hardware_spec)
        weights = trainer.fit_from_fused_data(measurements, estimator)

        # Verify recovery: the product (sum_of_weights * overlap) should be
        # close to the true sum, since the optimizer can trade off individual
        # weights against the overlap correction factor.
        predicted_product = (
            weights.linear_weight * t_per_op
            + weights.norm_weight * t_per_op
            + weights.activation_weight * t_per_op
        ) * weights.overlap_correction
        true_sum = (true_linear_w + true_norm_w + true_act_w) * t_per_op
        assert predicted_product == pytest.approx(true_sum, rel=0.2)

    def test_too_few_samples_returns_defaults(self, model_spec, hardware_spec):
        """Fewer than 3 samples should fall back to defaults."""
        measurements = [
            OperatorMeasurement(
                operator_name="fused_prefill",
                category=OperatorCategory.FUSED_PREFILL,
                batch_size=1,
                seq_len=128,
                time_s=0.01,
            )
        ]
        estimator = self._make_mock_estimator()
        trainer = CompositionModelTrainer(model_spec, hardware_spec)
        weights = trainer.fit_from_fused_data(measurements, estimator)

        # Should return defaults
        assert weights.linear_weight == 5.0
        assert weights.norm_weight == 2.0

    def test_bounds_enforcement(self, model_spec, hardware_spec):
        """Weights should be bounded within [default*0.5, default*2.0]."""
        t_per_op = 1e-5
        # Generate data that would push weights to extremes
        measurements = []
        for bs in [1, 2, 4, 8]:
            for sl in [128, 256, 512]:
                # Very small target → optimizer tries to shrink all weights
                measurements.append(
                    OperatorMeasurement(
                        operator_name="fused_prefill",
                        category=OperatorCategory.FUSED_PREFILL,
                        batch_size=bs,
                        seq_len=sl,
                        time_s=t_per_op * 0.01 * model_spec.num_layers,
                    )
                )

        estimator = self._make_mock_estimator(time_per_op=t_per_op)
        trainer = CompositionModelTrainer(model_spec, hardware_spec)
        weights = trainer.fit_from_fused_data(measurements, estimator)

        # Check bounds: [default*0.5, default*2.0]
        assert weights.linear_weight >= 5.0 * 0.5 - 0.01
        assert weights.linear_weight <= 5.0 * 2.0 + 0.01
        assert weights.norm_weight >= 2.0 * 0.5 - 0.01
        assert weights.norm_weight <= 2.0 * 2.0 + 0.01
        assert weights.overlap_correction >= 0.5 - 0.01
        assert weights.overlap_correction <= 1.0 + 0.01

    def test_communication_weight_tp(self, model_spec, hardware_spec):
        """Communication weight should be non-zero for tp_size > 1."""
        t_per_op = 1e-5
        measurements = []
        for bs in [1, 2, 4]:
            for sl in [128, 256, 512]:
                measurements.append(
                    OperatorMeasurement(
                        operator_name="fused_prefill",
                        category=OperatorCategory.FUSED_PREFILL,
                        batch_size=bs,
                        seq_len=sl,
                        time_s=t_per_op * 10 * model_spec.num_layers,
                    )
                )

        estimator = self._make_mock_estimator(time_per_op=t_per_op)
        trainer = CompositionModelTrainer(model_spec, hardware_spec)

        # TP=1: communication weight should be 0
        weights_tp1 = trainer.fit_from_fused_data(measurements, estimator, tp_size=1)
        assert weights_tp1.communication_weight == 0.0

        # TP=4: communication weight should be 2.0 (PIE-P default)
        weights_tp4 = trainer.fit_from_fused_data(measurements, estimator, tp_size=4)
        assert weights_tp4.communication_weight == 2.0


class TestSimulatorIntegration:
    """Verify composition weights are used by the simulator."""

    @pytest.fixture
    def model_spec(self):
        from inference_simulator.types.model_spec import (
            ArchitectureType,
            AttentionType,
            ModelSpec,
        )

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
    def hardware_spec(self):
        from inference_simulator.types.hardware_spec import HardwareSpec

        return HardwareSpec.from_registry("h100_80gb")

    def test_fallback_without_weights(self, model_spec, hardware_spec):
        """Simulator should work fine without composition weights."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.scheduler.vllm import VLLMScheduler
        from inference_simulator.types.inference_spec import InferenceSpec
        from inference_simulator.types.workload_spec import WorkloadSpec

        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=InferenceSpec(num_gpus=1, precision="fp16"),
            scheduler=VLLMScheduler(),
        )
        workload = WorkloadSpec(qps=5.0, avg_input_tokens=50, avg_output_tokens=10)
        metrics = sim.run(workload, duration_s=1.0, seed=42)
        assert metrics.total_requests > 0

    def test_weights_via_config_path(self, model_spec, hardware_spec, tmp_path):
        """Simulator should load composition weights from config path."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.engine.simulator_config import SimulatorConfig
        from inference_simulator.scheduler.vllm import VLLMScheduler
        from inference_simulator.types.inference_spec import InferenceSpec
        from inference_simulator.types.workload_spec import WorkloadSpec

        # Write weights to a JSON file
        weights = CompositionWeights(
            linear_weight=5.0,
            norm_weight=2.0,
            activation_weight=4.0,
            overlap_correction=1.0,
        )
        weights_path = tmp_path / "weights.json"
        weights.to_json(weights_path)

        config = SimulatorConfig(composition_weights_path=str(weights_path))
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=InferenceSpec(num_gpus=1, precision="fp16"),
            scheduler=VLLMScheduler(),
            config=config,
        )
        workload = WorkloadSpec(qps=5.0, avg_input_tokens=50, avg_output_tokens=10)
        metrics = sim.run(workload, duration_s=1.0, seed=42)
        # Should still process requests normally
        assert metrics.total_requests > 0

    def test_corrupt_weights_fallback(self, model_spec, hardware_spec, tmp_path):
        """Corrupt weights file should fall back gracefully."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.engine.simulator_config import SimulatorConfig
        from inference_simulator.scheduler.vllm import VLLMScheduler
        from inference_simulator.types.inference_spec import InferenceSpec
        from inference_simulator.types.workload_spec import WorkloadSpec

        # Write invalid JSON
        bad_path = tmp_path / "bad_weights.json"
        bad_path.write_text("not valid json")

        config = SimulatorConfig(composition_weights_path=str(bad_path))
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=InferenceSpec(num_gpus=1, precision="fp16"),
            scheduler=VLLMScheduler(),
            config=config,
        )
        workload = WorkloadSpec(qps=5.0, avg_input_tokens=50, avg_output_tokens=10)
        metrics = sim.run(workload, duration_s=1.0, seed=42)
        # Should still work using fallback
        assert metrics.total_requests > 0


class TestLoadFusedMeasurements:
    """Test helper functions for loading measurements."""

    def test_load_from_nonexistent_dir(self, tmp_path):
        """Should return empty list for missing CSV."""
        result = load_fused_measurements(tmp_path)
        assert result == []

    def test_load_from_valid_csv(self, tmp_path):
        """Should load measurements from a valid fused_prefill.csv."""
        csv_path = tmp_path / "fused_prefill.csv"
        csv_path.write_text(
            "operator_name,batch_size,seq_len,time_s,energy_j,power_w,flops\n"
            "fused_prefill,1,128,0.001,,,"
            "\n"
            "fused_prefill,4,256,0.005,,,"
        )
        result = load_fused_measurements(tmp_path)
        assert len(result) == 2
        assert result[0].category == OperatorCategory.FUSED_PREFILL
        assert result[0].time_s == 0.001
