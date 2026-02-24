"""End-to-end pipeline test: #1a → #1b → #2 → #3."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
    WorkloadType,
)
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


@pytest.fixture
def qwen3_8b_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen3-8B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=36,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=12288,
        vocab_size=151936,
    )


@pytest.fixture
def a100_spec() -> HardwareSpec:
    return HardwareSpec(
        name="A100-SXM4-80GB",
        vendor="nvidia",
        memory_gb=80,
        tdp_watts=400,
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        hbm_bandwidth_gb_s=2039.0,
        nvlink_bandwidth_gb_s=600.0,
        price_per_hour_usd=3.50,
    )


@pytest.fixture
def synthetic_profiling_data() -> list[OperatorMeasurement]:
    """Generate synthetic profiling data simulating Pipeline #1a output."""
    measurements = []
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [128, 256, 512, 1024, 2048]:
            tokens = batch_size * seq_len
            base_time = tokens * 1e-6

            measurements.append(
                OperatorMeasurement(
                    operator_name="linear_qkv",
                    category=OperatorCategory.LINEAR,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.0,
                    energy_j=base_time * 400,
                    power_w=400.0,
                    flops=int(tokens * 2 * 4096 * 6144),
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_prefill",
                    category=OperatorCategory.ATTENTION_PREFILL,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 1.5,
                    energy_j=base_time * 1.5 * 400,
                    power_w=420.0,
                )
            )
            measurements.append(
                OperatorMeasurement(
                    operator_name="attention_decode",
                    category=OperatorCategory.ATTENTION_DECODE,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    time_s=base_time * 0.3,
                    energy_j=base_time * 0.3 * 350,
                    power_w=350.0,
                )
            )
    return measurements


def _write_profiling_csvs(measurements, output_dir):
    """Write measurements to CSV files mimicking Pipeline #1a output."""
    token_ops_path = output_dir / "token_ops.csv"
    attention_path = output_dir / "attention.csv"

    # Token ops CSV
    with open(token_ops_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator_name", "batch_size", "seq_len",
                "time_s", "energy_j", "power_w", "flops", "bandwidth_gb_s",
            ],
        )
        writer.writeheader()
        for m in measurements:
            if m.category == OperatorCategory.LINEAR:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j or "",
                    "power_w": m.power_w or "",
                    "flops": m.flops or "",
                    "bandwidth_gb_s": "",
                })

    # Attention CSV
    with open(attention_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator_name", "variant", "batch_size", "seq_len",
                "kv_cache_size", "time_s", "energy_j", "power_w",
                "flops", "bandwidth_gb_s",
            ],
        )
        writer.writeheader()
        for m in measurements:
            if m.category in (
                OperatorCategory.ATTENTION_PREFILL,
                OperatorCategory.ATTENTION_DECODE,
            ):
                variant = (
                    "prefill"
                    if m.category == OperatorCategory.ATTENTION_PREFILL
                    else "decode"
                )
                writer.writerow({
                    "operator_name": m.operator_name,
                    "variant": variant,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "kv_cache_size": m.seq_len if variant == "decode" else "",
                    "time_s": m.time_s,
                    "energy_j": m.energy_j or "",
                    "power_w": m.power_w or "",
                    "flops": "",
                    "bandwidth_gb_s": "",
                })


class TestPipelineE2E:
    """End-to-end pipeline integration test (CPU-only, synthetic data)."""

    @pytest.fixture
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_pipeline_1a_to_1b(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Pipeline #1a → #1b: CSV → trained estimator → LUT bundle."""
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()

            # Step 1: Pipeline #1a output → CSVs
            _write_profiling_csvs(synthetic_profiling_data, profiling_dir)
            assert (profiling_dir / "token_ops.csv").exists()
            assert (profiling_dir / "attention.csv").exists()

            # Step 2: Pipeline #1b → Train estimator from CSVs
            est = RandomForestEstimator(n_estimators=20, random_state=42)
            scores = est.fit_from_csv([
                (profiling_dir / "token_ops.csv", OperatorCategory.LINEAR),
            ])
            assert est.is_fitted()
            assert scores["time_train_r2"] > 0.5

            # Step 3: Verify estimator makes reasonable predictions
            result = est.estimate(OperatorCategory.LINEAR, batch_size=1, seq_len=256)
            assert result.time_s > 0

    def test_simulator_with_estimator(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Pipeline #2: Run simulator with trained estimator."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.estimator.random_forest import RandomForestEstimator
        from inference_simulator.scheduler.vllm import VLLMScheduler

        # Train estimator
        est = RandomForestEstimator(n_estimators=20, random_state=42)
        est.fit(synthetic_profiling_data)

        # Run simulator
        sim = EventDrivenSimulator(
            model_spec=qwen3_8b_spec,
            hardware_spec=a100_spec,
            inference_spec=InferenceSpec(),
            scheduler=VLLMScheduler(),
            estimator=est,
        )
        metrics = sim.run(
            workload_spec=WorkloadSpec(qps=5.0, avg_input_tokens=64, avg_output_tokens=16),
            duration_s=2.0,
            seed=42,
        )

        assert metrics.total_requests > 0
        assert metrics.ttft_p50 > 0
        assert metrics.e2e_p50 > 0
        assert metrics.throughput_tps > 0
        assert metrics.total_energy_j > 0

    def test_search_with_sla(self, qwen3_8b_spec, a100_spec):
        """Pipeline #3: Search with SLA constraints."""
        from inference_search.cli import run_search
        from inference_search.types import SLAConstraint, SearchConfig

        config = SearchConfig(
            model_specs=[qwen3_8b_spec],
            hardware_specs=[a100_spec],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[
                SLAConstraint("ttft_s", 2.0, "max"),
            ],
            optimization_targets=["throughput_tps"],
        )
        result = run_search(config)

        assert len(result.all_results) >= 1
        assert result.all_results[0].max_qps > 0
        assert len(result.pareto_frontier) >= 1

    def test_full_e2e(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Full end-to-end: #1a → #1b → #2 → #3."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.estimator.random_forest import RandomForestEstimator
        from inference_simulator.scheduler.vllm import VLLMScheduler
        from inference_search.cli import run_search
        from inference_search.types import SLAConstraint, SearchConfig

        # 1. Train estimator (Pipeline #1b) on profiling data (#1a output)
        est = RandomForestEstimator(n_estimators=20, random_state=42)
        est.fit(synthetic_profiling_data)
        assert est.is_fitted()

        # 2. Run simulator (Pipeline #2) with trained estimator
        sim = EventDrivenSimulator(
            model_spec=qwen3_8b_spec,
            hardware_spec=a100_spec,
            inference_spec=InferenceSpec(),
            scheduler=VLLMScheduler(),
            estimator=est,
        )
        metrics = sim.run(
            workload_spec=WorkloadSpec(qps=5.0, avg_input_tokens=64, avg_output_tokens=16),
            duration_s=2.0,
            seed=42,
        )

        # Verify simulator metrics
        assert metrics.total_requests > 0
        assert metrics.ttft_p50 > 0
        assert metrics.e2e_p50 > metrics.ttft_p50  # E2E always > TTFT
        assert metrics.total_energy_j > 0

        # 3. Run search (Pipeline #3) with SLA constraints
        config = SearchConfig(
            model_specs=[qwen3_8b_spec],
            hardware_specs=[a100_spec],
            inference_specs=[InferenceSpec()],
            workload_spec=WorkloadSpec(),
            sla_constraints=[
                SLAConstraint("ttft_s", 2.0, "max"),
                SLAConstraint("throughput_tps", 1.0, "min"),
            ],
            optimization_targets=["throughput_tps"],
        )
        search_result = run_search(config)

        assert len(search_result.all_results) >= 1
        assert search_result.pareto_frontier  # Non-empty
        assert search_result.elapsed_seconds > 0

    def test_metrics_ipw_ipj(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Verify IPW/IPJ/cost metrics are computed when accuracy is provided."""
        from inference_simulator.metrics.collector import MetricsCollector
        from inference_simulator.request.request import Request, RequestState

        collector = MetricsCollector()
        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=100,
            max_output_tokens=50,
            state=RequestState.COMPLETED,
            tokens_generated=50,
            first_token_ns=100_000_000,
            completion_ns=1_000_000_000,
        )
        collector.record_request(req)
        collector.set_energy(300.0)
        collector.set_total_time(1.0)

        metrics = collector.compute(
            accuracy_score=0.85,
            price_per_gpu_hour_usd=3.50,
            num_gpus=1,
        )

        assert metrics.accuracy_score == 0.85
        assert metrics.ipw > 0  # accuracy / power
        assert metrics.ipj > 0  # accuracy / energy_per_query
        assert metrics.cost_per_query_usd > 0


# ---------------------------------------------------------------------------
# Helper: write per-category CSVs matching LUTGenerator csv_category_map
# ---------------------------------------------------------------------------

def _write_category_csvs(measurements, output_dir):
    """Write measurements to per-category CSV files (linear.csv, etc.)."""
    category_rows = {}
    for m in measurements:
        cat_name = m.category.value
        if cat_name not in category_rows:
            category_rows[cat_name] = []
        category_rows[cat_name].append({
            "operator_name": m.operator_name,
            "batch_size": m.batch_size,
            "seq_len": m.seq_len,
            "time_s": m.time_s,
            "energy_j": m.energy_j if m.energy_j is not None else "",
            "power_w": m.power_w if m.power_w is not None else "",
            "flops": m.flops if m.flops is not None else "",
        })

    fieldnames = [
        "operator_name", "batch_size", "seq_len",
        "time_s", "energy_j", "power_w", "flops",
    ]
    for cat_name, rows in category_rows.items():
        csv_path = output_dir / f"{cat_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class TestPipelineE2EExtended:
    """Extended E2E tests covering estimate CLI, compare-estimators,
    orchestrator, and ML-backed search."""

    @pytest.fixture
    def _skip_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_cli_estimate_e2e(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Use CliRunner to invoke 'estimate' with synthetic CSVs."""
        from click.testing import CliRunner
        from dataset_generator.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_profiling_data, profiling_dir)

            runner = CliRunner()
            result = runner.invoke(cli, [
                "estimate",
                "--profiling-dir", str(profiling_dir),
                "--output-dir", str(lut_dir),
                "--model", "Qwen/Qwen3-8B",
                "--hardware", "a100_80gb",
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "LUT bundle generated" in result.output
            assert (lut_dir / "gpu_token_ops.npz").exists()
            assert (lut_dir / "attention_prefill.npz").exists()
            assert (lut_dir / "attention_decode.npz").exists()

    def test_cli_compare_estimators_e2e(
        self, _skip_no_sklearn, synthetic_profiling_data
    ):
        """Use CliRunner for compare-estimators command."""
        from click.testing import CliRunner
        from dataset_generator.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()

            _write_category_csvs(synthetic_profiling_data, profiling_dir)

            runner = CliRunner()
            result = runner.invoke(cli, [
                "compare-estimators",
                "--profiling-dir", str(profiling_dir),
                "--output-dir", str(Path(tmpdir) / "comparison"),
            ])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "Comparing" in result.output
            assert "Best estimator" in result.output

    def test_full_pipeline_via_orchestrator(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Use PipelineOrchestrator.run_all() to chain #1b -> #2 -> #3."""
        from dataset_generator.pipeline.orchestrator import (
            PipelineConfig,
            PipelineOrchestrator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            _write_category_csvs(synthetic_profiling_data, profiling_dir)

            config = PipelineConfig(
                model_id="Qwen/Qwen3-8B",
                hardware_key="a100_80gb",
                profiling_dir=profiling_dir,
                lut_dir=lut_dir,
                accuracy_score=0.9,
                price_per_gpu_hour_usd=3.50,
                max_ttft=5.0,  # SLA needed so QPS search converges
            )

            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_all(config)

            assert "lut_bundle" in result
            assert result["lut_bundle"].exists()
            assert "simulation_metrics" in result
            assert "search_result" in result
            assert len(result["search_result"].all_results) >= 1

    def test_search_with_ml_oracle_e2e(
        self, _skip_no_sklearn, synthetic_profiling_data, qwen3_8b_spec, a100_spec
    ):
        """Create LUT bundle from CSVs, then run search with MLBackedOracle."""
        from inference_simulator.estimator.lut_generator import LUTGenerator
        from inference_search.cli import run_search
        from inference_search.ml_oracle import MLBackedOracle
        from inference_search.types import SLAConstraint, SearchConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            profiling_dir = Path(tmpdir) / "profiles"
            profiling_dir.mkdir()
            lut_dir = Path(tmpdir) / "luts"

            # Generate LUT bundle from synthetic CSVs
            _write_category_csvs(synthetic_profiling_data, profiling_dir)
            generator = LUTGenerator()
            bundle = generator.generate_full_bundle(
                profiling_dir, lut_dir, qwen3_8b_spec, a100_spec
            )
            assert bundle.exists()

            # Use the LUT bundle with MLBackedOracle
            oracle = MLBackedOracle(
                lut_bundle_dir=lut_dir,
                accuracy_score=0.85,
                price_per_hour_usd=3.50,
            )

            config = SearchConfig(
                model_specs=[qwen3_8b_spec],
                hardware_specs=[a100_spec],
                inference_specs=[InferenceSpec()],
                workload_spec=WorkloadSpec(qps=1.0, avg_input_tokens=64, avg_output_tokens=16),
                sla_constraints=[SLAConstraint("ttft_s", 5.0, "max")],
                optimization_targets=["throughput_tps"],
            )
            search_result = run_search(config, oracle=oracle)

            assert len(search_result.all_results) >= 1
            assert search_result.elapsed_seconds > 0
