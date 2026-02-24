"""Tests for grid_eval.output module."""

import json
import tempfile
from pathlib import Path

import pytest

from grid_eval.output import (
    ConfigSummary,
    GridMetadata,
    JSONLWriter,
    QueryResult,
)


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_creation(self):
        result = QueryResult(
            query_id="test_001",
            benchmark="hle",
            model="qwen3-8b",
            agent="react",
            gpu_type="a100_80gb",
            resource_config="1gpu_8cpu",
            hardware="a100_80gb/1gpu_8cpu",
            avg_joules=10.5,
            max_power_watts=200.0,
            latency_seconds=5.2,
            tools_used={"calculator": 2},
            turns=3,
            models_called={"qwen3-8b": 3},
            is_correct=True,
            response="42",
            ground_truth="42",
            error=None,
        )
        assert result.query_id == "test_001"
        assert result.is_correct is True
        assert result.error is None
        assert result.gpu_type == "a100_80gb"
        assert result.resource_config == "1gpu_8cpu"
        assert result.hardware == "a100_80gb/1gpu_8cpu"

    def test_with_error(self):
        result = QueryResult(
            query_id="test_002",
            benchmark="gaia",
            model="gpt-oss-20b",
            agent="openhands",
            gpu_type="h100_80gb",
            resource_config="4gpu_32cpu",
            hardware="h100_80gb/4gpu_32cpu",
            avg_joules=0.0,
            max_power_watts=0.0,
            latency_seconds=0.5,
            tools_used={},
            turns=0,
            models_called={},
            is_correct=False,
            response="",
            ground_truth="expected",
            error="Connection timeout",
        )
        assert result.is_correct is False
        assert result.error == "Connection timeout"
        assert result.gpu_type == "h100_80gb"


class TestConfigSummary:
    """Tests for ConfigSummary dataclass."""

    def test_creation(self):
        summary = ConfigSummary(
            benchmark="hle",
            model="qwen3-8b",
            agent="react",
            gpu_type="a100_80gb",
            resource_config="1gpu_8cpu",
            hardware="a100_80gb/1gpu_8cpu",
            num_queries=100,
            accuracy=0.85,
            avg_joules=12.5,
            avg_latency_seconds=4.2,
            max_power_watts=250.0,
            total_joules=1250.0,
            total_latency_seconds=420.0,
        )
        assert summary.accuracy == 0.85
        assert summary.num_queries == 100
        assert summary.gpu_type == "a100_80gb"
        assert summary.resource_config == "1gpu_8cpu"


class TestGridMetadata:
    """Tests for GridMetadata dataclass."""

    def test_creation(self):
        metadata = GridMetadata(
            gpu_types=["a100_80gb", "h100_80gb"],
            resource_configs=["1gpu_8cpu", "4gpu_32cpu"],
            benchmarks=["hle", "gaia"],
            models=["qwen3-8b", "gpt-oss-20b"],
            agents=["react", "openhands"],
            hardware_configs=["a100_80gb/1gpu_8cpu", "a100_80gb/4gpu_32cpu"],
            queries_per_benchmark=100,
            seed=42,
            timestamp="2025-01-01T00:00:00",
            total_combinations=16,
            total_queries=1600,
        )
        assert len(metadata.benchmarks) == 2
        assert metadata.seed == 42
        assert len(metadata.gpu_types) == 2
        assert len(metadata.resource_configs) == 2


class TestJSONLWriter:
    """Tests for JSONLWriter class."""

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                assert writer.results_path.exists() or writer._file_handle is not None

    def test_write_single_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                result = QueryResult(
                    query_id="test_001",
                    benchmark="hle",
                    model="qwen3-8b",
                    agent="react",
                    gpu_type="a100_80gb",
                    resource_config="1gpu_8cpu",
                    hardware="a100_80gb/1gpu_8cpu",
                    avg_joules=10.5,
                    max_power_watts=200.0,
                    latency_seconds=5.2,
                    tools_used={"calc": 1},
                    turns=2,
                    models_called={"m1": 2},
                    is_correct=True,
                    response="42",
                    ground_truth="42",
                )
                writer.write_query_result(result)

            # Verify file contents
            with open(writer.results_path) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["query_id"] == "test_001"
                assert data["is_correct"] is True
                assert data["gpu_type"] == "a100_80gb"
                assert data["resource_config"] == "1gpu_8cpu"

    def test_write_multiple_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                for i in range(5):
                    result = QueryResult(
                        query_id=f"test_{i:03d}",
                        benchmark="hle",
                        model="qwen3-8b",
                        agent="react",
                        gpu_type="a100_80gb",
                        resource_config="1gpu_8cpu",
                        hardware="a100_80gb/1gpu_8cpu",
                        avg_joules=float(i),
                        max_power_watts=100.0,
                        latency_seconds=1.0,
                        tools_used={},
                        turns=1,
                        models_called={},
                        is_correct=i % 2 == 0,
                        response=str(i),
                        ground_truth=str(i) if i % 2 == 0 else "wrong",
                    )
                    writer.write_query_result(result)

            # Verify file has 5 lines
            with open(writer.results_path) as f:
                lines = f.readlines()
                assert len(lines) == 5

    def test_write_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                # Write some results
                for i in range(3):
                    result = QueryResult(
                        query_id=f"test_{i}",
                        benchmark="hle",
                        model="qwen3-8b",
                        agent="react",
                        gpu_type="a100_80gb",
                        resource_config="1gpu_8cpu",
                        hardware="a100_80gb/1gpu_8cpu",
                        avg_joules=10.0,
                        max_power_watts=100.0,
                        latency_seconds=1.0,
                        tools_used={},
                        turns=1,
                        models_called={},
                        is_correct=i == 0,  # Only first one correct
                        response="r",
                        ground_truth="g",
                    )
                    writer.write_query_result(result)

                writer.write_summary()

            # Verify summary
            with open(writer.summary_path) as f:
                summary = json.load(f)
                assert summary["total_queries"] == 3
                assert abs(summary["overall_accuracy"] - 1 / 3) < 0.01

    def test_write_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                metadata = GridMetadata(
                    gpu_types=["a100_80gb"],
                    resource_configs=["1gpu_8cpu"],
                    benchmarks=["hle"],
                    models=["qwen3-8b"],
                    agents=["react"],
                    hardware_configs=["a100_80gb/1gpu_8cpu"],
                    queries_per_benchmark=100,
                    seed=42,
                    timestamp="2025-01-01T00:00:00",
                    total_combinations=1,
                    total_queries=100,
                )
                writer.write_metadata(metadata)

            # Verify metadata
            with open(writer.metadata_path) as f:
                meta = json.load(f)
                assert meta["seed"] == 42
                assert meta["total_queries"] == 100
                assert meta["gpu_types"] == ["a100_80gb"]
                assert meta["resource_configs"] == ["1gpu_8cpu"]

    def test_finalize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                result = QueryResult(
                    query_id="test_001",
                    benchmark="hle",
                    model="qwen3-8b",
                    agent="react",
                    gpu_type="a100_80gb",
                    resource_config="1gpu_8cpu",
                    hardware="a100_80gb/1gpu_8cpu",
                    avg_joules=10.0,
                    max_power_watts=100.0,
                    latency_seconds=1.0,
                    tools_used={},
                    turns=1,
                    models_called={},
                    is_correct=True,
                    response="r",
                    ground_truth="r",
                )
                writer.write_query_result(result)

                metadata = GridMetadata(
                    gpu_types=["a100_80gb"],
                    resource_configs=["1gpu_8cpu"],
                    benchmarks=["hle"],
                    models=["qwen3-8b"],
                    agents=["react"],
                    hardware_configs=["a100_80gb/1gpu_8cpu"],
                    queries_per_benchmark=1,
                    seed=42,
                    timestamp="2025-01-01",
                    total_combinations=1,
                    total_queries=1,
                )
                writer.finalize(metadata)

            # Verify all files exist
            assert writer.results_path.exists()
            assert writer.summary_path.exists()
            assert writer.metadata_path.exists()

    def test_aggregation_multiple_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                # Results for config 1
                for i in range(2):
                    writer.write_query_result(
                        QueryResult(
                            query_id=f"cfg1_{i}",
                            benchmark="hle",
                            model="qwen3-8b",
                            agent="react",
                            gpu_type="a100_80gb",
                            resource_config="1gpu_8cpu",
                            hardware="a100_80gb/1gpu_8cpu",
                            avg_joules=10.0,
                            max_power_watts=100.0,
                            latency_seconds=1.0,
                            tools_used={},
                            turns=1,
                            models_called={},
                            is_correct=True,
                            response="r",
                            ground_truth="r",
                        )
                    )

                # Results for config 2
                for i in range(2):
                    writer.write_query_result(
                        QueryResult(
                            query_id=f"cfg2_{i}",
                            benchmark="gaia",
                            model="gpt-oss-20b",
                            agent="openhands",
                            gpu_type="h100_80gb",
                            resource_config="4gpu_32cpu",
                            hardware="h100_80gb/4gpu_32cpu",
                            avg_joules=20.0,
                            max_power_watts=200.0,
                            latency_seconds=2.0,
                            tools_used={},
                            turns=1,
                            models_called={},
                            is_correct=False,
                            response="wrong",
                            ground_truth="right",
                        )
                    )

                writer.write_summary()

            # Verify summary has 2 config entries
            with open(writer.summary_path) as f:
                summary = json.load(f)
                assert len(summary["configs"]) == 2
                assert summary["total_queries"] == 4
                assert summary["overall_accuracy"] == 0.5


class TestAccuracyExcludesUnscored:
    """Tests that accuracy calculations exclude is_correct=None results."""

    def _make_result(self, query_id: str, is_correct, benchmark="hle"):
        return QueryResult(
            query_id=query_id,
            benchmark=benchmark,
            model="qwen3-8b",
            agent="react",
            gpu_type="a100_80gb",
            resource_config="1gpu_8cpu",
            hardware="a100_80gb/1gpu_8cpu",
            avg_joules=10.0,
            max_power_watts=100.0,
            latency_seconds=1.0,
            tools_used={},
            turns=1,
            models_called={},
            is_correct=is_correct,
            response="r",
            ground_truth="g",
        )

    def test_accuracy_excludes_none_is_correct(self):
        """1 True + 1 False + 1 None → accuracy 0.5 (not 0.333)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                writer.write_query_result(self._make_result("q1", True))
                writer.write_query_result(self._make_result("q2", False))
                writer.write_query_result(self._make_result("q3", None))
                writer.write_summary()

            with open(writer.summary_path) as f:
                summary = json.load(f)
                assert summary["total_queries"] == 3
                assert abs(summary["overall_accuracy"] - 0.5) < 1e-9

    def test_all_none_gives_zero_accuracy(self):
        """All unscored results → accuracy 0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                writer.write_query_result(self._make_result("q1", None))
                writer.write_query_result(self._make_result("q2", None))
                writer.write_summary()

            with open(writer.summary_path) as f:
                summary = json.load(f)
                assert summary["overall_accuracy"] == 0.0

    def test_config_summary_accuracy_excludes_none(self):
        """Per-config aggregation should also exclude None from accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                writer.write_query_result(self._make_result("q1", True))
                writer.write_query_result(self._make_result("q2", None))
                writer.write_query_result(self._make_result("q3", None))
                writer.write_summary()

            with open(writer.summary_path) as f:
                summary = json.load(f)
                cfg = summary["configs"][0]
                # 1 scored, 1 correct → 100%
                assert abs(cfg["accuracy"] - 1.0) < 1e-9


class TestQueryResultNewFields:
    """Tests for new per-step resource metric fields on QueryResult."""

    def test_new_fields_default_none(self):
        """New fields default to None when not provided."""
        result = QueryResult(
            query_id="test_001",
            benchmark="hle",
            model="qwen3-8b",
            agent="react",
            gpu_type="a100_80gb",
            resource_config="1gpu_8cpu",
            hardware="a100_80gb/1gpu_8cpu",
            avg_joules=10.0,
            max_power_watts=200.0,
            latency_seconds=5.0,
            tools_used={},
            turns=1,
            models_called={},
            is_correct=True,
            response="42",
            ground_truth="42",
        )
        assert result.gpu_energy_joules is None
        assert result.cpu_energy_joules is None
        assert result.gpu_max_power_watts is None
        assert result.gpu_avg_power_watts is None
        assert result.cpu_max_power_watts is None
        assert result.cpu_avg_power_watts is None
        assert result.gpu_compute_utilization_pct_avg is None
        assert result.gpu_compute_utilization_pct_max is None
        assert result.gpu_memory_bw_utilization_pct_avg is None
        assert result.gpu_memory_bw_utilization_pct_max is None
        assert result.gpu_tensor_core_utilization_pct_avg is None
        assert result.gpu_tensor_core_utilization_pct_max is None
        assert result.total_cost_usd is None
        assert result.cost_by_model is None

    def test_new_fields_populated(self):
        """New fields can be set explicitly."""
        result = QueryResult(
            query_id="test_002",
            benchmark="gaia",
            model="gpt-oss-20b",
            agent="orchestrator",
            gpu_type="a100_80gb",
            resource_config="4gpu_32cpu",
            hardware="a100_80gb/4gpu_32cpu",
            avg_joules=50.0,
            max_power_watts=300.0,
            latency_seconds=10.0,
            tools_used={"calculator": 1},
            turns=3,
            models_called={"gpt-4o": 2},
            is_correct=True,
            response="answer",
            ground_truth="answer",
            gpu_energy_joules=45.0,
            cpu_energy_joules=5.0,
            gpu_max_power_watts=250.0,
            gpu_avg_power_watts=200.0,
            cpu_max_power_watts=60.0,
            cpu_avg_power_watts=50.0,
            gpu_compute_utilization_pct_avg=85.0,
            gpu_compute_utilization_pct_max=95.0,
            total_cost_usd=0.08,
            cost_by_model={"gpt-4o": 0.08},
        )
        assert result.gpu_energy_joules == 45.0
        assert result.cpu_energy_joules == 5.0
        assert result.gpu_max_power_watts == 250.0
        assert result.total_cost_usd == 0.08
        assert result.cost_by_model == {"gpt-4o": 0.08}


class TestConfigSummaryNewFields:
    """Tests for new fields on ConfigSummary."""

    def test_new_fields_default_none(self):
        """New fields default to None."""
        summary = ConfigSummary(
            benchmark="hle",
            model="qwen3-8b",
            agent="react",
            gpu_type="a100_80gb",
            resource_config="1gpu_8cpu",
            hardware="a100_80gb/1gpu_8cpu",
            num_queries=10,
            accuracy=0.8,
            avg_joules=10.0,
            avg_latency_seconds=2.0,
            max_power_watts=200.0,
            total_joules=100.0,
            total_latency_seconds=20.0,
        )
        assert summary.avg_gpu_joules is None
        assert summary.avg_cpu_joules is None
        assert summary.total_gpu_joules is None
        assert summary.total_cpu_joules is None
        assert summary.gpu_max_power_watts is None
        assert summary.avg_cost_usd is None
        assert summary.total_cost_usd is None


class TestAggregationNewFields:
    """Tests for aggregation of new fields in _aggregate_results."""

    def test_aggregation_with_new_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                writer.write_query_result(
                    QueryResult(
                        query_id="q1",
                        benchmark="hle",
                        model="qwen3-8b",
                        agent="orchestrator",
                        gpu_type="a100_80gb",
                        resource_config="1gpu_8cpu",
                        hardware="a100_80gb/1gpu_8cpu",
                        avg_joules=40.0,
                        max_power_watts=300.0,
                        latency_seconds=5.0,
                        tools_used={},
                        turns=2,
                        models_called={},
                        is_correct=True,
                        response="r",
                        ground_truth="r",
                        gpu_energy_joules=35.0,
                        cpu_energy_joules=5.0,
                        gpu_max_power_watts=250.0,
                        gpu_avg_power_watts=200.0,
                        cpu_max_power_watts=60.0,
                        cpu_avg_power_watts=50.0,
                        gpu_compute_utilization_pct_avg=80.0,
                        gpu_compute_utilization_pct_max=90.0,
                        total_cost_usd=0.05,
                    )
                )
                writer.write_query_result(
                    QueryResult(
                        query_id="q2",
                        benchmark="hle",
                        model="qwen3-8b",
                        agent="orchestrator",
                        gpu_type="a100_80gb",
                        resource_config="1gpu_8cpu",
                        hardware="a100_80gb/1gpu_8cpu",
                        avg_joules=60.0,
                        max_power_watts=350.0,
                        latency_seconds=8.0,
                        tools_used={},
                        turns=3,
                        models_called={},
                        is_correct=False,
                        response="wrong",
                        ground_truth="right",
                        gpu_energy_joules=50.0,
                        cpu_energy_joules=10.0,
                        gpu_max_power_watts=300.0,
                        gpu_avg_power_watts=240.0,
                        cpu_max_power_watts=70.0,
                        cpu_avg_power_watts=55.0,
                        gpu_compute_utilization_pct_avg=70.0,
                        gpu_compute_utilization_pct_max=95.0,
                        total_cost_usd=0.10,
                    )
                )

                writer.write_summary()

            with open(writer.summary_path) as f:
                summary = json.load(f)
                cfg = summary["configs"][0]

                # GPU/CPU energy split
                assert cfg["avg_gpu_joules"] == 42.5  # (35+50)/2
                assert cfg["avg_cpu_joules"] == 7.5
                assert cfg["total_gpu_joules"] == 85.0
                assert cfg["total_cpu_joules"] == 15.0

                # Power: max-of-maxes, mean-of-averages
                assert cfg["gpu_max_power_watts"] == 300.0
                assert cfg["gpu_avg_power_watts"] == 220.0  # (200+240)/2
                assert cfg["cpu_max_power_watts"] == 70.0
                assert cfg["cpu_avg_power_watts"] == 52.5

                # Utilization
                assert cfg["gpu_compute_utilization_pct_avg"] == 75.0
                assert cfg["gpu_compute_utilization_pct_max"] == 95.0

                # Cost
                assert abs(cfg["avg_cost_usd"] - 0.075) < 1e-9
                assert abs(cfg["total_cost_usd"] - 0.15) < 1e-9

    def test_aggregation_all_none_fields(self):
        """Aggregation works when new fields are all None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with JSONLWriter(output_dir) as writer:
                writer.write_query_result(
                    QueryResult(
                        query_id="q1",
                        benchmark="hle",
                        model="qwen3-8b",
                        agent="react",
                        gpu_type="a100_80gb",
                        resource_config="1gpu_8cpu",
                        hardware="a100_80gb/1gpu_8cpu",
                        avg_joules=10.0,
                        max_power_watts=100.0,
                        latency_seconds=1.0,
                        tools_used={},
                        turns=1,
                        models_called={},
                        is_correct=True,
                        response="r",
                        ground_truth="r",
                    )
                )

                writer.write_summary()

            with open(writer.summary_path) as f:
                summary = json.load(f)
                cfg = summary["configs"][0]
                assert cfg["avg_gpu_joules"] is None
                assert cfg["total_cost_usd"] is None
                assert cfg["gpu_max_power_watts"] is None
