"""Tests for pipeline infrastructure: agent runner, distributions, checklist, tracking."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from dataset_generator.pipeline.agent_runner import AgentRunResult, AgentRunner
from dataset_generator.pipeline.distributions import (
    DistributionStats,
    WorkloadDistribution,
    compute_distribution_stats,
    compute_distributions,
    distributions_to_csv,
)
from dataset_generator.pipeline.checklist import (
    OPERATOR_CHECKLIST,
    get_checklist_status,
    get_total_operators,
    print_checklist,
)
from dataset_generator.pipeline.tracking import TrackingMatrix


# ---------------------------------------------------------------------------
# AgentRunResult dataclass tests
# ---------------------------------------------------------------------------


class TestAgentRunResult:
    def test_creation_minimal(self):
        result = AgentRunResult(
            query="Hello",
            response="Hi there",
            workload_type="chat",
            prefill_tokens=10,
            decode_tokens=20,
            num_steps=1,
            tool_calls=[],
            total_latency_s=0.5,
        )
        assert result.query == "Hello"
        assert result.response == "Hi there"
        assert result.workload_type == "chat"
        assert result.prefill_tokens == 10
        assert result.decode_tokens == 20
        assert result.num_steps == 1
        assert result.tool_calls == []
        assert result.total_latency_s == 0.5
        assert result.energy_j is None
        assert result.step_details == []

    def test_creation_with_all_fields(self):
        result = AgentRunResult(
            query="Solve x^2=4",
            response="x=2 or x=-2",
            workload_type="reasoning",
            prefill_tokens=50,
            decode_tokens=100,
            num_steps=3,
            tool_calls=["calculator", "think"],
            total_latency_s=2.5,
            energy_j=15.0,
            step_details=[{"type": "direct_api", "latency_s": 2.5}],
        )
        assert result.energy_j == 15.0
        assert len(result.step_details) == 1
        assert result.tool_calls == ["calculator", "think"]

    def test_default_mutable_fields_independent(self):
        r1 = AgentRunResult(
            query="a", response="b", workload_type="chat",
            prefill_tokens=1, decode_tokens=1, num_steps=1,
            tool_calls=[], total_latency_s=0.1,
        )
        r2 = AgentRunResult(
            query="c", response="d", workload_type="chat",
            prefill_tokens=1, decode_tokens=1, num_steps=1,
            tool_calls=[], total_latency_s=0.1,
        )
        r1.step_details.append({"x": 1})
        assert r2.step_details == []


# ---------------------------------------------------------------------------
# AgentRunner save/load roundtrip
# ---------------------------------------------------------------------------


class TestAgentRunnerIO:
    def _make_results(self) -> List[AgentRunResult]:
        return [
            AgentRunResult(
                query="What is 2+2?",
                response="4",
                workload_type="chat",
                prefill_tokens=10,
                decode_tokens=5,
                num_steps=1,
                tool_calls=["calculator"],
                total_latency_s=0.3,
                energy_j=1.5,
                step_details=[{"type": "direct_api"}],
            ),
            AgentRunResult(
                query="Write hello world",
                response="print('hello')",
                workload_type="agentic",
                prefill_tokens=20,
                decode_tokens=15,
                num_steps=2,
                tool_calls=["code_interpreter", "think"],
                total_latency_s=1.2,
            ),
        ]

    def test_save_and_load_roundtrip(self, tmp_path):
        results = self._make_results()
        runner = AgentRunner(dataset_name="test_ds")
        output_path = runner.save_results(results, tmp_path)

        assert output_path.exists()
        assert output_path.name == "test_ds_runs.jsonl"

        loaded = AgentRunner.load_results(output_path)
        assert len(loaded) == 2
        assert loaded[0].query == "What is 2+2?"
        assert loaded[0].prefill_tokens == 10
        assert loaded[0].tool_calls == ["calculator"]
        assert loaded[0].energy_j == 1.5
        assert loaded[1].workload_type == "agentic"
        assert loaded[1].num_steps == 2


# ---------------------------------------------------------------------------
# DistributionStats and compute_distribution_stats
# ---------------------------------------------------------------------------


class TestDistributionStats:
    def test_empty_values(self):
        stats = compute_distribution_stats([])
        assert stats.mean == 0.0
        assert stats.median == 0.0
        assert stats.std == 0.0
        assert stats.p99 == 0.0

    def test_single_value(self):
        stats = compute_distribution_stats([42.0])
        assert stats.mean == 42.0
        assert stats.median == 42.0
        assert stats.std == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.p50 == 42.0

    def test_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = compute_distribution_stats(values)
        assert stats.mean == 5.5
        assert stats.median == 5.5
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert stats.p50 == pytest.approx(5.5, abs=0.1)
        assert stats.p90 >= 9.0
        assert stats.p99 >= 9.5
        assert stats.std > 0

    def test_percentile_ordering(self):
        values = list(range(1, 101))
        stats = compute_distribution_stats([float(v) for v in values])
        assert stats.p50 <= stats.p90
        assert stats.p90 <= stats.p95
        assert stats.p95 <= stats.p99


# ---------------------------------------------------------------------------
# compute_distributions
# ---------------------------------------------------------------------------


class TestComputeDistributions:
    def _mock_results(self) -> List[AgentRunResult]:
        return [
            AgentRunResult(
                query=f"q{i}",
                response=f"r{i}",
                workload_type="chat",
                prefill_tokens=100 + i * 10,
                decode_tokens=50 + i * 5,
                num_steps=1 + (i % 3),
                tool_calls=["calc"] * (i % 2),
                total_latency_s=0.5 + i * 0.1,
            )
            for i in range(20)
        ]

    def test_basic_distributions(self):
        results = self._mock_results()
        dist = compute_distributions(results)

        assert dist.workload_type == "chat"
        assert dist.num_samples == 20
        assert dist.prefill_tokens.mean > 0
        assert dist.decode_tokens.mean > 0
        assert dist.num_steps.mean > 0
        assert dist.latency_s.mean > 0
        assert dist.prefill_tokens.min < dist.prefill_tokens.max

    def test_tool_type_counts(self):
        results = self._mock_results()
        dist = compute_distributions(results)
        assert "calc" in dist.tool_type_counts
        assert dist.tool_type_counts["calc"] == 10  # every other one

    def test_empty_results(self):
        dist = compute_distributions([])
        assert dist.num_samples == 0
        assert dist.prefill_tokens.mean == 0.0

    def test_distributions_to_csv(self, tmp_path):
        results = self._mock_results()
        dist = compute_distributions(results)
        csv_path = tmp_path / "dist.csv"
        distributions_to_csv(dist, csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5
        metric_names = {r["metric"] for r in rows}
        assert "prefill_tokens" in metric_names
        assert "decode_tokens" in metric_names
        assert "latency_s" in metric_names


# ---------------------------------------------------------------------------
# OPERATOR_CHECKLIST
# ---------------------------------------------------------------------------


class TestOperatorChecklist:
    def test_has_all_categories(self):
        expected = {
            "token_ops", "attention", "moe", "ssm", "mtp",
            "sampling", "communication", "cpu_host", "agentic",
        }
        assert set(OPERATOR_CHECKLIST.keys()) == expected

    def test_total_operators_at_least_60(self):
        total = get_total_operators()
        assert total >= 60, f"Expected >= 60 operators, got {total}"

    def test_get_checklist_status_empty_dir(self, tmp_path):
        status = get_checklist_status(tmp_path)
        for category, operators in status.items():
            assert category in OPERATOR_CHECKLIST
            for op_name, profiled in operators.items():
                assert profiled is False

    def test_get_checklist_status_with_csvs(self, tmp_path):
        # Create a fake profiling CSV with some operator names
        csv_path = tmp_path / "token_ops.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator_name", "time_s"])
            writer.writeheader()
            writer.writerow({"operator_name": "linear_qkv", "time_s": "0.001"})
            writer.writerow({"operator_name": "rmsnorm", "time_s": "0.0005"})

        status = get_checklist_status(tmp_path)
        assert status["token_ops"]["linear_qkv"] is True
        assert status["token_ops"]["rmsnorm"] is True
        assert status["token_ops"]["linear_o"] is False

    def test_print_checklist_output(self, tmp_path):
        status = get_checklist_status(tmp_path)
        output = print_checklist(status)
        assert "Operator Checklist" in output
        assert "[ ]" in output
        assert "token_ops" in output

    def test_print_checklist_with_profiled(self, tmp_path):
        csv_path = tmp_path / "ops.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator_name", "time_s"])
            writer.writeheader()
            writer.writerow({"operator_name": "embedding", "time_s": "0.001"})

        status = get_checklist_status(tmp_path)
        output = print_checklist(status)
        assert "[x] embedding" in output


# ---------------------------------------------------------------------------
# TrackingMatrix
# ---------------------------------------------------------------------------


class TestTrackingMatrix:
    def test_empty_directory(self, tmp_path):
        matrix = TrackingMatrix(
            datasets=["wildchat", "hotpotqa"],
            profiling_dir=tmp_path,
        )
        matrix.scan()
        assert matrix.completion_pct() == 0.0

    def test_scan_with_data(self, tmp_path):
        # Create dataset subdirectory with CSV
        ds_dir = tmp_path / "wildchat"
        ds_dir.mkdir()
        csv_path = ds_dir / "token_ops.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator_name", "time_s"])
            writer.writeheader()
            writer.writerow({"operator_name": "linear_qkv", "time_s": "0.001"})
            writer.writerow({"operator_name": "embedding", "time_s": "0.002"})

        matrix = TrackingMatrix(
            datasets=["wildchat", "hotpotqa"],
            profiling_dir=tmp_path,
        )
        matrix.scan()

        assert matrix.completion_pct() > 0.0
        md = matrix.to_markdown()
        assert "linear_qkv" in md
        assert "wildchat" in md

    def test_to_csv(self, tmp_path):
        ds_dir = tmp_path / "wildchat"
        ds_dir.mkdir()
        csv_prof = ds_dir / "ops.csv"
        with open(csv_prof, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator_name", "time_s"])
            writer.writeheader()
            writer.writerow({"operator_name": "linear_qkv", "time_s": "0.001"})

        matrix = TrackingMatrix(
            datasets=["wildchat"],
            profiling_dir=tmp_path,
        )
        matrix.scan()

        out_csv = tmp_path / "matrix.csv"
        matrix.to_csv(out_csv)
        assert out_csv.exists()

        with open(out_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Should have one row per operator
        assert len(rows) == get_total_operators()
        # linear_qkv should be marked as 1 for wildchat
        qkv_row = [r for r in rows if r["operator"] == "linear_qkv"][0]
        assert qkv_row["wildchat"] == "1"

    def test_to_markdown_format(self, tmp_path):
        matrix = TrackingMatrix(
            datasets=["ds1", "ds2"],
            profiling_dir=tmp_path,
        )
        matrix.scan()
        md = matrix.to_markdown()
        # Should have header with dataset names
        assert "ds1" in md
        assert "ds2" in md
        assert "| Operator |" in md
        # Check for separator line
        assert "---" in md
