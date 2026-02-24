"""Tests for dataset_generator CSV output writer."""

import csv
import pytest
from pathlib import Path

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from dataset_generator.profiler.output import ProfilingOutputWriter


@pytest.fixture
def writer():
    return ProfilingOutputWriter()


@pytest.fixture
def token_measurements():
    return [
        OperatorMeasurement(
            operator_name="linear_qkv",
            category=OperatorCategory.LINEAR,
            batch_size=1,
            seq_len=128,
            time_s=0.001,
            flops=1_000_000,
            bandwidth_gb_s=100.0,
        ),
        OperatorMeasurement(
            operator_name="rmsnorm",
            category=OperatorCategory.NORMALIZATION,
            batch_size=1,
            seq_len=128,
            time_s=0.0005,
        ),
    ]


@pytest.fixture
def attention_measurements():
    return [
        OperatorMeasurement(
            operator_name="attention_prefill",
            category=OperatorCategory.ATTENTION_PREFILL,
            batch_size=1,
            seq_len=128,
            time_s=0.002,
            flops=2_000_000,
        ),
        OperatorMeasurement(
            operator_name="attention_decode",
            category=OperatorCategory.ATTENTION_DECODE,
            batch_size=1,
            seq_len=256,
            time_s=0.001,
            metadata={"kv_cache_size": 256},
        ),
    ]


@pytest.fixture
def agentic_measurements():
    return [
        OperatorMeasurement(
            operator_name="tool_calculator",
            category=OperatorCategory.AGENTIC_TOOL,
            batch_size=1,
            seq_len=0,
            time_s=0.0001,
            metadata={
                "tool_name": "calculator",
                "complexity": "simple",
                "p50_s": 0.00008,
                "p90_s": 0.00012,
                "p99_s": 0.00015,
            },
        ),
    ]


class TestProfilingOutputWriter:
    def test_write_token_ops(self, writer, token_measurements, tmp_path):
        path = tmp_path / "token_ops.csv"
        writer.write_token_ops(token_measurements, path)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["operator_name"] == "linear_qkv"
        assert rows[0]["batch_size"] == "1"
        assert rows[0]["seq_len"] == "128"
        assert float(rows[0]["time_s"]) == pytest.approx(0.001)
        assert rows[1]["operator_name"] == "rmsnorm"

    def test_write_attention(self, writer, attention_measurements, tmp_path):
        path = tmp_path / "attention.csv"
        writer.write_attention(attention_measurements, path)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["variant"] == "prefill"
        assert rows[1]["variant"] == "decode"
        assert rows[1]["kv_cache_size"] == "256"

    def test_write_agentic(self, writer, agentic_measurements, tmp_path):
        path = tmp_path / "agentic.csv"
        writer.write_agentic(agentic_measurements, path)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["tool_name"] == "calculator"
        assert rows[0]["input_complexity"] == "simple"
        assert float(rows[0]["p50_s"]) == pytest.approx(0.00008)

    def test_csv_columns_token_ops(self, writer, token_measurements, tmp_path):
        path = tmp_path / "token_ops.csv"
        writer.write_token_ops(token_measurements, path)

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = [
            "operator_name", "batch_size", "seq_len", "time_s",
            "energy_j", "power_w", "flops", "bandwidth_gb_s",
        ]
        assert header == expected

    def test_csv_columns_attention(self, writer, attention_measurements, tmp_path):
        path = tmp_path / "attention.csv"
        writer.write_attention(attention_measurements, path)

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = [
            "operator_name", "variant", "batch_size", "seq_len",
            "kv_cache_size", "time_s", "energy_j", "power_w",
            "flops", "bandwidth_gb_s",
        ]
        assert header == expected

    def test_csv_columns_agentic(self, writer, agentic_measurements, tmp_path):
        path = tmp_path / "agentic.csv"
        writer.write_agentic(agentic_measurements, path)

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = [
            "tool_name", "input_complexity", "batch_size", "time_s",
            "p50_s", "p90_s", "p99_s", "energy_j", "power_w",
        ]
        assert header == expected

    def test_creates_parent_dirs(self, writer, token_measurements, tmp_path):
        path = tmp_path / "nested" / "dir" / "token_ops.csv"
        writer.write_token_ops(token_measurements, path)
        assert path.exists()
