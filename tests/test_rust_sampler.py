"""Tests for the h100-energy-sampler Rust crate.

Validates:
  - Cargo.toml is well-formed and declares expected dependencies
  - Proto file defines all 11 NVML fields
  - Source files exist and contain expected structures
  - CSV output format matches expected schema
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLER_ROOT = PROJECT_ROOT / "tools" / "h100-energy-sampler"


class TestRustCrateStructure:
    """Verify the Rust crate is structurally complete."""

    def test_cargo_toml_exists(self):
        assert (SAMPLER_ROOT / "Cargo.toml").exists()

    def test_cargo_toml_has_nvml_dep(self):
        cargo = (SAMPLER_ROOT / "Cargo.toml").read_text()
        assert "nvml-wrapper" in cargo, "Missing nvml-wrapper dependency"
        assert "tonic" in cargo, "Missing tonic (gRPC) dependency"
        assert "prost" in cargo, "Missing prost (protobuf) dependency"
        assert "tokio" in cargo, "Missing tokio async runtime"

    def test_build_rs_exists(self):
        assert (SAMPLER_ROOT / "build.rs").exists()

    def test_source_files_exist(self):
        expected = ["main.rs", "sampler.rs", "stream.rs", "recorder.rs", "lib.rs"]
        for f in expected:
            path = SAMPLER_ROOT / "src" / f
            assert path.exists(), f"Missing source file: src/{f}"


class TestProtoDefinition:
    """Verify the protobuf schema defines all required NVML fields."""

    def test_proto_exists(self):
        assert (SAMPLER_ROOT / "proto" / "telemetry.proto").exists()

    def test_proto_has_all_nvml_fields(self):
        proto = (SAMPLER_ROOT / "proto" / "telemetry.proto").read_text()
        required_fields = [
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "temperature_c",
            "clock_sm_mhz",
            "clock_mem_mhz",
            "power_limit_w",
            "total_energy_mj",
            "memory_used_bytes",
            "pcie_tx_kbps",
            "pcie_rx_kbps",
            "throttle_reasons",
            "fan_speed_pct",
        ]
        for field in required_fields:
            assert field in proto, f"Missing proto field: {field}"

    def test_proto_has_grpc_service(self):
        proto = (SAMPLER_ROOT / "proto" / "telemetry.proto").read_text()
        assert "service H100TelemetryService" in proto
        assert "rpc Health" in proto
        assert "rpc StreamSamples" in proto

    def test_proto_has_health_response(self):
        proto = (SAMPLER_ROOT / "proto" / "telemetry.proto").read_text()
        assert "gpu_count" in proto
        assert "platform" in proto


class TestSamplerSource:
    """Verify sampler.rs collects all NVML fields."""

    def test_sampler_imports_nvml(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "nvml_wrapper" in src, "sampler.rs should use nvml_wrapper"

    def test_sampler_collects_utilization(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "utilization_rates" in src

    def test_sampler_collects_temperature(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "temperature" in src.lower()

    def test_sampler_collects_clocks(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "clock_info" in src.lower() or "Clock::Graphics" in src

    def test_sampler_collects_energy(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "total_energy_consumption" in src

    def test_sampler_collects_memory(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "memory_info" in src

    def test_sampler_collects_pcie(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "pcie_throughput" in src.lower() or "PcieUtilCounter" in src

    def test_sampler_collects_throttle(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "throttle_reasons" in src.lower() or "current_throttle_reasons" in src

    def test_sampler_collects_fan(self):
        src = (SAMPLER_ROOT / "src" / "sampler.rs").read_text()
        assert "fan_speed" in src


class TestRecorderSource:
    """Verify recorder.rs supports CSV and JSONL output."""

    def test_csv_headers(self):
        src = (SAMPLER_ROOT / "src" / "recorder.rs").read_text()
        assert "timestamp_nanos" in src
        assert "gpu_util_pct" in src or "gpu_utilization_pct" in src
        assert "power_draw_w" in src
        assert "energy_since_baseline_j" in src

    def test_jsonl_format(self):
        src = (SAMPLER_ROOT / "src" / "recorder.rs").read_text()
        assert "serde_json" in src
        assert "jsonl" in src.lower() or "Jsonl" in src

    def test_csv_format(self):
        src = (SAMPLER_ROOT / "src" / "recorder.rs").read_text()
        assert "Csv" in src or "csv" in src.lower()


class TestStreamSource:
    """Verify stream.rs implements gRPC service."""

    def test_implements_h100_telemetry_service(self):
        src = (SAMPLER_ROOT / "src" / "stream.rs").read_text()
        assert "H100TelemetryService" in src

    def test_implements_health(self):
        src = (SAMPLER_ROOT / "src" / "stream.rs").read_text()
        assert "health" in src.lower()
        assert "HealthResponse" in src

    def test_implements_stream_samples(self):
        src = (SAMPLER_ROOT / "src" / "stream.rs").read_text()
        assert "stream_samples" in src.lower() or "StreamSamples" in src


class TestCSVOutputSchema:
    """Test that CSV output format matches expectations (schema test)."""

    def test_expected_csv_columns(self):
        """The recorder's CSV header should contain all expected telemetry columns."""
        src = (SAMPLER_ROOT / "src" / "recorder.rs").read_text()
        expected = [
            "timestamp_nanos", "gpu_index",
            "temperature_c", "clock_sm_mhz", "clock_mem_mhz",
            "power_limit_w", "total_energy_mj",
            "memory_used_bytes", "memory_total_bytes",
            "pcie_tx_kbps", "pcie_rx_kbps",
            "throttle_reasons", "fan_speed_pct",
            "power_draw_w", "energy_since_baseline_j",
        ]
        for col in expected:
            assert col in src, f"Missing CSV column: {col}"
