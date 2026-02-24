"""CSV output writer for profiling measurements."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec


class ProfilingOutputWriter:
    """Writes profiling measurements to structured CSV files."""

    def write_token_ops(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write token-level operator measurements to CSV.

        Columns: operator_name, batch_size, seq_len, time_s, energy_j,
                 power_w, flops, bandwidth_gb_s
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "batch_size",
            "seq_len",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
            "bandwidth_gb_s",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                    "bandwidth_gb_s": (
                        m.bandwidth_gb_s if m.bandwidth_gb_s is not None else ""
                    ),
                })

    def write_attention(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write attention operator measurements to CSV.

        Columns: operator_name, variant, batch_size, seq_len, kv_cache_size,
                 time_s, energy_j, power_w, flops, bandwidth_gb_s
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "variant",
            "batch_size",
            "seq_len",
            "kv_cache_size",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
            "bandwidth_gb_s",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                variant = (
                    "prefill"
                    if m.category == OperatorCategory.ATTENTION_PREFILL
                    else "decode"
                )
                kv_cache_size = m.metadata.get("kv_cache_size", "")

                writer.writerow({
                    "operator_name": m.operator_name,
                    "variant": variant,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "kv_cache_size": kv_cache_size,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                    "bandwidth_gb_s": (
                        m.bandwidth_gb_s if m.bandwidth_gb_s is not None else ""
                    ),
                })

    def write_agentic(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write agentic tool measurements to CSV.

        Columns: tool_name, input_complexity, batch_size, time_s,
                 p50_s, p90_s, p99_s, energy_j, power_w
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "tool_name",
            "input_complexity",
            "batch_size",
            "time_s",
            "p50_s",
            "p90_s",
            "p99_s",
            "energy_j",
            "power_w",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "tool_name": m.metadata.get("tool_name", m.operator_name),
                    "input_complexity": m.metadata.get("complexity", ""),
                    "batch_size": m.batch_size,
                    "time_s": m.time_s,
                    "p50_s": m.metadata.get("p50_s", ""),
                    "p90_s": m.metadata.get("p90_s", ""),
                    "p99_s": m.metadata.get("p99_s", ""),
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                })

    def write_communication(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write communication profiling measurements to CSV.

        Columns: operation, num_gpus, message_size_bytes, time_s,
                 bandwidth_gb_s, energy_j, power_w
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operation",
            "num_gpus",
            "message_size_bytes",
            "time_s",
            "bandwidth_gb_s",
            "energy_j",
            "power_w",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operation": m.metadata.get("operation", m.operator_name),
                    "num_gpus": m.metadata.get("num_gpus", m.batch_size),
                    "message_size_bytes": m.metadata.get("message_size_bytes", m.seq_len),
                    "time_s": m.time_s,
                    "bandwidth_gb_s": (
                        m.bandwidth_gb_s if m.bandwidth_gb_s is not None else ""
                    ),
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                })

    def write_moe(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write MoE profiling measurements to CSV.

        Columns: operator_name, batch_size, seq_len, num_experts,
                 experts_per_token, time_s, energy_j, power_w, flops
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "batch_size",
            "seq_len",
            "num_experts",
            "experts_per_token",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "num_experts": m.metadata.get("num_experts", ""),
                    "experts_per_token": m.metadata.get("experts_per_token", ""),
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                })

    def write_sampling(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write sampling operator measurements to CSV.

        Columns: operator_name, batch_size, seq_len, time_s, energy_j,
                 power_w, flops
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "batch_size",
            "seq_len",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                })

    def write_ssm(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write SSM operator measurements to CSV.

        Columns: operator_name, batch_size, seq_len, time_s, energy_j,
                 power_w, flops, bandwidth_gb_s
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "batch_size",
            "seq_len",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
            "bandwidth_gb_s",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                    "bandwidth_gb_s": (
                        m.bandwidth_gb_s if m.bandwidth_gb_s is not None else ""
                    ),
                })

    def write_mtp(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write MTP (multi-token prediction) operator measurements to CSV.

        Columns: operator_name, batch_size, seq_len, num_draft_tokens,
                 time_s, energy_j, power_w, flops
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "batch_size",
            "seq_len",
            "num_draft_tokens",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "num_draft_tokens": m.metadata.get("num_draft_tokens", ""),
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                })

    def write_cpu_host(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write CPU host operator measurements to CSV.

        Columns: operator_name, batch_size, seq_len, time_s, energy_j,
                 power_w, bandwidth_gb_s
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "batch_size",
            "seq_len",
            "time_s",
            "energy_j",
            "power_w",
            "bandwidth_gb_s",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow({
                    "operator_name": m.operator_name,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "bandwidth_gb_s": (
                        m.bandwidth_gb_s if m.bandwidth_gb_s is not None else ""
                    ),
                })

    def write_combined_dataset(
        self,
        measurements: List[OperatorMeasurement],
        path: Path,
        model_spec: Optional[ModelSpec] = None,
    ) -> None:
        """Write ALL measurements with model dimension columns for cross-model training.

        Columns: operator_name, category, batch_size, seq_len, time_s,
                 energy_j, power_w, flops, bytes_accessed, bandwidth_gb_s,
                 hidden_dim, num_heads, num_kv_heads, head_dim,
                 intermediate_dim, num_layers, vocab_size, num_experts,
                 experts_per_token
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "operator_name",
            "category",
            "batch_size",
            "seq_len",
            "time_s",
            "energy_j",
            "power_w",
            "flops",
            "bytes_accessed",
            "bandwidth_gb_s",
            "hidden_dim",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_dim",
            "num_layers",
            "vocab_size",
            "num_experts",
            "experts_per_token",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                row = {
                    "operator_name": m.operator_name,
                    "category": m.category.value,
                    "batch_size": m.batch_size,
                    "seq_len": m.seq_len,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j if m.energy_j is not None else "",
                    "power_w": m.power_w if m.power_w is not None else "",
                    "flops": m.flops if m.flops is not None else "",
                    "bytes_accessed": (
                        m.bytes_accessed if m.bytes_accessed is not None else ""
                    ),
                    "bandwidth_gb_s": (
                        m.bandwidth_gb_s if m.bandwidth_gb_s is not None else ""
                    ),
                    "hidden_dim": "",
                    "num_heads": "",
                    "num_kv_heads": "",
                    "head_dim": "",
                    "intermediate_dim": "",
                    "num_layers": "",
                    "vocab_size": "",
                    "num_experts": "",
                    "experts_per_token": "",
                }
                if model_spec is not None:
                    row["hidden_dim"] = model_spec.hidden_dim
                    row["num_heads"] = model_spec.num_attention_heads
                    row["num_kv_heads"] = model_spec.num_kv_heads
                    row["head_dim"] = model_spec.head_dim
                    row["intermediate_dim"] = model_spec.intermediate_dim
                    row["num_layers"] = model_spec.num_layers
                    row["vocab_size"] = model_spec.vocab_size
                    row["num_experts"] = (
                        model_spec.num_experts
                        if model_spec.num_experts is not None
                        else ""
                    )
                    row["experts_per_token"] = (
                        model_spec.experts_per_token
                        if model_spec.experts_per_token is not None
                        else ""
                    )
                writer.writerow(row)

    def write_parquet(
        self, measurements: List[OperatorMeasurement], path: Path
    ) -> None:
        """Write all measurements to Parquet format (optional).

        Requires pyarrow to be installed.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet output. "
                "Install with: pip install pyarrow"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for m in measurements:
            rows.append({
                "operator_name": m.operator_name,
                "category": m.category.value,
                "batch_size": m.batch_size,
                "seq_len": m.seq_len,
                "time_s": m.time_s,
                "energy_j": m.energy_j,
                "power_w": m.power_w,
                "flops": m.flops,
                "bytes_accessed": m.bytes_accessed,
                "bandwidth_gb_s": m.bandwidth_gb_s,
            })

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(path))
