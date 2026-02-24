"""Tests for scaling-law Step 3/4/5 helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from ipw.scaling_laws.aggregate import aggregate_summaries
from ipw.scaling_laws.fit import fit_scaling_laws
from ipw.scaling_laws import mini_pilot as mini_pilot_mod
from ipw.scaling_laws.mini_pilot import run_mini_pilot
from ipw.scaling_laws.sweep import (
    _apply_default_client_params,
    _find_summary_path,
    _is_oom_output,
    _quant_client_params,
    iter_points,
)


def test_aggregate_summaries_parses_expected_layout(tmp_path: Path) -> None:
    summary_dir = (
        tmp_path
        / "h100"
        / "qwen-qwen3-8b"
        / "fp16"
        / "B16_Sin2048_Sout256"
    )
    summary_dir.mkdir(parents=True)
    payload = {
        "model": "Qwen/Qwen3-8B",
        "totals": {"total_energy_j": 123.4},
        "profiler_config": {"client_params": {"dtype": "float16"}},
        "phase_summary": {
            "prefill": {
                "total_energy_j": 90.0,
                "mean_power_w": 300.0,
                "mean_duration_ms": 400.0,
                "mean_energy_per_input_token_j": 0.01,
            },
            "decode": {
                "total_energy_j": 33.4,
                "mean_power_w": 280.0,
                "mean_duration_ms": 200.0,
                "mean_energy_per_output_token_j": 0.05,
            },
        },
    }
    (summary_dir / "summary.json").write_text(json.dumps(payload))

    frame = aggregate_summaries(results_root=tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["gpu"] == "h100"
    assert row["batch_size"] == 16
    assert row["seq_in"] == 2048
    assert row["seq_out"] == 256
    assert math.isclose(float(row["active_params_b"]), 8.0)
    assert math.isclose(float(row["bytes_per_param"]), 2.0)
    assert math.isclose(float(row["E_prefill_j"]), 90.0)
    assert math.isclose(float(row["E_decode_j"]), 33.4)
    assert math.isclose(float(row["T_prefill_s"]), 0.4)
    assert math.isclose(float(row["T_decode_s"]), 0.2)


def test_fit_scaling_laws_recovers_high_r2_on_synthetic_data() -> None:
    rows = []
    for p in [1.7, 8.0, 14.0, 32.0]:
        for b in [1, 4, 16, 64]:
            for sin in [256, 1024, 4096, 8192]:
                for sout in [64, 256]:
                    for q in [2.0, 1.0]:
                        log_prefill = (
                            0.95 * math.log(p)
                            + 0.55 * math.log(b)
                            + 1.20 * math.log(sin)
                            + 0.80 * math.log(q)
                            + 0.5
                        )
                        log_decode = (
                            1.00 * math.log(p)
                            + 0.35 * math.log(b)
                            + 1.05 * math.log(sout)
                            + 0.10 * math.log(sin)
                            + 0.70 * math.log(q)
                            - 0.2
                        )
                        rows.append(
                            {
                                "gpu": "h100",
                                "active_params_b": p,
                                "batch_size": b,
                                "seq_in": sin,
                                "seq_out": sout,
                                "bytes_per_param": q,
                                "E_prefill_j": math.exp(log_prefill),
                                "E_decode_j": math.exp(log_decode),
                            }
                        )
    frame = pd.DataFrame(rows)
    report = fit_scaling_laws(
        frame,
        gpu="h100",
        cv_folds=5,
        cv_seed=42,
        heldout_model_b=14.0,
    )

    prefill_r2 = report["prefill"]["simple"]["r2"]
    decode_r2 = report["decode"]["simple"]["r2"]
    assert prefill_r2 is not None and prefill_r2 > 0.999
    assert decode_r2 is not None and decode_r2 > 0.999
    assert report["metadata"]["rows_fit"] == len(rows)


def test_sweep_helpers_quant_and_oom_detection() -> None:
    assert _quant_client_params("fp16", "float16") == {"dtype": "float16"}
    assert _quant_client_params("fp8", "float16") == {
        "quantization": "fp8",
        "dtype": "float16",
    }
    assert _quant_client_params("int4", None) == {"quantization": "int4"}

    assert _is_oom_output("CUDA out of memory. Tried to allocate")
    assert not _is_oom_output("all good")

    default_client = _apply_default_client_params(
        {},
        disable_prefix_caching=True,
    )
    assert default_client["enable_prefix_caching"] == "false"

    preserved_client = _apply_default_client_params(
        {"enable_prefix_caching": "true"},
        disable_prefix_caching=True,
    )
    assert preserved_client["enable_prefix_caching"] == "true"

    untouched_client = _apply_default_client_params(
        {},
        disable_prefix_caching=False,
    )
    assert "enable_prefix_caching" not in untouched_client


def test_iter_points_loop_order() -> None:
    points = list(
        iter_points(
            models=["m1", "m2"],
            quants=["fp16"],
            batches=[1, 8],
            seq_ins=[128],
            seq_outs=[64, 256],
        )
    )
    assert points[0].model == "m1"
    assert points[0].seq_out == 64
    assert points[0].batch == 1
    assert points[1].batch == 8
    assert points[-1].model == "m2"
    assert points[-1].seq_out == 256
    assert points[-1].batch == 8


def test_find_summary_path_detects_nested_layout(tmp_path: Path) -> None:
    config_dir = tmp_path / "B1_Sin256_Sout128"
    nested = config_dir / "profile_UNKNOWN_HW_qwen_qwen3_1_7b"
    nested.mkdir(parents=True)
    summary = nested / "summary.json"
    summary.write_text("{}")
    found = _find_summary_path(config_dir)
    assert found == summary


def test_mini_pilot_skip_sweep_runs_aggregate_and_fit(tmp_path: Path) -> None:
    summary_dir = (
        tmp_path
        / "h100"
        / "qwen-qwen3-8b"
        / "fp16"
        / "B1_Sin256_Sout128"
    )
    summary_dir.mkdir(parents=True)
    payload = {
        "model": "Qwen/Qwen3-8B",
        "totals": {"total_energy_j": 22.0},
        "profiler_config": {"client_params": {"dtype": "float16"}},
        "phase_summary": {
            "prefill": {
                "total_energy_j": 12.0,
                "mean_power_w": 240.0,
                "mean_duration_ms": 120.0,
                "mean_energy_per_input_token_j": 0.04,
            },
            "decode": {
                "total_energy_j": 10.0,
                "mean_power_w": 220.0,
                "mean_duration_ms": 100.0,
                "mean_energy_per_output_token_j": 0.08,
            },
        },
    }
    (summary_dir / "summary.json").write_text(json.dumps(payload))

    output_parquet = tmp_path / "mini_scaling_law_data.parquet"
    output_report = tmp_path / "mini_scaling_law_fit_report.json"
    args = argparse.Namespace(
        gpu_label="h100",
        model="Qwen/Qwen3-8B",
        results_root=str(tmp_path),
        quants="fp16",
        batches="1,8",
        seq_ins="256,1024",
        seq_outs="128",
        num_samples=4,
        max_queries=4,
        estimate_sec_per_config=60.0,
        dataset_param=[],
        client_param=[],
        quantized_dtype="float16",
        ipw_bin="ipw",
        sleep_seconds=0.0,
        stop_after_failures=1,
        disable_prefix_caching=True,
        resume=True,
        retry_failed=False,
        dry_run=False,
        skip_sweep=True,
        run_fit=True,
        fit_gpu_from_label=True,
        fit_gpu=None,
        cv_folds=2,
        cv_seed=42,
        heldout_model_b=14.0,
        output_parquet=str(output_parquet),
        output_report=str(output_report),
    )
    rc = run_mini_pilot(args)
    assert rc == 0
    assert output_parquet.exists()
    assert output_report.exists()

    report = json.loads(output_report.read_text())
    assert report["metadata"]["mini_pilot"] is True


def test_mini_pilot_dry_run_exits_before_aggregate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mini_pilot_mod, "run_sweep", lambda _: 0)

    args = argparse.Namespace(
        gpu_label="h100",
        model="Qwen/Qwen3-8B",
        results_root=str(tmp_path),
        quants="fp16",
        batches="1,8",
        seq_ins="256,1024",
        seq_outs="128",
        num_samples=4,
        max_queries=4,
        estimate_sec_per_config=60.0,
        dataset_param=[],
        client_param=[],
        quantized_dtype="float16",
        ipw_bin="ipw",
        sleep_seconds=0.0,
        stop_after_failures=1,
        disable_prefix_caching=True,
        resume=True,
        retry_failed=False,
        dry_run=True,
        skip_sweep=False,
        run_fit=True,
        fit_gpu_from_label=True,
        fit_gpu=None,
        cv_folds=2,
        cv_seed=42,
        heldout_model_b=14.0,
        output_parquet=str(tmp_path / "x.parquet"),
        output_report=str(tmp_path / "x.json"),
    )
    assert run_mini_pilot(args) == 0
