"""Tests for Step-6 hardware normalization helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ipw.scaling_laws.hardware_normalize import estimate_hardware_normalization


def test_estimate_hardware_normalization_recovers_k_ratio_and_epsilons() -> None:
    k_prefill = 1.5
    k_decode = 2.0
    rows = []
    for batch, seq_in, seq_out in [
        (1, 1024, 256),
        (8, 1024, 256),
        (1, 2048, 256),
        (8, 2048, 256),
    ]:
        log_p = math.log(8.0)
        log_b = math.log(float(batch))
        log_sin = math.log(float(seq_in))
        log_sout = math.log(float(seq_out))
        log_q = math.log(2.0)

        log_prefill_source = 0.2 + 1.0 * log_p + 0.6 * log_b + 1.1 * log_sin + 0.8 * log_q
        log_decode_source = (
            -0.1 + 1.0 * log_p + 0.4 * log_b + 1.0 * log_sout + 0.1 * log_sin + 0.7 * log_q
        )
        rows.append(
            {
                "gpu": "a100",
                "active_params_b": 8.0,
                "batch_size": batch,
                "seq_in": seq_in,
                "seq_out": seq_out,
                "bytes_per_param": 2.0,
                "E_prefill_j": math.exp(log_prefill_source + math.log(k_prefill)),
                "E_decode_j": math.exp(log_decode_source + math.log(k_decode)),
            }
        )
    frame = pd.DataFrame(rows)

    fit_report = {
        "metadata": {"gpu_filter": "h100"},
        "prefill": {
            "simple": {
                "intercept": 0.2,
                "coefficients": {
                    "logP": 1.0,
                    "logB": 0.6,
                    "logSin": 1.1,
                    "logQ": 0.8,
                },
            }
        },
        "decode": {
            "simple": {
                "intercept": -0.1,
                "coefficients": {
                    "logP": 1.0,
                    "logB": 0.4,
                    "logSout": 1.0,
                    "logSin": 0.1,
                    "logQ": 0.7,
                },
            }
        },
    }

    source_bench = {
        "memory": {"total_pj_per_bit": 10.0},
        "compute": {"fp16": {"total_pj_per_flop": 2.0}},
        "gemm": {"fp16": {"total_pj_per_flop": 3.0}},
        "inference": {
            "gemm_prefill": {"fp16": 1.0},
            "gemm_decode": {"fp16": 1.5},
        },
    }
    target_bench = {
        "memory": {"total_pj_per_bit": 12.0},
        "compute": {"fp16": {"total_pj_per_flop": 1.5}},
        "gemm": {"fp16": {"total_pj_per_flop": 2.4}},
        "inference": {
            "gemm_prefill": {"fp16": 1.1},
            "gemm_decode": {"fp16": 1.8},
        },
    }

    report = estimate_hardware_normalization(
        frame,
        fit_report,
        target_gpu="a100",
        model_kind="simple",
        dtype="fp16",
        source_benchmark_parameters=source_bench,
        target_benchmark_parameters=target_bench,
    )

    assert report["metadata"]["source_gpu"] == "h100"
    assert report["metadata"]["rows_target"] == 4
    assert report["k_hw"]["prefill"]["n"] == 4
    assert report["k_hw"]["decode"]["n"] == 4
    assert report["k_hw"]["prefill"]["ratio_target_over_source"] == pytest.approx(k_prefill, rel=1e-6)
    assert report["k_hw"]["decode"]["ratio_target_over_source"] == pytest.approx(k_decode, rel=1e-6)

    eps_ratio = report["benchmark_epsilons"]["ratio_target_over_source"]
    assert eps_ratio["memory_total_pj_per_bit"] == pytest.approx(1.2, rel=1e-6)
    assert eps_ratio["compute_total_pj_per_flop"] == pytest.approx(0.75, rel=1e-6)
    assert eps_ratio["gemm_total_pj_per_flop"] == pytest.approx(0.8, rel=1e-6)
