"""Workload model: project single-inference cost to full agentic workloads.

For agentic workloads (RAG, tool-use, multi-turn reasoning), the total
energy is more than a single inference call. This module wraps the
single-inference predictor with a workload multiplier:

  E_total = sum_turns[ E_prefill(ctx_k) + E_decode(out_k) ] + P_idle * T_idle

Where:
  - ctx_k grows each turn (accumulating conversation history + tool results)
  - T_idle = time spent on tool calls (web search, code execution)
  - The distribution of turns and tool calls per workload type is learned
    from grid_eval JSONL traces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ipw.simulator.hardware_specs import HardwareSpecs
from ipw.simulator.inference_model import predict as predict_single
from ipw.simulator.types import (
    CalibrationFactors,
    PhaseResult,
    SimulationResult,
    SingleInferenceResult,
    WorkloadProfile,
    WorkloadType,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkloadDistribution:
    """Learned distribution of workload characteristics per (benchmark, agent).

    Fitted from grid_eval JSONL traces.

    Attributes:
        benchmark: Benchmark name (e.g. "gaia", "hle").
        agent: Agent type (e.g. "react", "openhands").
        avg_turns: Average number of agent turns per query.
        std_turns: Standard deviation of turns.
        avg_tool_calls: Average tool calls per query.
        avg_tool_latency_s: Average latency per tool call in seconds.
        avg_input_tokens_per_turn: Average input tokens per turn.
        avg_output_tokens_per_turn: Average output tokens per turn.
        context_growth_per_turn: Tokens added to context each turn.
        sample_count: Number of queries used to fit this distribution.
    """

    benchmark: str = ""
    agent: str = ""
    avg_turns: float = 1.0
    std_turns: float = 0.0
    avg_tool_calls: float = 0.0
    avg_tool_latency_s: float = 1.0
    avg_input_tokens_per_turn: float = 500.0
    avg_output_tokens_per_turn: float = 200.0
    context_growth_per_turn: float = 300.0
    sample_count: int = 0


def fit_workload_distribution(
    jsonl_path: Path,
    benchmark: str = "",
    agent: str = "",
) -> Dict[str, WorkloadDistribution]:
    """Fit workload distributions from grid_eval JSONL traces.

    Groups QueryResult records by (benchmark, agent) and computes
    statistics on turns, tool calls, and latency.

    Args:
        jsonl_path: Path to grid_eval results JSONL file.
        benchmark: Filter to this benchmark (empty = all).
        agent: Filter to this agent (empty = all).

    Returns:
        Dictionary keyed by "benchmark/agent" with fitted distributions.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec_bench = rec.get("benchmark", "")
            rec_agent = rec.get("agent", "")

            if benchmark and rec_bench != benchmark:
                continue
            if agent and rec_agent != agent:
                continue

            key = f"{rec_bench}/{rec_agent}"
            if key not in groups:
                groups[key] = []
            groups[key].append(rec)

    distributions: Dict[str, WorkloadDistribution] = {}

    for key, records in groups.items():
        bench_name, agent_name = key.split("/", 1)

        turns_list = [r.get("turns", 1) for r in records]
        tool_counts = []
        for r in records:
            tools = r.get("tools_used", {})
            if isinstance(tools, dict):
                tool_counts.append(sum(tools.values()))
            elif isinstance(tools, list):
                tool_counts.append(len(tools))
            else:
                tool_counts.append(0)

        latencies = [r.get("latency_seconds", 0.0) for r in records]

        # Estimate per-turn tokens from total energy and turns
        # (crude approximation when per-action data is not available)
        avg_turns_val = float(np.mean(turns_list)) if turns_list else 1.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        # Estimate tool latency: if tools were used, approximate
        # tool time as a fraction of total latency
        avg_tools = float(np.mean(tool_counts)) if tool_counts else 0.0
        if avg_tools > 0 and avg_turns_val > 0:
            # Rough heuristic: tool calls take ~1s each on average
            avg_tool_lat = min(1.0, avg_latency / (avg_turns_val + avg_tools))
        else:
            avg_tool_lat = 0.0

        distributions[key] = WorkloadDistribution(
            benchmark=bench_name,
            agent=agent_name,
            avg_turns=avg_turns_val,
            std_turns=float(np.std(turns_list)) if len(turns_list) > 1 else 0.0,
            avg_tool_calls=avg_tools,
            avg_tool_latency_s=avg_tool_lat,
            sample_count=len(records),
        )

    return distributions


def project(
    hw: HardwareSpecs,
    active_params_b: float,
    workload: WorkloadProfile,
    bytes_per_param: float = 1.0,
    calibration: Optional[CalibrationFactors] = None,
    num_gpus: int = 1,
) -> SimulationResult:
    """Project single-inference cost to a full workload.

    For SINGLE_QUERY: wraps predict() directly.
    For agentic workloads: sums over turns with growing context,
    adds idle energy during tool calls.

    Args:
        hw: Hardware specifications.
        active_params_b: Active parameters in billions.
        workload: Workload description.
        bytes_per_param: Bytes per parameter.
        calibration: Optional calibration factors.
        num_gpus: Number of GPUs.

    Returns:
        SimulationResult with full workload prediction.
    """
    if workload.workload_type == WorkloadType.SINGLE_QUERY or workload.avg_turns <= 1:
        # Single inference - delegate directly
        single = predict_single(
            hw=hw,
            active_params_b=active_params_b,
            input_tokens=workload.avg_input_tokens,
            output_tokens=workload.avg_output_tokens,
            bytes_per_param=bytes_per_param,
            calibration=calibration,
            num_gpus=num_gpus,
        )
        return _single_to_result(single, workload)

    # Multi-turn agentic workload
    total_prefill_time = 0.0
    total_prefill_energy = 0.0
    total_decode_time = 0.0
    total_decode_energy = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    # Idle power during tool calls
    idle_power = hw.tdp_watts * num_gpus * 0.1  # ~10% TDP at idle

    for turn in range(workload.avg_turns):
        # Context grows each turn: initial input + accumulated history
        ctx_tokens = (
            workload.avg_input_tokens
            + turn * workload.context_growth_per_turn
        )
        out_tokens = workload.avg_output_tokens

        single = predict_single(
            hw=hw,
            active_params_b=active_params_b,
            input_tokens=ctx_tokens,
            output_tokens=out_tokens,
            bytes_per_param=bytes_per_param,
            calibration=calibration,
            num_gpus=num_gpus,
        )

        total_prefill_time += single.prefill.time_seconds
        total_prefill_energy += single.prefill.energy_joules
        total_decode_time += single.decode.time_seconds
        total_decode_energy += single.decode.energy_joules
        total_input_tokens += ctx_tokens
        total_output_tokens += out_tokens

    # Idle time from tool calls
    idle_time = workload.avg_tool_calls * workload.avg_tool_latency_seconds
    idle_energy = idle_power * idle_time

    total_time = total_prefill_time + total_decode_time + idle_time
    total_energy = total_prefill_energy + total_decode_energy + idle_energy

    return SimulationResult(
        total_energy_joules=total_energy,
        total_time_seconds=total_time,
        avg_power_watts=total_energy / total_time if total_time > 0 else 0.0,
        prefill_time_seconds=total_prefill_time,
        prefill_energy_joules=total_prefill_energy,
        decode_time_seconds=total_decode_time,
        decode_energy_joules=total_decode_energy,
        idle_time_seconds=idle_time,
        idle_energy_joules=idle_energy,
        num_turns=workload.avg_turns,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def _single_to_result(
    single: SingleInferenceResult,
    workload: WorkloadProfile,
) -> SimulationResult:
    """Convert a SingleInferenceResult to a SimulationResult."""
    return SimulationResult(
        total_energy_joules=single.total_energy_joules,
        total_time_seconds=single.total_time_seconds,
        avg_power_watts=single.avg_power_watts,
        prefill_time_seconds=single.prefill.time_seconds,
        prefill_energy_joules=single.prefill.energy_joules,
        decode_time_seconds=single.decode.time_seconds,
        decode_energy_joules=single.decode.energy_joules,
        idle_time_seconds=0.0,
        idle_energy_joules=0.0,
        num_turns=1,
        total_input_tokens=workload.avg_input_tokens,
        total_output_tokens=workload.avg_output_tokens,
    )


__all__ = [
    "WorkloadDistribution",
    "fit_workload_distribution",
    "project",
]
