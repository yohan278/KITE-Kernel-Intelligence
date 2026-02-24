"""CLI subcommand for the LLM inference simulator.

Usage:
    ipw simulate --gpu h100_80gb --model qwen3-8b \\
        --workload single_query --input-tokens 500 --output-tokens 200

    ipw simulate --gpu a100_80gb --model qwen3-8b \\
        --workload agentic_reasoning --turns 5 --tool-calls 3 \\
        --calibration calibration.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from ipw.cli._console import error, info, success
from ipw.simulator.hardware_specs import HARDWARE_SPECS_REGISTRY
from ipw.simulator.simulator import InferenceSimulator, format_result
from ipw.simulator.types import (
    SimulatorConfig,
    WorkloadProfile,
    WorkloadType,
)


@click.command(help="Simulate LLM inference energy and latency for a (hardware, model, workload) combination.")
@click.option(
    "--gpu", "-g",
    "gpu_type",
    required=True,
    type=click.Choice(sorted(HARDWARE_SPECS_REGISTRY.keys()), case_sensitive=False),
    help="GPU hardware type",
)
@click.option(
    "--model", "-m",
    "model_type",
    required=True,
    help="Model type (e.g. qwen3-8b, qwen3-4b)",
)
@click.option(
    "--resource-config", "-r",
    default="1gpu_8cpu",
    help="Resource allocation (e.g. 1gpu_8cpu, 4gpu_32cpu)",
)
@click.option(
    "--workload", "-w",
    "workload_type",
    type=click.Choice([w.value for w in WorkloadType], case_sensitive=False),
    default="single_query",
    help="Workload type",
)
@click.option(
    "--input-tokens", "-i",
    type=int,
    default=500,
    help="Average input tokens per inference call",
)
@click.option(
    "--output-tokens", "-o",
    type=int,
    default=200,
    help="Average output tokens per inference call",
)
@click.option(
    "--turns",
    type=int,
    default=1,
    help="Number of agent turns (for agentic workloads)",
)
@click.option(
    "--tool-calls",
    type=int,
    default=0,
    help="Number of tool calls (for agentic workloads)",
)
@click.option(
    "--tool-latency",
    type=float,
    default=1.0,
    help="Average tool call latency in seconds",
)
@click.option(
    "--context-growth",
    type=int,
    default=300,
    help="Tokens added to context per turn (for multi-turn workloads)",
)
@click.option(
    "--calibration",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to calibration JSON file",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Output results as JSON instead of human-readable text",
)
def simulate(
    gpu_type: str,
    model_type: str,
    resource_config: str,
    workload_type: str,
    input_tokens: int,
    output_tokens: int,
    turns: int,
    tool_calls: int,
    tool_latency: float,
    context_growth: int,
    calibration: Optional[Path],
    json_output: bool,
) -> None:
    """Simulate LLM inference energy and latency."""
    workload = WorkloadProfile(
        workload_type=WorkloadType(workload_type),
        avg_input_tokens=input_tokens,
        avg_output_tokens=output_tokens,
        avg_turns=turns,
        avg_tool_calls=tool_calls,
        avg_tool_latency_seconds=tool_latency,
        context_growth_per_turn=context_growth,
    )

    config = SimulatorConfig(
        gpu_type=gpu_type,
        model_type=model_type,
        resource_config=resource_config,
        workload=workload,
        calibration_path=str(calibration) if calibration else None,
    )

    simulator = InferenceSimulator()

    if calibration:
        try:
            simulator.load_calibration(calibration)
        except Exception as e:
            error(f"Failed to load calibration: {e}")
            raise click.Abort()

    try:
        result = simulator.simulate(config)
    except KeyError as e:
        error(str(e))
        raise click.Abort()

    if json_output:
        output = {
            "total_energy_joules": result.total_energy_joules,
            "total_time_seconds": result.total_time_seconds,
            "avg_power_watts": result.avg_power_watts,
            "prefill_time_seconds": result.prefill_time_seconds,
            "prefill_energy_joules": result.prefill_energy_joules,
            "decode_time_seconds": result.decode_time_seconds,
            "decode_energy_joules": result.decode_energy_joules,
            "idle_time_seconds": result.idle_time_seconds,
            "idle_energy_joules": result.idle_energy_joules,
            "num_turns": result.num_turns,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "confidence": result.confidence.value,
            "calibration_used": result.calibration_used,
            "metadata": result.metadata,
        }
        info(json.dumps(output, indent=2))
    else:
        info(format_result(result))


__all__ = ["simulate"]
