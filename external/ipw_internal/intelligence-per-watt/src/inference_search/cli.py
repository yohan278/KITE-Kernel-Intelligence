"""CLI entry point for inference search."""

from __future__ import annotations

import json
import logging
import time
from typing import List

import click

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)

from inference_search.enumerator import enumerate_configurations
from inference_search.oracle import RooflineOracle
from inference_search.pareto import compute
from inference_search.qps_search import search as qps_search
from inference_search.types import SLAConstraint, SearchConfig, SearchResult

logger = logging.getLogger(__name__)


# Example model specs for CLI convenience — all 6 Qwen3 dense models
_EXAMPLE_MODELS = {
    "qwen3-0.6b": ModelSpec(
        model_id="Qwen/Qwen3-0.6B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=28,
        hidden_dim=1024,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=64,
        intermediate_dim=3072,
        vocab_size=151936,
    ),
    "qwen3-1.7b": ModelSpec(
        model_id="Qwen/Qwen3-1.7B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=28,
        hidden_dim=2048,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=6144,
        vocab_size=151936,
    ),
    "qwen3-4b": ModelSpec(
        model_id="Qwen/Qwen3-4B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=36,
        hidden_dim=2560,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=9216,
        vocab_size=151936,
    ),
    "qwen3-8b": ModelSpec(
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
    ),
    "qwen3-14b": ModelSpec(
        model_id="Qwen/Qwen3-14B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=48,
        hidden_dim=5120,
        num_attention_heads=40,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=13824,
        vocab_size=151936,
    ),
    "qwen3-32b": ModelSpec(
        model_id="Qwen/Qwen3-32B",
        architecture_type=ArchitectureType.DENSE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=64,
        hidden_dim=5120,
        num_attention_heads=40,
        num_kv_heads=8,
        head_dim=128,
        intermediate_dim=25600,
        vocab_size=151936,
    ),
    "qwen3-30b-a3b": ModelSpec(
        model_id="Qwen/Qwen3-30B-A3B",
        architecture_type=ArchitectureType.MOE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=48,
        hidden_dim=2048,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_dim=2560,
        vocab_size=151936,
        num_experts=128,
        experts_per_token=8,
    ),
    "glm-4.7-flash": ModelSpec(
        model_id="zai-org/GLM-4.7-Flash",
        architecture_type=ArchitectureType.MOE_TRANSFORMER,
        attention_type=AttentionType.GQA,
        num_layers=40,
        hidden_dim=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        intermediate_dim=2816,
        vocab_size=152064,
        num_experts=64,
        experts_per_token=6,
    ),
}

_EXAMPLE_HARDWARE = {
    "h100_80gb": HardwareSpec.from_registry("h100_80gb"),
    "a100_80gb": HardwareSpec.from_registry("a100_80gb"),
}

# Inference spec variants for different GPU counts
_EXAMPLE_INFERENCE_SPECS = {
    "1gpu-fp16": InferenceSpec(num_gpus=1, tensor_parallel=1, precision="fp16"),
    "1gpu-fp8": InferenceSpec(num_gpus=1, tensor_parallel=1, precision="fp8"),
    "8gpu-fp16": InferenceSpec(num_gpus=8, tensor_parallel=8, precision="fp16"),
    "8gpu-fp8": InferenceSpec(num_gpus=8, tensor_parallel=8, precision="fp8"),
}

# Workload type factory mapping
_WORKLOAD_FACTORIES = {
    "chat": WorkloadSpec.for_chat,
    "reasoning": WorkloadSpec.for_reasoning,
    "agentic": WorkloadSpec.for_agentic,
    "rag": WorkloadSpec.for_rag,
    "coding": WorkloadSpec.for_coding,
}


def _objective_direction(target: str) -> str:
    """Determine whether a metric should be maximized or minimized."""
    maximize_metrics = {"throughput_tps", "throughput_rps", "ipw", "ipj"}
    if target in maximize_metrics:
        return "maximize"
    return "minimize"


def run_search(
    search_config: SearchConfig,
    oracle: "SimulatorOracle | None" = None,
) -> SearchResult:
    """Execute the full search pipeline.

    1. Enumerate feasible configurations.
    2. For each, run QPS binary search (exhaustive or Bayesian).
    3. Compute Pareto frontier over optimization targets.

    Args:
        search_config: Full search specification.
        oracle: Optional simulator oracle. If None, uses RooflineOracle.

    Returns:
        SearchResult with all results and Pareto frontier.
    """
    if oracle is None:
        oracle = RooflineOracle(
            accuracy_score=search_config.accuracy_score,
            price_per_hour_usd=search_config.price_per_gpu_hour_usd,
        )
    start = time.monotonic()

    configs = enumerate_configurations(search_config)
    logger.info("Enumerated %d feasible configurations", len(configs))

    all_results = []
    total_sims = 0

    if search_config.search_method == "bayesian":
        from inference_search.bayesian_search import BayesianSearcher

        searcher = BayesianSearcher(oracle, search_config, configs)
        all_results = searcher.search()
        total_sims = len(all_results) * 20
    else:
        for model, hw, inf in configs:
            result = qps_search(
                model_spec=model,
                hardware_spec=hw,
                inference_spec=inf,
                workload_spec=search_config.workload_spec,
                sla_constraints=search_config.sla_constraints,
                simulator=oracle,
            )
            all_results.append(result)
            # Rough estimate: each binary search does ~20 iterations
            total_sims += 20

    # Compute Pareto frontier
    objectives = [
        (target, _objective_direction(target))
        for target in search_config.optimization_targets
    ]
    frontier = compute(all_results, objectives) if objectives else all_results

    elapsed = time.monotonic() - start

    return SearchResult(
        all_results=all_results,
        pareto_frontier=frontier,
        search_config=search_config,
        total_simulations=total_sims,
        elapsed_seconds=elapsed,
    )


@click.command("search")
@click.option(
    "--models",
    "-m",
    multiple=True,
    default=["qwen3-8b"],
    help="Model names to search over (e.g., qwen3-8b).",
)
@click.option(
    "--hardware",
    "-hw",
    multiple=True,
    default=["h100_80gb"],
    help="Hardware targets (e.g., h100_80gb, a100_80gb).",
)
@click.option("--max-ttft", type=float, default=2.0, help="Max TTFT P95 in seconds.")
@click.option("--max-tbt", type=float, default=0.1, help="Max TBT P95 in seconds.")
@click.option("--max-e2e", type=float, default=30.0, help="Max E2E P95 latency in seconds.")
@click.option("--min-throughput-tps", type=float, default=10.0, help="Min throughput (tokens/s).")
@click.option("--max-power", type=float, default=None, help="Max average power in watts.")
@click.option(
    "--input-tokens", type=int, default=500, help="Average input tokens per request."
)
@click.option(
    "--output-tokens", type=int, default=200, help="Average output tokens per request."
)
@click.option("--json-output", is_flag=True, help="Output results as JSON.")
@click.option(
    "--search-method",
    type=click.Choice(["exhaustive", "bayesian"]),
    default="exhaustive",
    help="Search strategy.",
)
@click.option(
    "--accuracy-score",
    type=float,
    default=1.0,
    help="Model accuracy score for IPW/IPJ computation.",
)
@click.option(
    "--price-per-gpu-hour",
    type=float,
    default=0.0,
    help="Price per GPU-hour in USD for cost estimation.",
)
@click.option(
    "--workload-type",
    type=click.Choice(["chat", "reasoning", "agentic", "rag", "coding"]),
    default=None,
    help="Workload type (uses predefined token distributions).",
)
@click.option(
    "--lut-bundle-dir",
    type=click.Path(exists=True),
    default=None,
    help="LUT bundle directory (uses ML-backed simulator instead of roofline).",
)
def main(
    models: tuple[str, ...],
    hardware: tuple[str, ...],
    max_ttft: float | None,
    max_tbt: float | None,
    max_e2e: float | None,
    min_throughput_tps: float | None,
    max_power: float | None,
    input_tokens: int,
    output_tokens: int,
    json_output: bool,
    search_method: str,
    accuracy_score: float,
    price_per_gpu_hour: float,
    workload_type: str | None,
    lut_bundle_dir: str | None,
) -> None:
    """Search for optimal inference configurations under SLA constraints."""
    logging.basicConfig(level=logging.INFO)

    # Build model specs
    model_specs: List[ModelSpec] = []
    for name in models:
        if name in _EXAMPLE_MODELS:
            model_specs.append(_EXAMPLE_MODELS[name])
        else:
            click.echo(f"Unknown model: {name}. Available: {list(_EXAMPLE_MODELS)}", err=True)
            raise SystemExit(1)

    # Build hardware specs
    hw_specs: List[HardwareSpec] = []
    for name in hardware:
        try:
            hw_specs.append(HardwareSpec.from_registry(name))
        except KeyError:
            click.echo(f"Unknown hardware: {name}", err=True)
            raise SystemExit(1)

    # Build SLA constraints — use P95 metric names for percentile-aware SLAs
    sla: List[SLAConstraint] = []
    if max_ttft is not None:
        sla.append(SLAConstraint("ttft_p95", max_ttft, "max"))
    if max_tbt is not None:
        sla.append(SLAConstraint("tbt_p95", max_tbt, "max"))
    if max_e2e is not None:
        sla.append(SLAConstraint("e2e_p95", max_e2e, "max"))
    if min_throughput_tps is not None:
        sla.append(SLAConstraint("throughput_tps", min_throughput_tps, "min"))
    if max_power is not None:
        sla.append(SLAConstraint("avg_power_w", max_power, "max"))

    # Build workload spec
    if workload_type is not None:
        factory = _WORKLOAD_FACTORIES[workload_type]
        workload = factory(qps=1.0)
    else:
        workload = WorkloadSpec(
            qps=1.0,
            avg_input_tokens=input_tokens,
            avg_output_tokens=output_tokens,
        )

    # Build inference specs: include 1-GPU and 8-GPU variants
    inference_specs = [
        _EXAMPLE_INFERENCE_SPECS["1gpu-fp16"],
        _EXAMPLE_INFERENCE_SPECS["8gpu-fp16"],
    ]

    # Build oracle
    search_oracle = None
    if lut_bundle_dir:
        from pathlib import Path
        from inference_search.ml_oracle import MLBackedOracle
        search_oracle = MLBackedOracle(
            lut_bundle_dir=Path(lut_bundle_dir),
            accuracy_score=accuracy_score,
            price_per_hour_usd=price_per_gpu_hour,
        )

    config = SearchConfig(
        model_specs=model_specs,
        hardware_specs=hw_specs,
        inference_specs=inference_specs,
        workload_spec=workload,
        sla_constraints=sla,
        search_method=search_method,
        accuracy_score=accuracy_score,
        price_per_gpu_hour_usd=price_per_gpu_hour,
    )

    result = run_search(config, oracle=search_oracle)

    if json_output:
        _print_json(result)
    else:
        _print_table(result)


def _print_table(result: SearchResult) -> None:
    """Print search results as a formatted table."""
    click.echo(f"\nSearch completed in {result.elapsed_seconds:.2f}s")
    click.echo(f"Configurations evaluated: {len(result.all_results)}")
    click.echo(f"Pareto frontier size: {len(result.pareto_frontier)}")
    click.echo()

    header = f"{'Model':<25} {'Hardware':<20} {'Max QPS':>10} {'TTFT (s)':>10} {'TBT (s)':>10} {'Power (W)':>10}"
    click.echo(header)
    click.echo("-" * len(header))

    for r in result.pareto_frontier:
        click.echo(
            f"{r.model_spec.model_id:<25} "
            f"{r.hardware_spec.name:<20} "
            f"{r.max_qps:>10.1f} "
            f"{r.metrics.get('ttft_s', 0):>10.4f} "
            f"{r.metrics.get('tbt_s', 0):>10.4f} "
            f"{r.metrics.get('avg_power_w', 0):>10.1f}"
        )


def _print_json(result: SearchResult) -> None:
    """Print search results as JSON."""
    output = {
        "elapsed_seconds": result.elapsed_seconds,
        "total_simulations": result.total_simulations,
        "num_configurations": len(result.all_results),
        "pareto_frontier": [
            {
                "model_id": r.model_spec.model_id,
                "hardware": r.hardware_spec.name,
                "max_qps": r.max_qps,
                "metrics": r.metrics,
                "sla_violations": r.sla_violations,
            }
            for r in result.pareto_frontier
        ],
    }
    click.echo(json.dumps(output, indent=2))


__all__ = ["main", "run_search"]
