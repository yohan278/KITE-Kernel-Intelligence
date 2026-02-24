"""CLI interface for grid evaluation runner."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click

from grid_eval.config import (
    AgentType,
    BenchmarkType,
    GridConfig,
    GpuType,
    HardwareConfig,
    ModelType,
    ResourceConfig,
)
from grid_eval.runner import GridEvalRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_enum_list(value: str, enum_class) -> List:
    """Parse comma-separated string into list of enum values."""
    if not value:
        return list(enum_class)

    items = [v.strip() for v in value.split(",")]
    result = []
    for item in items:
        try:
            result.append(enum_class(item))
        except ValueError:
            valid = [e.value for e in enum_class]
            raise click.BadParameter(
                f"Invalid value '{item}'. Valid options: {valid}"
            )
    return result


@click.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results. Default: results/grid_eval_YYYYMMDD_HHMMSS",
)
@click.option(
    "--gpu-types",
    "-g",
    type=str,
    default="",
    help="Comma-separated GPU types (a100_80gb,h100_80gb,mi300x,m4_max,m4_pro,m3_max,m3_pro). Default: all",
)
@click.option(
    "--resource-configs",
    "-r",
    type=str,
    default="",
    help="Comma-separated resource configs (1gpu_8cpu,4gpu_32cpu). Default: all",
)
@click.option(
    "--benchmarks",
    "-b",
    type=str,
    default="",
    help="Comma-separated benchmarks (hle,gaia). Default: all",
)
@click.option(
    "--models",
    "-m",
    type=str,
    default="",
    help="Comma-separated models (qwen3-8b,gpt-oss-20b). Default: all",
)
@click.option(
    "--agents",
    "-a",
    type=str,
    default="",
    help="Comma-separated agents (react,openhands). Default: all",
)
@click.option(
    "--hardware",
    "-h",
    type=str,
    default="",
    help="DEPRECATED: Use --gpu-types and --resource-configs instead. "
         "Legacy comma-separated hardware configs (a100_1gpu,a100_4gpu).",
)
@click.option(
    "--queries",
    "-q",
    type=int,
    default=100,
    help="Number of queries per benchmark. Default: 100",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    help="Random seed for reproducibility. Default: 42",
)
@click.option(
    "--vllm-url",
    type=str,
    default="http://localhost:8000",
    help="Base URL for vLLM server. Default: http://localhost:8000",
)
@click.option(
    "--ollama-url",
    type=str,
    default="http://localhost:11434",
    help="Base URL for Ollama server (Apple Silicon). Default: http://localhost:11434",
)
@click.option(
    "--openai-url",
    type=str,
    default=None,
    help="Base URL for OpenAI-compatible API (optional)",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "vllm", "ollama"]),
    default="auto",
    help="Inference backend: auto (select based on GPU vendor), vllm, ollama. Default: auto",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show configuration without running",
)
@click.option(
    "--full-dataset",
    is_flag=True,
    help="Use full dataset instead of --queries limit",
)
@click.option(
    "--conflict-policy",
    type=click.Choice(["fail", "kill", "skip"]),
    default="kill",
    help="How to handle vLLM port conflicts: fail (error), kill (take over), skip (use next port). Default: kill",
)
@click.option(
    "--no-cleanup",
    is_flag=True,
    help="Skip cleanup of orphaned vLLM servers on startup",
)
@click.option(
    "--grader-model",
    type=str,
    default="gpt-5-mini-2025-08-07",
    help="Model to use for LLM judge scoring. Default: gpt-5-mini-2025-08-07",
)
@click.option(
    "--grader-api-key",
    type=str,
    default=None,
    help="API key for grader model. Default: uses OPENAI_API_KEY env var",
)
@click.option(
    "--use-exact-match",
    is_flag=True,
    help="Bypass LLM judge and use exact string matching for scoring",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=1,
    help="Number of parallel workers for running queries (default: 1 = sequential). "
         "Higher values speed up runs but disable per-query energy attribution.",
)
@click.option(
    "--cuda-device",
    type=str,
    default=None,
    help="Override CUDA_VISIBLE_DEVICES for 1-GPU resource configs. "
         "Use with --vllm-url and different ports to run parallel processes on different GPUs.",
)
@click.option(
    "--batch",
    is_flag=True,
    help="Batched parallel eval with per-model batch sizes and amortized energy.",
)
@click.option(
    "--include-cloud-tools",
    is_flag=True,
    help="Include cloud LLM tools (gpt-4o, claude, gemini) in agent toolset. "
         "Default: excluded to prevent contamination of local model scores.",
)
def main(
    output_dir: Optional[Path],
    gpu_types: str,
    resource_configs: str,
    benchmarks: str,
    models: str,
    agents: str,
    hardware: str,
    queries: int,
    seed: int,
    vllm_url: str,
    ollama_url: str,
    openai_url: Optional[str],
    backend: str,
    dry_run: bool,
    full_dataset: bool,
    conflict_policy: str,
    no_cleanup: bool,
    grader_model: str,
    grader_api_key: Optional[str],
    use_exact_match: bool,
    workers: int,
    cuda_device: Optional[str],
    batch: bool,
    include_cloud_tools: bool,
) -> None:
    """Run grid search evaluation across benchmarks, models, agents, and hardware.

    Grid Search Loop Order (outermost to innermost):
        1. GPU Type (hardware choice) - e.g., a100_80gb, h100_80gb, m4_max
        2. Resource Config (allocation) - e.g., 1gpu_8cpu, 4gpu_32cpu
        3. Agent (harness) - e.g., react, openhands
        4. Model (LM) - e.g., qwen3-8b, gpt-oss-20b
        5. Benchmark (innermost) - e.g., gaia, hle

    Examples:

        # Full grid with NVIDIA GPUs
        python -m grid_eval.cli --output-dir results/grid_eval \\
            --gpu-types a100_80gb \\
            --resource-configs 1gpu_8cpu,4gpu_32cpu

        # Subset for testing
        python -m grid_eval.cli \\
            --gpu-types a100_80gb \\
            --resource-configs 1gpu_8cpu \\
            --benchmarks hle \\
            --models qwen3-8b \\
            --agents react \\
            --queries 5

        # Apple Silicon with Ollama (auto-detected backend)
        python -m grid_eval.cli \\
            --gpu-types m4_max \\
            --resource-configs 1gpu_8cpu \\
            --models qwen3-4b \\
            --benchmarks hle \\
            --queries 5

        # Force Ollama backend on any hardware
        python -m grid_eval.cli \\
            --backend ollama \\
            --models qwen3-4b \\
            --benchmarks hle

        # Dry run to see config
        python -m grid_eval.cli --dry-run
    """
    # Override CUDA device for 1-GPU configs if specified
    if cuda_device is not None:
        from grid_eval.config import RESOURCE_CONFIG_REGISTRY
        for rc, settings in RESOURCE_CONFIG_REGISTRY.items():
            if settings.get("gpu_count") == 1:
                settings["CUDA_VISIBLE_DEVICES"] = cuda_device
        click.echo(f"GPU override: 1-GPU configs will use CUDA device {cuda_device}")

    # Parse enum lists (new API)
    gpu_type_list = parse_enum_list(gpu_types, GpuType)
    resource_config_list = parse_enum_list(resource_configs, ResourceConfig)
    benchmark_list = parse_enum_list(benchmarks, BenchmarkType)
    model_list = parse_enum_list(models, ModelType)
    agent_list = parse_enum_list(agents, AgentType)

    # Parse legacy hardware configs (if provided)
    hardware_list = parse_enum_list(hardware, HardwareConfig) if hardware else []
    if hardware:
        click.echo(
            "WARNING: --hardware is deprecated. Use --gpu-types and --resource-configs instead.",
            err=True,
        )

    # Warn if --batch and --workers are both set
    if batch and workers > 1:
        click.echo(
            "WARNING: --batch takes precedence over --workers. "
            "Batch sizes are per-model from MODEL_REGISTRY.",
            err=True,
        )

    # Create config with new API
    # use_full_datasets=True ignores --queries limit; default is to apply limit
    config = GridConfig(
        gpu_types=gpu_type_list,
        resource_configs=resource_config_list,
        benchmarks=benchmark_list,
        models=model_list,
        agents=agent_list,
        hardware_configs=hardware_list if hardware else [
            HardwareConfig.A100_1GPU,
            HardwareConfig.A100_4GPU,
        ],
        queries_per_benchmark=queries,
        seed=seed,
        use_full_datasets=full_dataset,
        grader_model=grader_model,
        grader_api_key=grader_api_key,
        use_exact_match=use_exact_match,
        num_workers=workers,
        batch_mode=batch,
        include_cloud_tools=include_cloud_tools,
    )

    # Show configuration
    click.echo(config.describe())
    click.echo()

    if dry_run:
        click.echo("Dry run - exiting without running evaluation")
        click.echo()
        click.echo("Combinations that would be run (5-tuple: gpu/resource/agent/model/benchmark):")
        for i, (g, r, a, m, b) in enumerate(config.get_all_combinations(), 1):
            click.echo(f"  {i}. {g.value} / {r.value} / {a.value} / {m.value} / {b.value}")
        return

    # Set default output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/grid_eval_{timestamp}")

    click.echo(f"Output directory: {output_dir}")
    click.echo()

    # Map CLI policy to runner policy
    policy_map = {"fail": "fail", "kill": "kill", "skip": "skip_port"}
    runner_policy = policy_map.get(conflict_policy, "kill")

    # Create and run evaluation
    runner = GridEvalRunner(
        config=config,
        vllm_base_url=vllm_url,
        openai_base_url=openai_url,
        ollama_base_url=ollama_url,
        conflict_policy=runner_policy,
        cleanup_orphans=not no_cleanup,
        backend=backend,
        grader_model=grader_model,
        grader_api_key=grader_api_key,
    )

    try:
        runner.run(output_dir=output_dir)
        click.echo()
        click.echo(f"Evaluation complete! Results written to: {output_dir}")
    except KeyboardInterrupt:
        click.echo()
        click.echo("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Evaluation failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
