"""CLI entry point for the dataset generator."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click

from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.types.operators import OperatorCategory
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.runner import ProfilingRunner


def _load_model_spec(model_id: str):
    """Load a ModelSpec for the given model ID."""
    model_lower = model_id.lower()
    if "qwen3" in model_lower or "qwen2" in model_lower:
        from dataset_generator.model_loader.qwen3 import Qwen3ModelLoader
        return Qwen3ModelLoader().load(model_id)
    else:
        from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
        return HuggingFaceModelLoader().load(model_id)


@click.group()
def cli():
    """Dataset Generator — operator-level profiling for LLM inference."""
    pass


@cli.command()
@click.option("--model", required=True, help="Model ID (e.g., Qwen/Qwen3-8B)")
@click.option("--hardware", required=True, help="Hardware key (e.g., a100_80gb)")
@click.option("--precision", default="fp16", help="Precision (fp16, fp8, bf16)")
@click.option(
    "--output-dir",
    default="data/profiles",
    type=click.Path(),
    help="Output directory for CSV files",
)
@click.option(
    "--profilers",
    default=None,
    help="Comma-separated profilers to run (token_ops,attention,agentic)",
)
@click.option(
    "--batch-sizes",
    default=None,
    help="Override batch sizes (comma-separated ints)",
)
@click.option(
    "--seq-lengths",
    default=None,
    help="Override sequence lengths (comma-separated ints)",
)
@click.option("--warmup", default=5, type=int, help="Warmup iterations")
@click.option("--iterations", default=20, type=int, help="Measurement iterations")
@click.option(
    "--energy/--no-energy",
    default=False,
    help="Enable energy measurement (requires energy monitor)",
)
def profile(
    model: str,
    hardware: str,
    precision: str,
    output_dir: str,
    profilers: Optional[str],
    batch_sizes: Optional[str],
    seq_lengths: Optional[str],
    warmup: int,
    iterations: int,
    energy: bool,
):
    """Run operator profiling for a model × hardware configuration."""
    click.echo(f"Loading model spec for {model}...")
    model_spec = _load_model_spec(model)
    click.echo(
        f"  {model_spec.num_layers} layers, {model_spec.hidden_dim} hidden, "
        f"{model_spec.num_attention_heads} heads, {model_spec.vocab_size} vocab"
    )

    click.echo(f"Loading hardware spec for {hardware}...")
    hw_spec = HardwareSpec.from_registry(hardware)
    click.echo(f"  {hw_spec.name}: {hw_spec.peak_fp16_tflops} FP16 TFLOPS")

    # Build sweep config
    sweep_kwargs = {
        "warmup_iterations": warmup,
        "measurement_iterations": iterations,
        "use_energy": energy,
    }
    if batch_sizes:
        sweep_kwargs["batch_sizes"] = [int(x) for x in batch_sizes.split(",")]
    if seq_lengths:
        sweep_kwargs["prefill_seq_lengths"] = [int(x) for x in seq_lengths.split(",")]

    sweep_config = SweepConfig(**sweep_kwargs)

    # Build profiler list
    profiler_list = None
    if profilers:
        profiler_list = [p.strip() for p in profilers.split(",")]

    # Run
    runner = ProfilingRunner(
        model_spec=model_spec,
        hardware_spec=hw_spec,
        sweep_config=sweep_config,
        output_dir=Path(output_dir),
        precision=precision,
    )
    result = runner.run(profilers=profiler_list)
    click.echo(f"Done. {result.num_measurements} measurements recorded.")


@cli.command("list-models")
def list_models():
    """Show available model loaders."""
    from dataset_generator.model_loader import (
        HuggingFaceModelLoader,
        Qwen3ModelLoader,
    )
    from dataset_generator.model_loader.gpt_oss import GPTOSSModelLoader
    from dataset_generator.model_loader.glm import GLMModelLoader
    from dataset_generator.model_loader.kimi import KimiModelLoader

    loaders = [
        HuggingFaceModelLoader,
        Qwen3ModelLoader,
        GPTOSSModelLoader,
        GLMModelLoader,
        KimiModelLoader,
    ]
    click.echo("Available model loaders:")
    for loader_cls in loaders:
        loader = loader_cls()
        archs = ", ".join(loader.supported_architectures)
        click.echo(f"  {loader_cls.__name__}: [{archs}]")


@cli.command("list-operators")
def list_operators():
    """Show available operator profiler categories."""
    click.echo("Operator categories:")
    for cat in OperatorCategory:
        click.echo(f"  {cat.value}")

    click.echo("\nAvailable profilers:")
    for name in ProfilingRunner.PROFILER_REGISTRY:
        click.echo(f"  {name}")


QWEN3_FAMILY: List[str] = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

WORKLOAD_PRESETS = {
    "chat": "for_chat",
    "reasoning": "for_reasoning",
    "agentic": "for_agentic",
    "rag": "for_rag",
}


@cli.command("profile-batch")
@click.option(
    "--models",
    default=None,
    help="Comma-separated model IDs. If omitted, uses Qwen3 family.",
)
@click.option("--hardware", required=True, help="Hardware key (e.g., a100_80gb)")
@click.option(
    "--workload",
    default=None,
    type=click.Choice(["chat", "reasoning", "agentic", "rag"]),
    help="Workload preset for sweep configuration",
)
@click.option(
    "--precisions",
    default="fp16",
    help="Comma-separated precisions (e.g., fp16,bf16)",
)
@click.option(
    "--output-dir",
    default="data/profiles",
    type=click.Path(),
    help="Output directory for CSV files",
)
@click.option(
    "--profilers",
    default=None,
    help="Comma-separated profilers to run",
)
@click.option("--warmup", default=5, type=int, help="Warmup iterations")
@click.option("--iterations", default=20, type=int, help="Measurement iterations")
@click.option(
    "--energy/--no-energy",
    default=False,
    help="Enable energy measurement",
)
def profile_batch(
    models: Optional[str],
    hardware: str,
    workload: Optional[str],
    precisions: str,
    output_dir: str,
    profilers: Optional[str],
    warmup: int,
    iterations: int,
    energy: bool,
):
    """Run operator profiling across multiple models and precisions."""
    model_ids = [m.strip() for m in models.split(",")] if models else QWEN3_FAMILY
    precision_list = [p.strip() for p in precisions.split(",")]

    click.echo(f"Loading hardware spec for {hardware}...")
    hw_spec = HardwareSpec.from_registry(hardware)
    click.echo(f"  {hw_spec.name}: {hw_spec.peak_fp16_tflops} FP16 TFLOPS")

    # Build sweep config
    sweep_kwargs = {
        "warmup_iterations": warmup,
        "measurement_iterations": iterations,
        "use_energy": energy,
    }
    if workload and workload in WORKLOAD_PRESETS:
        factory_method = WORKLOAD_PRESETS[workload]
        sweep_config = getattr(SweepConfig, factory_method)(**sweep_kwargs)
    else:
        sweep_config = SweepConfig(**sweep_kwargs)

    profiler_list = None
    if profilers:
        profiler_list = [p.strip() for p in profilers.split(",")]

    total_measurements = 0
    for model_id in model_ids:
        for precision in precision_list:
            click.echo(f"\n--- {model_id} @ {precision} ---")
            try:
                model_spec = _load_model_spec(model_id)
                click.echo(
                    f"  {model_spec.num_layers} layers, "
                    f"{model_spec.hidden_dim} hidden, "
                    f"{model_spec.num_attention_heads} heads"
                )
            except Exception as e:
                click.echo(f"  Failed to load model spec: {e}")
                continue

            runner = ProfilingRunner(
                model_spec=model_spec,
                hardware_spec=hw_spec,
                sweep_config=sweep_config,
                output_dir=Path(output_dir),
                precision=precision,
            )
            result = runner.run(profilers=profiler_list)
            total_measurements += result.num_measurements
            click.echo(f"  {result.num_measurements} measurements recorded.")

    click.echo(f"\nBatch complete. {total_measurements} total measurements.")


@cli.command("estimate")
@click.option("--profiling-dir", required=True, type=click.Path(exists=True),
              help="Directory containing profiling CSV files")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Output directory for LUT files")
@click.option("--model", required=True, help="Model ID (e.g., Qwen/Qwen3-8B)")
@click.option("--hardware", required=True, help="Hardware key (e.g., a100_80gb)")
def estimate(profiling_dir: str, output_dir: str, model: str, hardware: str):
    """Train estimators and generate LUT bundle from profiling CSVs (Pipeline 1b)."""
    from inference_simulator.estimator.lut_generator import LUTGenerator

    click.echo(f"Loading model spec for {model}...")
    model_spec = _load_model_spec(model)
    click.echo(f"  {model_spec.num_layers} layers, {model_spec.hidden_dim} hidden dim")

    click.echo(f"Loading hardware spec for {hardware}...")
    hw_spec = HardwareSpec.from_registry(hardware)
    click.echo(f"  {hw_spec.name}: {hw_spec.peak_fp16_tflops} FP16 TFLOPS")

    click.echo(f"\nTraining estimators from {profiling_dir}...")
    generator = LUTGenerator()
    bundle = generator.generate_full_bundle(
        Path(profiling_dir), Path(output_dir), model_spec, hw_spec
    )

    click.echo(f"\nLUT bundle generated in {output_dir}")
    click.echo(f"  Estimator: {bundle.metadata.get('estimator_class', 'unknown')}")
    scores = bundle.metadata.get('training_scores', {})
    if scores:
        for key, val in scores.items():
            if isinstance(val, float):
                click.echo(f"  {key}: {val:.4f}")
    click.echo(f"  Token ops LUT: {bundle.gpu_token_ops_lut}")
    click.echo(f"  Attention prefill LUT: {bundle.gpu_attention_prefill_lut}")
    click.echo(f"  Attention decode LUT: {bundle.gpu_attention_decode_lut}")
    if bundle.tool_distributions:
        click.echo(f"  Tool distributions: {bundle.tool_distributions}")


@cli.command("compare-estimators")
@click.option("--profiling-dir", required=True, type=click.Path(exists=True),
              help="Directory containing profiling CSV files")
@click.option("--output-dir", default="data/estimator_comparison", type=click.Path(),
              help="Output directory for comparison results")
@click.option("--per-category", is_flag=True, default=False,
              help="Show per-category R² breakdown")
def compare_estimators_cmd(profiling_dir: str, output_dir: str, per_category: bool):
    """Compare runtime estimator approaches on profiling data."""
    from inference_simulator.estimator.model_comparison import (
        compare_estimators as _compare,
        pick_best_estimator,
    )
    from inference_simulator.estimator.sklearn_base import (
        load_csv_measurements,
        load_csv_measurements_auto_category,
    )
    from inference_simulator.estimator.random_forest import RandomForestEstimator
    from inference_simulator.types.operators import OperatorCategory

    # Auto-detect CSV files
    csv_category_map = {
        "linear": OperatorCategory.LINEAR,
        "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
        "attention_decode": OperatorCategory.ATTENTION_DECODE,
        "embedding": OperatorCategory.EMBEDDING,
        "normalization": OperatorCategory.NORMALIZATION,
        "activation": OperatorCategory.ACTIVATION,
        "moe_routing": OperatorCategory.MOE_ROUTING,
        "moe_expert": OperatorCategory.MOE_EXPERT,
        "ssm_scan": OperatorCategory.SSM_SCAN,
        "communication": OperatorCategory.COMMUNICATION,
        "agentic_tool": OperatorCategory.AGENTIC_TOOL,
    }

    # Operator name -> category mapping for combined CSVs
    operator_name_to_category = {
        "linear_qkv": OperatorCategory.LINEAR,
        "linear_o": OperatorCategory.LINEAR,
        "mlp_up": OperatorCategory.LINEAR,
        "mlp_gate": OperatorCategory.LINEAR,
        "mlp_down": OperatorCategory.LINEAR,
        "lm_head": OperatorCategory.LINEAR,
        "embedding": OperatorCategory.EMBEDDING,
        "rotary_embedding": OperatorCategory.EMBEDDING,
        "layernorm": OperatorCategory.NORMALIZATION,
        "rmsnorm": OperatorCategory.NORMALIZATION,
        "gelu_activation": OperatorCategory.ACTIVATION,
        "silu_activation": OperatorCategory.ACTIVATION,
        "softmax": OperatorCategory.ACTIVATION,
        "dropout": OperatorCategory.ACTIVATION,
        "cross_entropy_loss": OperatorCategory.ACTIVATION,
        "residual_add": OperatorCategory.ACTIVATION,
        "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
        "attention_decode": OperatorCategory.ATTENTION_DECODE,
        "sliding_window_attention": OperatorCategory.ATTENTION_DECODE,
        "kv_cache_append": OperatorCategory.KV_CACHE,
        "kv_cache_evict": OperatorCategory.KV_CACHE,
        "mqa_gqa_expansion": OperatorCategory.KV_CACHE,
    }
    combined_csvs = {"token_ops", "attention", "agentic", "sampling",
                     "communication", "moe", "ssm", "mtp", "cpu_host"}

    profiling_path = Path(profiling_dir)
    all_measurements = []
    found_csvs = []
    for name, cat in csv_category_map.items():
        csv_file = profiling_path / f"{name}.csv"
        if csv_file.exists():
            ms = load_csv_measurements(csv_file, cat)
            all_measurements.extend(ms)
            found_csvs.append(f"{name}.csv ({len(ms)} rows)")

    # Also load combined CSVs
    for csv_file in profiling_path.glob("*.csv"):
        stem = csv_file.stem.lower()
        if stem in combined_csvs:
            ms = load_csv_measurements_auto_category(csv_file, operator_name_to_category)
            all_measurements.extend(ms)
            found_csvs.append(f"{stem}.csv ({len(ms)} rows, combined)")

    if not all_measurements:
        click.echo(f"No profiling CSVs found in {profiling_dir}")
        return

    click.echo(f"Loaded {len(all_measurements)} measurements from:")
    for csv_name in found_csvs:
        click.echo(f"  {csv_name}")

    # Collect estimator classes
    estimator_classes = [RandomForestEstimator]
    try:
        from inference_simulator.estimator.ridge import RidgeRegressionEstimator
        estimator_classes.append(RidgeRegressionEstimator)
    except ImportError:
        pass
    try:
        from inference_simulator.estimator.knn import KNNEstimator
        estimator_classes.append(KNNEstimator)
    except ImportError:
        pass

    click.echo(f"\nComparing {len(estimator_classes)} estimator types...")
    comparison = _compare(all_measurements, None, estimator_classes)

    # Print comparison table
    click.echo(f"\n{'Estimator':<30} {'Time R²':>10} {'Time MAE':>12} {'Time RMSE':>12}")
    click.echo("-" * 66)
    for entry in comparison:
        name = entry.get("estimator", "unknown")
        if "error" in entry:
            click.echo(f"{name:<30} ERROR: {entry['error']}")
        else:
            r2 = entry.get("time_r2", float("nan"))
            mae = entry.get("time_mae", float("nan"))
            rmse = entry.get("time_rmse", float("nan"))
            click.echo(f"{name:<30} {r2:>10.4f} {mae:>12.6f} {rmse:>12.6f}")

    # Pick best
    try:
        best = pick_best_estimator(comparison, "time_r2")
        click.echo(f"\nBest estimator: {best}")
    except ValueError as e:
        click.echo(f"\nCould not pick best: {e}")

    # Per-category breakdown
    if per_category:
        from inference_simulator.estimator.model_comparison import (
            compare_estimators_by_category,
        )

        click.echo(f"\n--- Per-Category Breakdown ---")
        cat_results = compare_estimators_by_category(
            all_measurements, None, estimator_classes,
            include_per_operator=True,
        )

        # Collect all estimator names across categories
        all_est_names = set()
        for cat_metrics in cat_results.values():
            all_est_names.update(cat_metrics.keys())
        est_names = sorted(all_est_names)

        # Print header
        header = f"{'Category':<25}"
        for est_name in est_names:
            short = est_name.replace("Estimator", "").replace("Regression", "")
            header += f" | {short:>12} R²"
        click.echo(f"\n{header}")
        click.echo("-" * len(header))

        for cat_name in sorted(cat_results.keys()):
            row = f"{cat_name:<25}"
            for est_name in est_names:
                metrics = cat_results[cat_name].get(est_name, {})
                if "error" in metrics:
                    row += f" | {'ERROR':>15}"
                else:
                    r2 = metrics.get("time_r2", float("nan"))
                    row += f" | {r2:>15.4f}"
            click.echo(row)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    click.echo(f"\nComparison complete. Output dir: {output_dir}")


@cli.command("generate-lut")
@click.option("--profiling-dir", required=True, type=click.Path(exists=True),
              help="Directory containing profiling CSV files")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Output directory for LUT .npz files")
@click.option("--model", required=True, help="Model ID (e.g., Qwen/Qwen3-8B)")
@click.option("--hardware", required=True, help="Hardware key (e.g., a100_80gb)")
def generate_lut(profiling_dir: str, output_dir: str, model: str, hardware: str):
    """Generate LUT bundle from profiling CSVs (dedicated LUT generation)."""
    from inference_simulator.estimator.lut_generator import LUTGenerator

    model_spec = _load_model_spec(model)
    hw_spec = HardwareSpec.from_registry(hardware)

    click.echo(f"Generating LUTs from {profiling_dir}...")
    generator = LUTGenerator()
    bundle = generator.generate_full_bundle(
        Path(profiling_dir), Path(output_dir), model_spec, hw_spec
    )

    click.echo(f"\nLUT bundle generated:")
    click.echo(f"  Base dir: {bundle.base_dir}")
    click.echo(f"  Token ops: {bundle.gpu_token_ops_lut}")
    click.echo(f"  Attention prefill: {bundle.gpu_attention_prefill_lut}")
    click.echo(f"  Attention decode: {bundle.gpu_attention_decode_lut}")
    if bundle.tool_distributions:
        click.echo(f"  Tool distributions: {bundle.tool_distributions}")
    click.echo(f"  Estimator: {bundle.metadata.get('estimator_class', 'unknown')}")
    n_measurements = bundle.metadata.get('num_measurements', 0)
    click.echo(f"  Measurements used: {n_measurements}")


@cli.command("run-agent")
@click.option("--dataset", required=True, help="Dataset name")
@click.option(
    "--model-url",
    default="http://localhost:8001/v1",
    help="vLLM API URL",
)
@click.option("--limit", default=None, type=int, help="Max queries")
@click.option(
    "--output-dir",
    default="data/agent_runs",
    type=click.Path(),
    help="Output directory for agent run results",
)
def run_agent(dataset: str, model_url: str, limit: Optional[int], output_dir: str):
    """Run agent workloads against a model and capture token/tool metrics."""
    from dataset_generator.pipeline.agent_runner import AgentRunner

    click.echo(f"Running agent workloads from dataset '{dataset}'...")
    runner = AgentRunner(model_url=model_url, dataset_name=dataset, limit=limit)
    results = runner.run()
    output_path = runner.save_results(results, Path(output_dir))
    click.echo(f"Saved {len(results)} results to {output_path}")


@cli.command("show-checklist")
@click.option(
    "--profiling-dir",
    default="data/profiles",
    type=click.Path(),
    help="Directory containing profiling CSV files",
)
def show_checklist(profiling_dir: str):
    """Show operator profiling checklist with completion status."""
    from dataset_generator.pipeline.checklist import (
        get_checklist_status,
        print_checklist,
    )

    status = get_checklist_status(Path(profiling_dir))
    click.echo(print_checklist(status))


@cli.command("show-matrix")
@click.option(
    "--profiling-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing profiling CSV files",
)
@click.option(
    "--datasets",
    default="wildchat,openthoughts,hotpotqa,agentdata,swebench",
    help="Comma-separated dataset names",
)
def show_matrix(profiling_dir: str, datasets: str):
    """Show dataset x operator tracking matrix."""
    from dataset_generator.pipeline.tracking import TrackingMatrix

    dataset_list = [d.strip() for d in datasets.split(",")]
    matrix = TrackingMatrix(datasets=dataset_list, profiling_dir=Path(profiling_dir))
    matrix.scan()
    click.echo(matrix.to_markdown())
    click.echo(f"\nCompletion: {matrix.completion_pct():.1f}%")


@cli.command("analyze-distributions")
@click.option(
    "--agent-run-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing agent run JSONL files",
)
@click.option(
    "--output-csv",
    default=None,
    type=click.Path(),
    help="Optional CSV output path for distribution stats",
)
def analyze_distributions(agent_run_dir: str, output_csv: Optional[str]):
    """Analyze token, step, and tool distributions from agent runs."""
    from dataset_generator.pipeline.agent_runner import AgentRunner
    from dataset_generator.pipeline.distributions import (
        compute_distributions,
        distributions_to_csv,
    )

    run_dir = Path(agent_run_dir)
    all_results = []
    for jsonl_file in run_dir.glob("*.jsonl"):
        results = AgentRunner.load_results(jsonl_file)
        all_results.extend(results)
        click.echo(f"Loaded {len(results)} results from {jsonl_file.name}")

    if not all_results:
        click.echo("No agent run results found.")
        return

    dist = compute_distributions(all_results)
    click.echo(f"\nWorkload type: {dist.workload_type}")
    click.echo(f"Samples: {dist.num_samples}")
    click.echo(f"Prefill tokens: mean={dist.prefill_tokens.mean:.1f}, "
               f"p50={dist.prefill_tokens.p50:.1f}, p99={dist.prefill_tokens.p99:.1f}")
    click.echo(f"Decode tokens:  mean={dist.decode_tokens.mean:.1f}, "
               f"p50={dist.decode_tokens.p50:.1f}, p99={dist.decode_tokens.p99:.1f}")
    click.echo(f"Steps:          mean={dist.num_steps.mean:.1f}, "
               f"p50={dist.num_steps.p50:.1f}, p99={dist.num_steps.p99:.1f}")
    click.echo(f"Tool calls/q:   mean={dist.tool_calls_per_query.mean:.1f}, "
               f"p50={dist.tool_calls_per_query.p50:.1f}, "
               f"p99={dist.tool_calls_per_query.p99:.1f}")
    click.echo(f"Latency (s):    mean={dist.latency_s.mean:.2f}, "
               f"p50={dist.latency_s.p50:.2f}, p99={dist.latency_s.p99:.2f}")

    if dist.tool_type_counts:
        click.echo(f"\nTool type counts:")
        for tool_name, count in sorted(
            dist.tool_type_counts.items(), key=lambda x: -x[1]
        ):
            click.echo(f"  {tool_name}: {count}")

    if output_csv:
        distributions_to_csv(dist, Path(output_csv))
        click.echo(f"\nDistribution stats written to {output_csv}")


@cli.command("characterize-workloads")
@click.option(
    "--workload-type",
    type=click.Choice(["chat", "reasoning", "agentic", "rag", "coding", "all"]),
    default="all",
    help="Workload type to characterize",
)
@click.option("--limit", type=int, default=None, help="Max samples per dataset")
@click.option(
    "--output-dir",
    default="data/workload_profiles",
    type=click.Path(),
    help="Output directory for profile JSON files",
)
def characterize_workloads(workload_type: str, limit: Optional[int], output_dir: str):
    """Extract statistical distributions from real datasets."""
    from dataset_generator.characterization.registry import (
        characterize_all,
        characterize_workload,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Map workload type names to characterizer registry keys (dataset names)
    _workload_to_dataset = {
        "chat": "wildchat",
        "reasoning": "openthoughts",
        "agentic": "agentdata",
        "rag": "hotpotqa",
        "coding": "swebench",
    }

    if workload_type == "all":
        profiles = characterize_all(limit=limit, output_dir=str(output_path))
        click.echo(f"Characterized {len(profiles)} workload types:")
        for name, profile in profiles.items():
            click.echo(f"  {name}: {profile.n_samples} samples")
    else:
        dataset_name = _workload_to_dataset.get(workload_type, workload_type)
        profile = characterize_workload(dataset_name, limit=limit)
        profile.save(output_path / f"{dataset_name}_profile.json")
        click.echo(f"Characterized workload:")
        click.echo(f"  {dataset_name}: {profile.n_samples} samples")

    click.echo(f"Profiles saved to {output_path}")


@cli.command("run-pipeline")
@click.option("--model", required=True, help="Model ID (e.g., Qwen/Qwen3-8B)")
@click.option("--hardware", required=True, help="Hardware key (e.g., a100_80gb)")
@click.option("--profiling-dir", required=True, type=click.Path(exists=True),
              help="Directory containing profiling CSV files")
@click.option("--output-dir", default="data/pipeline_output", type=click.Path(),
              help="Output directory for pipeline results")
@click.option("--precision", default="fp16", help="Precision (fp16, fp8, bf16)")
@click.option("--workload-type", type=click.Choice(["chat", "reasoning", "agentic", "rag", "coding"]),
              default="chat", help="Workload type")
@click.option("--max-ttft", type=float, default=None, help="Max TTFT SLA (seconds)")
@click.option("--min-throughput-tps", type=float, default=None, help="Min throughput SLA (tokens/s)")
@click.option("--accuracy-score", type=float, default=1.0, help="Model accuracy score")
@click.option("--price-per-gpu-hour", type=float, default=0.0, help="GPU cost per hour (USD)")
@click.option("--characterize-workload/--no-characterize-workload", default=False,
              help="Run Stage 2 workload characterization from real datasets")
@click.option("--profile-limit", type=int, default=None,
              help="Max samples for workload characterization")
@click.option("--workload-profile", type=click.Path(exists=True), default=None,
              help="Path to pre-built WorkloadProfile JSON (skips Stage 2)")
@click.option("--duration-s", type=float, default=10.0,
              help="Simulation duration in seconds")
def run_pipeline(
    model: str,
    hardware: str,
    profiling_dir: str,
    output_dir: str,
    precision: str,
    workload_type: str,
    max_ttft: Optional[float],
    min_throughput_tps: Optional[float],
    accuracy_score: float,
    price_per_gpu_hour: float,
    characterize_workload: bool,
    profile_limit: Optional[int],
    workload_profile: Optional[str],
    duration_s: float,
):
    """Run the full pipeline: train estimators -> simulate -> search."""
    from dataset_generator.pipeline.orchestrator import PipelineConfig, PipelineOrchestrator

    config = PipelineConfig(
        model_id=model,
        hardware_key=hardware,
        precision=precision,
        profiling_dir=Path(profiling_dir),
        lut_dir=Path(output_dir) / "luts",
        output_dir=Path(output_dir),
        workload_type=workload_type,
        max_ttft=max_ttft,
        min_throughput_tps=min_throughput_tps,
        accuracy_score=accuracy_score,
        price_per_gpu_hour_usd=price_per_gpu_hour,
        duration_s=duration_s,
        characterize_workload=characterize_workload,
        workload_profile_limit=profile_limit,
        workload_profile_path=Path(workload_profile) if workload_profile else None,
    )

    click.echo(f"Running full pipeline for {model} on {hardware}...")
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run_all(config)

    # Print results summary
    bundle = results["lut_bundle"]
    sim = results["simulation_metrics"]
    search = results["search_result"]

    click.echo(f"\n=== Pipeline Results ===")
    click.echo(f"\nPipeline #1b (Estimator Training):")
    click.echo(f"  Estimator: {bundle.metadata.get('estimator_class', 'unknown')}")
    click.echo(f"  Measurements: {bundle.metadata.get('num_measurements', 0)}")

    click.echo(f"\nPipeline #2 (Simulation):")
    click.echo(f"  Requests: {sim.total_requests}")
    click.echo(f"  Throughput: {sim.throughput_tps:.1f} tok/s")
    click.echo(f"  TTFT P50: {sim.ttft_p50:.4f}s")
    click.echo(f"  E2E P50: {sim.e2e_p50:.4f}s")

    click.echo(f"\nPipeline #3 (Search):")
    click.echo(f"  Configurations: {len(search.all_results)}")
    click.echo(f"  Pareto frontier: {len(search.pareto_frontier)}")
    if search.pareto_frontier:
        best = search.pareto_frontier[0]
        click.echo(f"  Best config: max_qps={best.max_qps:.1f}")


def main():
    """Entry point for the dataset-generator CLI."""
    cli()


if __name__ == "__main__":
    main()
