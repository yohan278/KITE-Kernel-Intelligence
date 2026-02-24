#!/usr/bin/env python3
"""CLI command for generating SFT trajectories using Gemini as teacher.

Usage:
    python -m src.cli.generate_trajectories --source-dataset generalthought --limit 100
    python -m src.cli.generate_trajectories --config configs/trajectory_generation.yaml --source-dataset generalthought --limit 20000
    python -m src.cli.generate_trajectories --help
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    try:
        import yaml
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        print("Error: PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config from {yaml_path}: {e}")
        sys.exit(1)


def parse_tools(tools_str: str) -> list:
    """Parse comma-separated tools string.

    Args:
        tools_str: Comma-separated tool names

    Returns:
        List of tool names
    """
    return [t.strip() for t in tools_str.split(",") if t.strip()]


def parse_domains(domains_str: str) -> list:
    """Parse comma-separated domains string.

    Args:
        domains_str: Comma-separated domain names

    Returns:
        List of domain names
    """
    return [d.strip() for d in domains_str.split(",") if d.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT trajectories using Gemini 3.0 Flash as teacher model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )

    # Source dataset
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="mixed",
        choices=["generalthought", "agentdata", "mixed"],
        help="Source dataset for tasks",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by task category",
    )

    # GeneralThought-specific
    parser.add_argument(
        "--min-verifier-score",
        type=float,
        default=0.5,
        help="Minimum verifier score for GeneralThought (0-1)",
    )

    # AgentData-specific
    parser.add_argument(
        "--domains",
        type=str,
        default="codeact,agenttuning_db,agenttuning_webshop",
        help="Comma-separated AgentData domains to use",
    )

    # Teacher model
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model to use as teacher",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per trajectory",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for teacher model",
    )

    # Generation settings
    parser.add_argument(
        "--traces-per-query",
        type=int,
        default=2,
        help="Number of traces to generate per query",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per trace before giving up",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for generation",
    )

    # Tools
    parser.add_argument(
        "--tools",
        type=str,
        default="calculator,think,code_interpreter,web_search",
        help="Comma-separated list of available tools",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/trajectories",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="arrow",
        choices=["arrow", "parquet", "jsonl"],
        help="Output format",
    )

    # HuggingFace Hub
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (e.g., 'myorg/my-dataset')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Hub dataset private",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or use HF_TOKEN env var)",
    )

    # Execution
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        default=True,
        help="Include failed trajectories in output",
    )
    parser.add_argument(
        "--no-include-failures",
        action="store_false",
        dest="include_failures",
        help="Exclude failed trajectories from output",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable energy telemetry collection",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N samples",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--code-isolation",
        type=str,
        default="auto",
        help="Code isolation mode: auto, bubblewrap, or null",
    )

    args = parser.parse_args()

    # Initialize code isolation (will be overridden by config if provided)
    code_isolation = args.code_isolation

    # Load config from YAML if provided
    if args.config:
        config = load_config_from_yaml(args.config)
        # Apply config values (args override config)
        teacher_config = config.get("teacher", {})
        source_config = config.get("source", {})
        output_config = config.get("output", {})
        exec_config = config.get("execution", {})
        gen_config = config.get("generation", {})

        if args.teacher_model == "gemini-3-flash-preview":  # default
            args.teacher_model = teacher_config.get("model", args.teacher_model)
        if args.max_turns == 10:  # default
            args.max_turns = teacher_config.get("max_turns", args.max_turns)
        if args.temperature == 0.7:  # default
            args.temperature = teacher_config.get("temperature", args.temperature)
        if args.tools == "calculator,think,code_interpreter,web_search":  # default
            tools_list = config.get("tools", [])
            if tools_list:
                args.tools = ",".join(tools_list)

        # Generation config
        if args.traces_per_query == 2:  # default
            args.traces_per_query = gen_config.get("traces_per_query", args.traces_per_query)
        if args.max_attempts == 3:  # default
            args.max_attempts = gen_config.get("max_attempts", args.max_attempts)
        if args.max_tokens == 8192:  # default
            args.max_tokens = gen_config.get("max_tokens", args.max_tokens)

        # Source config
        if args.source_dataset == "mixed":  # default
            args.source_dataset = source_config.get("dataset", args.source_dataset)
        gt_config = source_config.get("generalthought", {})
        if args.min_verifier_score == 0.5:  # default
            args.min_verifier_score = gt_config.get("min_verifier_score", args.min_verifier_score)
        # Get limit from dataset-specific config if not set via CLI
        if args.limit is None:
            args.limit = gt_config.get("limit") or source_config.get("limit")
        ad_config = source_config.get("agentdata", {})
        if args.domains == "codeact,agenttuning_db,agenttuning_webshop":  # default
            domains_list = ad_config.get("domains", [])
            if domains_list:
                args.domains = ",".join(domains_list)

        # Output config
        if args.output_format == "arrow":  # default
            args.output_format = output_config.get("format", args.output_format)
        args.include_failures = output_config.get("include_failures", args.include_failures)

        # Execution config
        if args.workers == 4:  # default
            args.workers = exec_config.get("workers", args.workers)
        if not args.telemetry:
            args.telemetry = exec_config.get("telemetry", args.telemetry)
        if args.checkpoint_every == 100:  # default
            args.checkpoint_every = exec_config.get("checkpoint_every", args.checkpoint_every)
        # Code isolation setting (defaults to "auto" if not specified)
        code_isolation = exec_config.get("code_isolation", "auto")
        # Convert YAML null to Python None
        if code_isolation is None or code_isolation == "null":
            code_isolation = None

        # Monitoring config (for TrajectoryStats)
        monitoring_config = config.get("monitoring", {})

    # Parse tool and domain lists
    tools = parse_tools(args.tools)
    domains = parse_domains(args.domains)

    print("=" * 60)
    print("Trajectory Generation Configuration")
    print("=" * 60)
    print(f"Source dataset: {args.source_dataset}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Traces per query: {args.traces_per_query}")
    print(f"Max attempts: {args.max_attempts}")
    print(f"Tools ({len(tools)}): {tools}")
    print(f"Output: {args.output_dir} ({args.output_format})")
    if args.push_to_hub:
        print(f"Hub repo: {args.hub_repo}")
    print(f"Workers: {args.workers}")
    print(f"Include failures: {args.include_failures}")
    print(f"Telemetry: {args.telemetry}")
    print(f"Code isolation: {code_isolation}")
    print("=" * 60)

    # Initialize telemetry collector if enabled
    telemetry_collector = None
    if args.telemetry:
        try:
            from ipw.telemetry import EnergyMonitorCollector
            telemetry_collector = EnergyMonitorCollector()
            print("Telemetry collection enabled")
        except ImportError:
            print("Warning: ipw telemetry not available, running without energy monitoring")

    # Load source dataset
    print(f"\nLoading source dataset: {args.source_dataset}...")
    samples = load_source_dataset(
        dataset_name=args.source_dataset,
        split=args.split,
        limit=args.limit,
        category=args.category,
        min_verifier_score=args.min_verifier_score,
        domains=domains,
    )
    print(f"Loaded {len(samples)} samples")

    # Resume from checkpoint if specified
    resumed_dataset = None
    if args.resume:
        from orchestrator.data.trajectory_dataset import TrajectoryDataset
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            try:
                # Load checkpoint dataset
                resumed_dataset = TrajectoryDataset.load(str(checkpoint_path))

                # Get already processed sample IDs (use source_task_id which maps to original task IDs)
                # Convert to strings for consistent comparison (checkpoint stores as strings)
                processed_ids = set(str(s.source_task_id) for s in resumed_dataset if s.source_task_id)
                original_count = len(samples)

                # Filter out already processed samples (samples can be dataclass objects)
                def get_sample_id(s):
                    # Try common ID attribute names
                    for attr in ["task_id", "id", "sample_id"]:
                        val = getattr(s, attr, None)
                        if val is not None:
                            return str(val)  # Convert to string for comparison
                    # Fallback to dict access
                    if hasattr(s, "get"):
                        val = s.get("id") or s.get("task_id")
                        return str(val) if val else str(hash(str(s)))
                    return str(hash(str(s)))

                samples = [s for s in samples if get_sample_id(s) not in processed_ids]

                print(f"\nResuming from checkpoint: {len(processed_ids)} samples already processed")
                print(f"Remaining samples: {len(samples)} (filtered from {original_count})")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting from scratch...")
                resumed_dataset = None

    # Initialize generator
    from orchestrator.data.trajectory_generator import TrajectoryGenerator, TrajectoryStats
    from orchestrator.data.trajectory_dataset import TrajectoryDataset

    # Initialize stats tracking if monitoring is enabled in config
    stats = None
    if args.config:
        monitoring_config = config.get("monitoring", {})
        if monitoring_config.get("enabled", False):
            stats_interval = monitoring_config.get("stats_interval", 100)
            stats_log_file = monitoring_config.get("log_file")
            stats = TrajectoryStats(
                stats_interval=stats_interval,
                log_file=stats_log_file,
            )
            print(f"Stats monitoring enabled (interval={stats_interval})")
            if stats_log_file:
                print(f"Stats log file: {stats_log_file}")

    generator = TrajectoryGenerator(
        teacher_model=args.teacher_model,
        available_tools=tools,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        traces_per_query=args.traces_per_query,
        max_attempts=args.max_attempts,
        telemetry_collector=telemetry_collector,
        include_failures=args.include_failures,
        code_isolation=code_isolation,
        stats=stats,
    )

    # Generate trajectories with progress
    print(f"\nGenerating trajectories...")
    dataset = TrajectoryDataset()

    def progress_callback(completed, total, trajectory):
        # Add trajectory to dataset as it's generated
        if trajectory is not None:
            dataset.add_trajectory(trajectory)

        pct = completed / total * 100
        print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end="", flush=True)

        # Checkpoint (only if we have data)
        if completed > 0 and completed % args.checkpoint_every == 0:
            if len(dataset) > 0:
                checkpoint_path = Path(args.output_dir) / "checkpoint"
                dataset.save(str(checkpoint_path), format=args.output_format)
                print(f"\n  Checkpoint saved at {completed} samples ({len(dataset)} trajectories)")

    trajectories = generator.generate_batch(
        samples=samples,
        max_workers=args.workers,
        progress_callback=progress_callback,
    )
    print()  # newline after progress

    # Note: trajectories already added via progress_callback

    # Merge with resumed dataset if we were resuming
    if resumed_dataset is not None:
        print(f"\nMerging with checkpoint ({len(resumed_dataset)} samples)...")
        # Add all samples from resumed dataset to current dataset
        for sample in resumed_dataset:
            dataset._sft_samples.append(sample)
        print(f"Total after merge: {len(dataset)} samples")

    # Print statistics
    dataset_stats = dataset.get_statistics()
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total trajectories: {dataset_stats['total_samples']}")
    print(f"Successful: {dataset_stats['success_count']} ({dataset_stats['success_rate']:.1%})")
    print(f"Failed: {dataset_stats['failure_count']}")
    print(f"Avg turns: {dataset_stats['avg_turns']:.1f}")
    print(f"Avg energy: {dataset_stats['avg_energy_joules']:.4f} J")
    print(f"Avg latency: {dataset_stats['avg_latency_seconds']:.2f} s")
    print(f"Avg cost: ${dataset_stats['avg_cost_usd']:.6f}")
    print(f"Total cost: ${dataset_stats['total_cost_usd']:.4f}")
    print("=" * 60)

    # Print final tool usage stats if monitoring was enabled
    if stats is not None:
        print("\n" + "=" * 60)
        print("Final Tool Usage Statistics")
        print("=" * 60)
        stats.print_summary()

    # Save locally
    print(f"\nSaving to {args.output_dir}...")
    dataset.save(args.output_dir, format=args.output_format)

    # Push to Hub if requested
    if args.push_to_hub:
        if not args.hub_repo:
            print("Error: --hub-repo required when using --push-to-hub")
            sys.exit(1)

        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}...")
        dataset.push_to_hub(
            repo_id=args.hub_repo,
            private=args.private,
            token=args.hf_token,
        )

    print("\nDone!")


def load_source_dataset(
    dataset_name: str,
    split: str = "train",
    limit: int = None,
    category: str = None,
    min_verifier_score: float = 0.5,
    domains: list = None,
) -> list:
    """Load samples from source dataset.

    Args:
        dataset_name: Dataset to load (generalthought, agentdata, mixed)
        split: Dataset split
        limit: Maximum samples
        category: Filter by category
        min_verifier_score: Minimum verifier score for GeneralThought
        domains: Domains for AgentData

    Returns:
        List of samples
    """
    samples = []

    if dataset_name == "generalthought":
        from orchestrator.data.generalthought_loader import GeneralThoughtDataset

        dataset = GeneralThoughtDataset(
            split=split,
            limit=limit,
            min_verifier_score=min_verifier_score,
        )
        samples = list(dataset)

    elif dataset_name == "agentdata":
        from orchestrator.data.agentdata_loader import AgentDataCollectionDataset

        dataset = AgentDataCollectionDataset(
            domains=domains,
            limit=limit,
        )
        samples = list(dataset)

    elif dataset_name == "mixed":
        from orchestrator.data.generalthought_loader import GeneralThoughtDataset
        from orchestrator.data.agentdata_loader import AgentDataCollectionDataset

        # Load from both datasets
        gt_limit = limit // 2 if limit else None
        ad_limit = limit - gt_limit if limit else None

        try:
            gt_dataset = GeneralThoughtDataset(
                split=split,
                limit=gt_limit,
                min_verifier_score=min_verifier_score,
            )
            samples.extend(list(gt_dataset))
        except Exception as e:
            print(f"Warning: Could not load GeneralThought: {e}")

        try:
            ad_dataset = AgentDataCollectionDataset(
                domains=domains,
                limit=ad_limit,
            )
            samples.extend(list(ad_dataset))
        except Exception as e:
            print(f"Warning: Could not load AgentData: {e}")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Filter by category if specified
    if category:
        samples = [s for s in samples if getattr(s, 'category', None) == category]

    return samples


if __name__ == "__main__":
    main()
