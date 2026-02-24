#!/usr/bin/env python3
"""CLI command for SFT training with ToolOrchestra + ADP tools.

Generates multiple traces per query (successful + unsuccessful) and
trains using supervised fine-tuning.

Usage:
    python src/cli/train_sft.py --model Qwen/Qwen3-1.7B --epochs 3
    python src/cli/train_sft.py --traces-per-query 4 --success-weight 2.0
    python src/cli/train_sft.py --help
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator.training.sft.sft_trainer import SFTTrainer, SFTConfig


def parse_tools(tools_str: str) -> list:
    """Parse comma-separated tools string."""
    return [t.strip() for t in tools_str.split(",") if t.strip()]


def parse_domains(domains_str: str) -> list:
    """Parse comma-separated domains string."""
    return [d.strip() for d in domains_str.split(",") if d.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train orchestrator using SFT with ToolOrchestra + ADP tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )

    # Trace generation
    parser.add_argument(
        "--traces-per-query",
        type=int,
        default=2,
        help="Target number of successful traces per query",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default="gemini:gemini-2.5-flash",
        help="Model for generating trajectories (e.g., gemini:gemini-2.5-flash, openai:gpt-5-mini-2025-08-07)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts to generate one successful trace",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for trace generation",
    )

    # Dataset mode (GeneralThought + Agent Data Collection)
    parser.add_argument(
        "--datasets",
        type=str,
        default="mixed",
        choices=["generalthought", "agentdata", "mixed"],
        help="Dataset mode: 'generalthought', 'agentdata', or 'mixed' for both",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per dataset",
    )
    parser.add_argument(
        "--generalthought-weight",
        type=float,
        default=0.5,
        help="Weight for GeneralThought in mixed mode",
    )
    parser.add_argument(
        "--agentdata-weight",
        type=float,
        default=0.5,
        help="Weight for Agent Data in mixed mode",
    )
    parser.add_argument(
        "--min-verifier-score",
        type=float,
        default=0.5,
        help="Minimum verifier score for GeneralThought quality filtering (0-1)",
    )
    parser.add_argument(
        "--agentdata-domains",
        type=str,
        default=None,
        help="Comma-separated ADP domains (default: all recommended)",
    )

    # Tools
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Comma-separated list of tools (default: all ToolOrchestra + ADP tools)",
    )

    # Caching & workflow control
    parser.add_argument(
        "--trace-cache",
        type=str,
        default="data/sft_traces.jsonl",
        help="Path to trace cache file (JSONL format)",
    )
    parser.add_argument(
        "--trajectory-dataset",
        type=str,
        default=None,
        help="Path to pre-generated trajectory dataset (Arrow format, e.g., data/gpt-oss-trajectories/checkpoint/dataset)",
    )
    parser.add_argument(
        "--regenerate-traces",
        action="store_true",
        default=False,
        help="Regenerate traces even if cache exists",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate traces, don't train (useful for data preparation)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train using existing traces (requires --trace-cache or --trajectory-dataset)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sft",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/sft",
        help="Directory for logs",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="orchestrator-sft",
        help="W&B project name",
    )

    # Memory optimization
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    # Early stopping with eval tasks
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on eval task performance",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience (number of evals with no improvement)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.01,
        help="Minimum improvement threshold (e.g., 0.01 = 1%% success rate)",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        default=None,
        help="Path to evaluation tasks (JSONL or Arrow dataset)",
    )
    parser.add_argument(
        "--max-eval-tasks",
        type=int,
        default=100,
        help="Maximum number of eval tasks to run",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate every N epochs",
    )

    args = parser.parse_args()

    # Validate trajectory dataset + train-only workflow
    if args.train_only:
        if args.trajectory_dataset:
            # Convert trajectory dataset to trace cache
            print(f"Loading trajectory dataset: {args.trajectory_dataset}")
            from orchestrator.data import PreGeneratedTrajectoryDataset
            import json
            
            dataset = PreGeneratedTrajectoryDataset(
                dataset_path=args.trajectory_dataset,
                success_only=True,
                limit=args.limit,
            )
            
            # Convert to trace cache format
            trace_cache_path = Path(args.trace_cache)
            trace_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Converting {len(dataset)} trajectories to trace cache...")
            with open(trace_cache_path, "w") as f:
                for sample in dataset:
                    trace = {
                        "trajectory_id": sample.sample_id,
                        "conversations": sample.conversations,
                        "tool_calls": sample.tool_calls,
                        "ground_truth": sample.ground_truth,
                        "final_answer": sample.final_answer,
                        "success": sample.success,
                        "total_energy_joules": sample.total_energy_joules,
                        "total_latency_seconds": sample.total_latency_seconds,
                        "total_cost_usd": sample.total_cost_usd,
                        "total_tokens": sample.total_tokens,
                        "num_turns": sample.num_turns,
                        "source_dataset": sample.source_dataset,
                        "source_task_id": sample.source_task_id,
                        "category": sample.category,
                        "teacher_model": sample.teacher_model,
                        "tools_used": sample.tools_used,
                    }
                    f.write(json.dumps(trace) + "\n")
            
            print(f"✅ Saved traces to {trace_cache_path}")
            args.trace_cache = str(trace_cache_path)
        elif not Path(args.trace_cache).exists():
            print(f"Error: --train-only requires existing traces")
            print(f"  Provide --trace-cache (JSONL) or --trajectory-dataset (Arrow)")
            sys.exit(1)
    
    # Create config
    config = SFTConfig(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        traces_per_query=args.traces_per_query,
        generation_model=args.generation_model,
        max_attempts_per_trace=args.max_attempts,
        generation_temperature=args.generation_temperature,
        dataset_mode=args.datasets,
        dataset_limit=args.limit,
        generalthought_weight=args.generalthought_weight,
        agentdata_weight=args.agentdata_weight,
        min_verifier_score=args.min_verifier_score,
        agentdata_domains=parse_domains(args.agentdata_domains) if args.agentdata_domains else None,
        available_tools=parse_tools(args.tools) if args.tools else None,
        trace_cache_path=args.trace_cache,
        regenerate_traces=args.regenerate_traces,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every,
        log_dir=args.log_dir,
        log_every_n_steps=args.log_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Early stopping with eval tasks
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        eval_tasks_path=args.eval_tasks,
        max_eval_tasks=args.max_eval_tasks,
        eval_every_n_epochs=args.eval_every,
    )

    # Print configuration
    print("=" * 70)
    print("SFT Training Configuration")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Max seq length: {config.max_seq_length}")
    print()
    print("Dataset (GeneralThought + Agent Data Collection):")
    print(f"  Mode: {config.dataset_mode}")
    if config.dataset_limit:
        print(f"  Limit: {config.dataset_limit} samples")
    if config.dataset_mode == "mixed":
        print(f"  GeneralThought weight: {config.generalthought_weight:.0%}")
        print(f"  AgentData weight: {config.agentdata_weight:.0%}")
    if config.dataset_mode in ("generalthought", "mixed"):
        print(f"  Min verifier score: {config.min_verifier_score}")
    if config.dataset_mode in ("agentdata", "mixed"):
        print(f"  ADP domains: {', '.join(config.agentdata_domains[:3])}...")
    print()
    print("Trace generation (successful trajectories only):")
    print(f"  Target traces per query: {config.traces_per_query}")
    print(f"  Generation model: {config.generation_model}")
    print(f"  Max attempts per trace: {config.max_attempts_per_trace}")
    print(f"  Temperature: {config.generation_temperature}")
    print(f"  Cache: {config.trace_cache_path}")
    print(f"  Regenerate: {config.regenerate_traces}")
    print()
    print("Training:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup ratio: {config.warmup_ratio}")
    print()
    print(f"Tools ({len(config.available_tools)}):")
    for tool in config.available_tools[:5]:
        print(f"  - {tool}")
    if len(config.available_tools) > 5:
        print(f"  ... and {len(config.available_tools) - 5} more")
    print()
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Logs: {config.log_dir}")
    if args.generate_only and not args.train_only:
        print(f"Mode: GENERATE ONLY (no training)")
    elif args.train_only and not args.generate_only:
        print(f"Mode: TRAIN ONLY (using existing traces)")
    else:
        print(f"Mode: TRAIN AND GENERATE (default)")
    print("=" * 70)
    print()

    # Handle --generate-only: just generate traces and exit
    if args.generate_only and not args.train_only:
        from orchestrator.data import generate_sft_dataset, GeneralThoughtDataset, AgentDataCollectionDataset, create_mixed_dataset
        
        print("Generating traces only...")
        
        # Load datasets based on mode
        generalthought_dataset = None
        agentdata_dataset = None
        
        if config.dataset_mode in ("generalthought", "mixed"):
            print("  Loading GeneralThought...")
            try:
                generalthought_dataset = GeneralThoughtDataset(
                    split="train",
                    limit=config.dataset_limit,
                    min_verifier_score=config.min_verifier_score,
                )
                print(f"    Loaded {len(generalthought_dataset)} samples")
            except Exception as e:
                print(f"    Warning: {e}")
        
        if config.dataset_mode in ("agentdata", "mixed"):
            print("  Loading Agent Data Collection...")
            try:
                agentdata_dataset = AgentDataCollectionDataset(
                    domains=config.agentdata_domains,
                    limit=config.dataset_limit,
                )
                print(f"    Loaded {len(agentdata_dataset)} samples")
            except Exception as e:
                print(f"    Warning: {e}")
        
        # Create mixed dataset if needed
        if config.dataset_mode == "mixed" and generalthought_dataset and agentdata_dataset:
            dataset = create_mixed_dataset(
                generalthought_dataset=generalthought_dataset,
                agentdata_dataset=agentdata_dataset,
                generalthought_weight=config.generalthought_weight,
                agentdata_weight=config.agentdata_weight,
            )
            samples = list(dataset)
        elif generalthought_dataset:
            samples = list(generalthought_dataset)
        elif agentdata_dataset:
            samples = list(agentdata_dataset)
        else:
            print("Error: No datasets loaded")
            sys.exit(1)
        
        # Generate traces
        generate_sft_dataset(
            toolscale_samples=samples,
            adp_samples=None,
            available_tools=config.available_tools,
            generation_model=config.generation_model,
            traces_per_query=config.traces_per_query,
            max_attempts_per_trace=config.max_attempts_per_trace,
            temperature=config.generation_temperature,
            output_path=config.trace_cache_path,
        )
        
        print(f"\n✅ Traces generated: {config.trace_cache_path}")
        sys.exit(0)

    # Handle --train-only: require existing traces
    if args.train_only and not args.generate_only:
        if not Path(config.trace_cache_path).exists():
            print(f"Error: --train-only requires existing traces at {config.trace_cache_path}")
            print("Generate traces first with: --generate-only")
            sys.exit(1)
        # Force skip regeneration
        config.regenerate_traces = False

    # Create trainer
    try:
        trainer = SFTTrainer(config)
    except Exception as e:
        print(f"Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ SFT Training complete!")
    print(f"Final checkpoint: {config.checkpoint_dir}")
    print(f"Logs: {config.log_dir}")


if __name__ == "__main__":
    main()
