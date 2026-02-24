#!/usr/bin/env python3
"""CLI command for training orchestrator with GRPO.

Usage:
    python src/cli/train.py --model Qwen/Qwen3-1.7B --epochs 3
    python src/cli/train.py --config training/configs/training.yaml
    python src/cli/train.py --help
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator.training.rl.trainer import GRPOTrainer, GRPOConfig


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


def main():
    parser = argparse.ArgumentParser(
        description="Train orchestrator using GRPO (Group Relative Policy Optimization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=24000,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=8768,
        help="Maximum response length",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of samples per prompt (GRPO group size)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.0001,
        help="KL divergence coefficient",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="nvidia/ToolScale",
        help="Dataset name (for single-dataset mode)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )

    # Multi-dataset mode
    parser.add_argument(
        "--datasets",
        type=str,
        default="toolscale",
        choices=["toolscale", "generalthought", "mixed"],
        help="Dataset mode: 'toolscale', 'generalthought', or 'mixed' for both",
    )
    parser.add_argument(
        "--toolscale-weight",
        type=float,
        default=0.5,
        help="Weight for ToolScale in mixed dataset mode",
    )
    parser.add_argument(
        "--generalthought-weight",
        type=float,
        default=0.5,
        help="Weight for GeneralThought in mixed dataset mode",
    )
    parser.add_argument(
        "--min-verifier-score",
        type=float,
        default=0.5,
        help="Minimum verifier score for GeneralThought samples (0-1)",
    )

    # Environment
    parser.add_argument(
        "--tools",
        type=str,
        default="calculator,ollama:llama3.2:1b,ollama:qwen2.5:1.5b",
        help="Comma-separated list of available tools",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per episode",
    )

    # Cache
    parser.add_argument(
        "--telemetry-cache",
        type=str,
        default="data/telemetry_cache.db",
        help="Path to telemetry cache database",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=3,
        help="Keep last N checkpoints",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
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
        default="orchestrator",
        help="W&B project name",
    )

    # Online data collection
    parser.add_argument(
        "--online",
        action="store_true",
        default=False,
        help="Enable online data collection (real execution, measures actual energy)",
    )
    parser.add_argument(
        "--offline",
        dest="online",
        action="store_false",
        help="Use cached telemetry (faster training, default)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * accumulation)",
    )

    # Memory optimization
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (saves memory, ~30%% slower)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--8bit-ref",
        action="store_true",
        default=True,
        help="Load reference model in 8-bit (saves ~50%% memory)",
    )
    parser.add_argument(
        "--no-8bit-ref",
        dest="8bit_ref",
        action="store_false",
        help="Disable 8-bit reference model",
    )
    parser.add_argument(
        "--8bit-optimizer",
        action="store_true",
        default=False,
        help="Use 8-bit AdamW optimizer (saves ~50%% optimizer memory)",
    )

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)

        # Extract relevant sections
        model_config = yaml_config.get("model", {})
        training_config = yaml_config.get("training", {})
        dataset_config = yaml_config.get("dataset", {})
        env_config = yaml_config.get("environment", {})
        cache_config = yaml_config.get("cache", {})
        checkpoint_config = yaml_config.get("checkpointing", {})
        logging_config = yaml_config.get("logging", {})

        # Override args with YAML values
        config = GRPOConfig(
            model_name=model_config.get("name", args.model),
            max_prompt_length=model_config.get("max_tokens", args.max_prompt_length),
            max_response_length=model_config.get("max_tokens", args.max_response_length),
            num_epochs=training_config.get("num_epochs", args.epochs),
            batch_size=training_config.get("batch_size", args.batch_size),
            learning_rate=training_config.get("learning_rate", args.lr),
            num_samples_per_prompt=env_config.get("rollout_agents", args.group_size),
            temperature=model_config.get("temperature", args.temperature),
            kl_coef=training_config.get("kl_coef", args.kl_coef),
            dataset_name=dataset_config.get("name", args.dataset),
            dataset_split=dataset_config.get("split", args.split),
            dataset_limit=dataset_config.get("limit", args.limit),
            available_tools=env_config.get("tools", parse_tools(args.tools)),
            max_turns=env_config.get("max_turns", args.max_turns),
            telemetry_cache_path=cache_config.get("telemetry_cache_path", args.telemetry_cache),
            checkpoint_dir=checkpoint_config.get("save_dir", args.checkpoint_dir),
            save_every_n_epochs=checkpoint_config.get("save_every_n_epochs", args.save_every),
            keep_last_n=checkpoint_config.get("keep_last_n", args.keep_last),
            log_dir=logging_config.get("log_dir", args.log_dir),
            log_every_n_steps=logging_config.get("log_every_n_steps", args.log_every),
            use_wandb=logging_config.get("use_wandb", args.wandb),
            wandb_project=logging_config.get("wandb_project", args.wandb_project),
        )
    else:
        # Determine if using mixed dataset mode
        use_mixed = args.datasets == "mixed"

        # Use command-line args
        config = GRPOConfig(
            model_name=args.model,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_samples_per_prompt=args.group_size,
            temperature=args.temperature,
            kl_coef=args.kl_coef,
            dataset_name=args.dataset,
            dataset_split=args.split,
            dataset_limit=args.limit,
            use_mixed_dataset=use_mixed,
            toolscale_weight=args.toolscale_weight,
            generalthought_weight=args.generalthought_weight,
            min_verifier_score=args.min_verifier_score,
            available_tools=parse_tools(args.tools),
            max_turns=args.max_turns,
            telemetry_cache_path=args.telemetry_cache,
            checkpoint_dir=args.checkpoint_dir,
            save_every_n_epochs=args.save_every,
            keep_last_n=args.keep_last,
            log_dir=args.log_dir,
            log_every_n_steps=args.log_every,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            use_online_collection=args.online,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_8bit_ref=getattr(args, '8bit_ref', True),
            use_8bit_optimizer=getattr(args, '8bit_optimizer', False),
        )

    # Print configuration
    print("=" * 70)
    print("GRPO Training Configuration")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    if config.use_mixed_dataset:
        print(f"Dataset mode: Mixed (ToolScale + GeneralThought)")
        print(f"  ToolScale weight: {config.toolscale_weight:.0%}")
        print(f"  GeneralThought weight: {config.generalthought_weight:.0%}")
        print(f"  Min verifier score: {config.min_verifier_score}")
    else:
        print(f"Dataset: {config.dataset_name} ({config.dataset_split})")
    if config.dataset_limit:
        print(f"Limit: {config.dataset_limit} samples")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Group size: {config.num_samples_per_prompt}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Temperature: {config.temperature}")
    print(f"KL coef: {config.kl_coef}")
    print(f"Tools: {len(config.available_tools)} available")
    for tool in config.available_tools:
        print(f"  - {tool}")
    print(f"Telemetry cache: {config.telemetry_cache_path}")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Logs: {config.log_dir}")
    print(f"Data collection: {'ONLINE (real execution)' if config.use_online_collection else 'Cached (fast training)'}")
    if config.use_online_collection:
        print("  ⚠️  Online mode: Each step executes real MCP tools for accurate energy measurement")
    print(f"Memory optimizations:")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps} steps")
    print(f"  8-bit reference model: {config.use_8bit_ref}")
    print(f"  8-bit optimizer: {config.use_8bit_optimizer}")
    print("=" * 70)
    print()

    # Create trainer
    try:
        trainer = GRPOTrainer(config)
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
        print(f"Partial checkpoint saved to: {config.checkpoint_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Training complete!")
    print(f"Final checkpoint: {config.checkpoint_dir}")
    print(f"Logs: {config.log_dir}")


if __name__ == "__main__":
    main()
