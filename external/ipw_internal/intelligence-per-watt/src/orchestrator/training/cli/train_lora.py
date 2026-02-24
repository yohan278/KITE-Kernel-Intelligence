#!/usr/bin/env python3
"""CLI command for LoRA SFT training on trajectory datasets.

Trains models using LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.
Supports 4-bit quantization (QLoRA) and wandb logging with validation.

Usage:
    python src/cli/train_lora.py --model Qwen/Qwen3-4B --epochs 3
    python src/cli/train_lora.py --model Qwen/Qwen3-8B --lora-rank 64 --wandb
    python src/cli/train_lora.py --help
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Train orchestrator using LoRA SFT on trajectory datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
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
        default=1,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (higher for LoRA, typically 1e-4 to 5e-4)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )

    # LoRA configuration
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (r parameter). Higher = more capacity but more params",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (scaling factor). Typically 2x lora-rank",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA",
    )

    # Quantization
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--no-4bit",
        dest="use_4bit",
        action="store_false",
        help="Disable 4-bit quantization",
    )

    # Data
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/gpt-oss-trajectories/success_only/trajectories.jsonl",
        help="Path to training JSONL file with conversations",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of training samples (for quick testing)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/lora",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps",
    )

    # Logging
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Run evaluation every N steps",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="orchestrator-lora",
        help="W&B project name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )

    args = parser.parse_args()

    # Parse target modules
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    # Print configuration
    print("=" * 70)
    print("LoRA SFT Training Configuration")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Max seq length: {args.max_seq_length}")
    print()
    print("LoRA Configuration:")
    print(f"  Rank (r): {args.lora_rank}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"  Target modules: {', '.join(target_modules[:4])}...")
    print(f"  4-bit quantization: {args.use_4bit}")
    print()
    print("Training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print()
    print("Data:")
    print(f"  Train file: {args.train_file}")
    print(f"  Val split: {args.val_split:.0%}")
    if args.limit:
        print(f"  Limit: {args.limit} samples")
    print()
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Logging: wandb={args.wandb}, log_every={args.log_every}, eval_every={args.eval_steps}")
    print("=" * 70)
    print()

    # Import dependencies
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Install with: pip install torch transformers datasets peft trl bitsandbytes")
        sys.exit(1)

    # Setup wandb
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (local file or HuggingFace Hub)
    train_path = Path(args.train_file)
    if train_path.exists():
        print(f"Loading dataset from {args.train_file}...")
        dataset = load_dataset("json", data_files=args.train_file)["train"]
    else:
        # Try loading as HuggingFace Hub dataset
        print(f"Loading dataset from HuggingFace Hub: {args.train_file}...")
        dataset = load_dataset(args.train_file, split="train")
    print(f"Loaded {len(dataset)} samples")

    # Apply limit if specified
    if args.limit and args.limit < len(dataset):
        print(f"Limiting to {args.limit} samples")
        dataset = dataset.select(range(args.limit))

    # Format conversations to text
    def format_to_text(example):
        """Convert conversations to text using chat template."""
        messages = example.get("conversations") or example.get("messages", [])
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Fallback formatting if chat template fails
            parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
                elif role == "tool":
                    parts.append(f"<|im_start|>tool\n{content}<|im_end|>")
            text = "\n".join(parts)
        return {"text": text}

    print("Formatting dataset...")
    dataset = dataset.map(format_to_text, remove_columns=dataset.column_names)

    # Split into train/val
    print(f"Splitting dataset (val_split={args.val_split:.0%})...")
    split = dataset.train_test_split(test_size=args.val_split, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(eval_dataset)}")

    # Setup quantization
    if args.use_4bit:
        print("Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Detect if running under accelerate (FSDP/DeepSpeed) — device_map must be
    # omitted so the distributed framework handles device placement.
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    # Load model
    print(f"Loading model: {args.model}...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    if not is_distributed:
        model_kwargs["device_map"] = "auto"
    else:
        print(f"  Distributed mode (WORLD_SIZE={os.environ['WORLD_SIZE']}): skipping device_map")

    # Use flash attention when available and not using 4-bit quantization
    if not args.use_4bit:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        except ImportError:
            print("  Flash Attention not available, using default (SDPA)")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Generate run name if not provided
    if args.run_name is None:
        model_short = args.model.split("/")[-1]
        args.run_name = f"{model_short}-lora-r{args.lora_rank}"

    # Create output directory
    output_dir = Path(args.checkpoint_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure training
    print("Configuring training...")
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_length,
        logging_steps=args.log_every,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        dataset_text_field="text",
        packing=True,
        seed=42,
        ddp_find_unused_parameters=False,
    )

    # Create trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print()
    print("=" * 70)
    print("Starting LoRA Training")
    print("=" * 70)
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model(str(output_dir / "interrupted"))
        tokenizer.save_pretrained(str(output_dir / "interrupted"))
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save final model
    print("\nSaving final model...")
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print()
    print("=" * 70)
    print("✅ LoRA Training complete!")
    print("=" * 70)
    print(f"Final checkpoint: {final_path}")
    if args.wandb:
        print(f"W&B project: {args.wandb_project}")


if __name__ == "__main__":
    main()
