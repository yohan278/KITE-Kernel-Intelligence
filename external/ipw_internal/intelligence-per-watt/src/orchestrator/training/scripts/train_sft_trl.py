#!/usr/bin/env python3
"""SFT Training script using TRL's SFTTrainer with FSDP.

This script trains Qwen3-8B on orchestrator trajectories using multi-turn
conversation format.

Usage:
    # Single GPU
    python scripts/train_sft_trl.py

    # Multi-GPU with FSDP (8 GPUs)
    accelerate launch --config_file training/configs/accelerate_fsdp.yaml scripts/train_sft_trl.py
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer, SFTConfig


@dataclass
class ScriptArguments:
    """Arguments for training script."""

    model_name: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Model name or path"}
    )
    train_file: str = field(
        default="data/trajectories/trajectories.jsonl",
        metadata={"help": "Path to training JSONL file"}
    )
    output_dir: str = field(
        default="checkpoints/qwen3-8b-orchestrator",
        metadata={"help": "Output directory for checkpoints"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    max_seq_length: int = field(
        default=32768,
        metadata={"help": "Maximum sequence length"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use Weights & Biases logging"}
    )
    wandb_project: str = field(
        default="orchestrator-sft",
        metadata={"help": "W&B project name"}
    )


def load_jsonl_dataset(file_path: str) -> Dataset:
    """Load JSONL file as HuggingFace Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_conversation(example):
    """Format conversation for training.

    Converts the conversations list into the format expected by the model.
    """
    messages = example["conversations"]
    return {"messages": messages}


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup wandb
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Per-device batch size: {args.per_device_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_batch_size * args.gradient_accumulation_steps * 8}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max seq length: {args.max_seq_length}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use PyTorch SDPA instead of flash_attention_2
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Load dataset
    print(f"\nLoading dataset from {args.train_file}...")
    dataset = load_jsonl_dataset(args.train_file)
    print(f"Loaded {len(dataset)} samples")

    # Format conversations
    dataset = dataset.map(format_conversation, remove_columns=["conversations"])

    # Split into train/val (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(eval_dataset)}")

    # Show sample
    print("\nSample formatted message:")
    sample = train_dataset[0]["messages"]
    for msg in sample[:2]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100]
        print(f"  {role}: {content}...")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=args.per_device_batch_size,
        bf16=True,
        gradient_checkpointing=False,  # Using FSDP activation checkpointing instead
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        packing=False,  # Don't pack sequences for multi-turn
        dataset_text_field=None,  # Using messages format
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen3-8b-orchestrator-{args.num_epochs}ep",
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {args.output_dir}/final...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
