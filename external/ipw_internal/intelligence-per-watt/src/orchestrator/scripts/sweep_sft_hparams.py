#!/usr/bin/env python3
"""SFT Hyperparameter Sweep with Parallel GPU Agents and Early Stopping.

Runs hyperparameter optimization for SFT fine-tuning using W&B Sweeps.
Supports running 8 parallel agents on 8 GPUs, each training one model.

Features:
- Parallel training on multiple GPUs (one model per GPU)
- Early stopping when eval loss stops improving
- Configurable sweep parameters (SFT rank, learning rate, epochs)
- 128K context size support
- Reusable for any dataset

Usage:
    # 1. Initialize the sweep (run once)
    python scripts/sweep_sft_hparams.py --init --train-file ./data/filtered_traces/filtered_traces.parquet
    
    # 2. Launch all 8 GPU agents in parallel (run once, spawns 8 processes)
    python scripts/sweep_sft_hparams.py --sweep-id <SWEEP_ID> --launch-all
    
    # Or manually launch individual agents:
    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_sft_hparams.py --sweep-id <SWEEP_ID> &
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_sft_hparams.py --sweep-id <SWEEP_ID> &
    ...
    
    # 3. View results
    python scripts/sweep_sft_hparams.py --sweep-id <SWEEP_ID> --results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 131072  # 128K context
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_PROJECT = "orchestrator-sft-sweep"
DEFAULT_SAMPLE_LIMIT = 2000  # Limit samples for faster sweep iterations

# Sweep configuration focused on performance optimization
SWEEP_CONFIG = {
    "name": "sft-8b-hparam-sweep",
    "method": "bayes",  # Bayesian optimization for efficient search
    "metric": {
        "name": "eval/loss",
        "goal": "minimize"
    },
    "parameters": {
        # Learning rate - log uniform for better exploration
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4
        },
        # Epochs
        "num_epochs": {
            "values": [1, 2, 3, 4]
        },
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 50,  # Minimum steps before early termination
        "eta": 3,
        "max_iter": 500,  # Maximum steps per run
    },
    # Stop sweep after finding good result
    "run_cap": 24,  # Maximum number of runs
}

# Fixed parameters (not part of sweep)
FIXED_PARAMS = {
    "warmup_ratio": 0.05,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,  # Effective batch size = 8
    "use_4bit": False,  # Full precision for better performance
}

# Global state for tracking best results
_best_eval_loss = float("inf")
_best_config = None
_consecutive_no_improvement = 0
_max_no_improvement = 5  # Stop after 5 runs with no improvement


def get_sweep_config(train_file: str, sample_limit: int = DEFAULT_SAMPLE_LIMIT) -> Dict[str, Any]:
    """Get sweep configuration with train file and sample limit baked in."""
    config = SWEEP_CONFIG.copy()
    config["parameters"] = SWEEP_CONFIG["parameters"].copy()
    config["parameters"]["train_file"] = {"value": train_file}
    config["parameters"]["sample_limit"] = {"value": sample_limit}
    return config


def init_sweep(train_file: str, project: str = DEFAULT_PROJECT, 
               max_runs: int = 24, sample_limit: int = DEFAULT_SAMPLE_LIMIT) -> str:
    """Initialize a W&B sweep and return the sweep ID."""
    import wandb
    
    config = get_sweep_config(train_file, sample_limit)
    config["run_cap"] = max_runs
    
    sweep_id = wandb.sweep(
        sweep=config,
        project=project,
    )
    
    print("=" * 70)
    print("🚀 SFT Hyperparameter Sweep Initialized!")
    print("=" * 70)
    print(f"Sweep ID: {sweep_id}")
    print(f"Project:  {project}")
    print(f"Method:   {config['method']}")
    print(f"Max Runs: {max_runs}")
    print()
    print("Parameters being optimized:")
    print(f"  Learning Rate: {SWEEP_CONFIG['parameters']['learning_rate']['min']:.0e} - {SWEEP_CONFIG['parameters']['learning_rate']['max']:.0e}")
    print(f"  Epochs:        {SWEEP_CONFIG['parameters']['num_epochs']['values']}")
    print()
    print("Fixed parameters:")
    for k, v in FIXED_PARAMS.items():
        print(f"  {k}: {v}")
    print()
    print("Data:")
    print(f"  Train file:    {train_file}")
    print(f"  Sample limit:  {sample_limit:,} (for faster sweep iterations)")
    print(f"  Model:         {DEFAULT_MODEL}")
    print(f"  Context size:  {DEFAULT_MAX_SEQ_LENGTH:,} tokens (128K)")
    print("=" * 70)
    print()
    print("To launch agents, run:")
    print(f"  python scripts/sweep_sft_hparams.py --sweep-id {sweep_id} --launch-all")
    print()
    print("Or launch individually:")
    for i in range(8):
        print(f"  CUDA_VISIBLE_DEVICES={i} python scripts/sweep_sft_hparams.py --sweep-id {sweep_id} &")
    print("=" * 70)
    
    return sweep_id


def train_with_config():
    """Training function called by W&B sweep agent."""
    import torch
    import wandb
    from datasets import load_dataset, Dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from trl import SFTTrainer, SFTConfig
    
    global _best_eval_loss, _best_config, _consecutive_no_improvement
    
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config
    
    # Get parameters from sweep
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    train_file = config.train_file
    sample_limit = config.get("sample_limit", DEFAULT_SAMPLE_LIMIT)
    
    # Get GPU info
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    # Print configuration
    print()
    print("=" * 70)
    print(f"🔧 SFT Training Run - GPU {gpu_id}")
    print("=" * 70)
    print(f"Run ID:   {run.id}")
    print(f"Run Name: {run.name}")
    print()
    print("Sweep Parameters:")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Epochs:        {num_epochs}")
    print()
    print("Fixed Parameters:")
    print(f"  Batch Size:    {FIXED_PARAMS['batch_size']} x {FIXED_PARAMS['gradient_accumulation_steps']} = {FIXED_PARAMS['batch_size'] * FIXED_PARAMS['gradient_accumulation_steps']}")
    print(f"  4-bit:         {FIXED_PARAMS['use_4bit']}")
    print(f"  Max Length:    {DEFAULT_MAX_SEQ_LENGTH:,} (128K)")
    print(f"  Sample Limit:  {sample_limit:,}")
    print()
    print(f"Current Best Eval Loss: {_best_eval_loss:.4f}")
    print("=" * 70)
    print()
    
    # Check if train file exists
    train_path = Path(train_file)
    if not train_path.exists():
        print(f"❌ Error: Training file not found: {train_file}")
        wandb.log({"error": "train_file_not_found"})
        wandb.finish()
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL, 
        trust_remote_code=True,
        model_max_length=DEFAULT_MAX_SEQ_LENGTH,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset from {train_file}...")
    suffix = train_path.suffix.lower()
    if suffix == ".parquet":
        dataset = Dataset.from_parquet(str(train_path))
    elif suffix in [".json", ".jsonl"]:
        dataset = load_dataset("json", data_files=str(train_path))["train"]
    else:
        # Try loading as HF dataset directory
        from datasets import load_from_disk
        dataset = load_from_disk(str(train_path))
    
    print(f"Loaded {len(dataset)} samples")
    
    # Apply sample limit for faster sweep iterations
    if sample_limit and len(dataset) > sample_limit:
        print(f"Limiting to {sample_limit:,} samples for sweep...")
        dataset = dataset.shuffle(seed=42).select(range(sample_limit))
        print(f"Using {len(dataset):,} samples")
    
    # Format conversations to text
    def format_to_text(example):
        messages = example.get("conversations", [])
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
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
    print(f"Splitting dataset (val_split={DEFAULT_VAL_SPLIT:.0%})...")
    split = dataset.train_test_split(test_size=DEFAULT_VAL_SPLIT, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset):,}, Val: {len(eval_dataset):,}")
    
    # Log dataset info
    wandb.log({
        "dataset/train_samples": len(train_dataset),
        "dataset/eval_samples": len(eval_dataset),
    })
    
    # Setup quantization if enabled
    if FIXED_PARAMS["use_4bit"]:
        print("Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model - try flash_attention_2, fall back to sdpa if unavailable
    print(f"Loading model: {DEFAULT_MODEL}...")
    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},  # Single GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        print(f"Using attention: {attn_impl}")
    except ImportError:
        attn_impl = "sdpa"
        print(f"Flash Attention 2 not available, falling back to: {attn_impl}")
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},  # Single GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    
    if FIXED_PARAMS["use_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params
    print(f"Trainable: {trainable_params:,} / {all_params:,} ({trainable_pct:.2f}%)")
    
    wandb.log({
        "model/trainable_params": trainable_params,
        "model/total_params": all_params,
        "model/trainable_pct": trainable_pct,
    })
    
    # Create output directory
    run_name = f"lr{learning_rate:.0e}-e{num_epochs}"
    output_dir = Path("checkpoints/sweep") / run.id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure training with early stopping
    print("Configuring training...")
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=FIXED_PARAMS["batch_size"],
        per_device_eval_batch_size=FIXED_PARAMS["batch_size"],
        gradient_accumulation_steps=FIXED_PARAMS["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_ratio=FIXED_PARAMS["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_length=DEFAULT_MAX_SEQ_LENGTH,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=50,  # Evaluate frequently for early stopping
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name=run_name,
        dataset_text_field="text",
        packing=False,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer with early stopping callback
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Stop if no improvement for 3 evals
                early_stopping_threshold=0.001,  # Minimum improvement threshold
            )
        ],
    )
    
    # Train
    print()
    print("=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        train_time = time.time() - start_time
        
        # Get final metrics
        final_eval = trainer.evaluate()
        final_eval_loss = final_eval["eval_loss"]
        
        # Update best tracking
        if final_eval_loss < _best_eval_loss:
            _best_eval_loss = final_eval_loss
            _best_config = {
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "eval_loss": final_eval_loss,
            }
            _consecutive_no_improvement = 0
            print(f"\n🏆 NEW BEST! Eval Loss: {final_eval_loss:.4f}")
        else:
            _consecutive_no_improvement += 1
            print(f"\nNo improvement. Current: {final_eval_loss:.4f}, Best: {_best_eval_loss:.4f}")
            print(f"Consecutive no improvement: {_consecutive_no_improvement}/{_max_no_improvement}")
        
        # Log final metrics
        wandb.log({
            "final/eval_loss": final_eval_loss,
            "final/train_loss": train_result.training_loss,
            "final/train_runtime": train_time,
            "best/eval_loss": _best_eval_loss,
            "best/learning_rate": _best_config["learning_rate"] if _best_config else None,
        })
        
        # Print summary
        print()
        print("=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        print(f"Run ID:        {run.id}")
        print(f"GPU:           {gpu_id}")
        print(f"Train Time:    {train_time/60:.1f} min")
        print()
        print("This Run:")
        print(f"  Learning Rate: {learning_rate:.2e}")
        print(f"  Epochs:        {num_epochs}")
        print(f"  Eval Loss:     {final_eval_loss:.4f}")
        print(f"  Train Loss:    {train_result.training_loss:.4f}")
        print()
        print("🏆 Current Best:")
        if _best_config:
            print(f"  Learning Rate: {_best_config['learning_rate']:.2e}")
            print(f"  Epochs:        {_best_config['num_epochs']}")
            print(f"  Eval Loss:     {_best_config['eval_loss']:.4f}")
        print("=" * 70)
        
        # Save summary
        summary = {
            "run_id": run.id,
            "gpu": gpu_id,
            "config": {
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            },
            "results": {
                "eval_loss": final_eval_loss,
                "train_loss": train_result.training_loss,
                "train_time": train_time,
            },
            "best_so_far": _best_config,
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Check if we should stop the sweep due to no improvement
        if _consecutive_no_improvement >= _max_no_improvement:
            print(f"\n⚠️ No improvement for {_max_no_improvement} consecutive runs.")
            print("Consider stopping the sweep early.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"error": str(e)})
    
    finally:
        wandb.finish()


def run_agent(sweep_id: str, project: str, count: Optional[int] = None):
    """Run a sweep agent that pulls configs and trains."""
    import wandb
    
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"🤖 Starting sweep agent on GPU {gpu_id}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project:  {project}")
    if count:
        print(f"Max runs: {count}")
    print()
    
    wandb.agent(
        sweep_id,
        function=train_with_config,
        project=project,
        count=count,
    )


def launch_all_agents(
    sweep_id: str,
    project: str,
    num_gpus: int = 8,
    runs_per_agent: Optional[int] = None,
):
    """Launch agents on all GPUs using subprocess."""
    import wandb
    
    # Get the entity (user/org) from wandb
    try:
        api = wandb.Api()
        entity = api.default_entity
    except:
        entity = "ENTITY"  # Fallback if can't get entity
    
    print(f"🚀 Launching {num_gpus} agents...")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project:  {project}")
    print(f"Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print()
    
    script_path = Path(__file__).absolute()
    processes = []
    
    for gpu_id in range(num_gpus):
        cmd = [
            sys.executable,
            str(script_path),
            "--sweep-id", sweep_id,
            "--project", project,
        ]
        if runs_per_agent:
            cmd.extend(["--count", str(runs_per_agent)])
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Create log file for each agent
        log_file = Path(f"logs/sweep_gpu{gpu_id}.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "w")
        
        print(f"  GPU {gpu_id}: Starting agent... (log: {log_file})")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu_id, proc, log_handle))
    
    print()
    print(f"✅ Launched {num_gpus} agents")
    print(f"Monitor progress at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print()
    print("Waiting for agents to complete...")
    print("Press Ctrl+C to stop all agents")
    
    try:
        for gpu_id, proc, log_handle in processes:
            proc.wait()
            log_handle.close()
            print(f"  GPU {gpu_id}: Agent finished (exit code: {proc.returncode})")
    except KeyboardInterrupt:
        print("\n\n⚠️ Stopping all agents...")
        for gpu_id, proc, log_handle in processes:
            proc.terminate()
        # Wait a bit for graceful termination
        time.sleep(2)
        # Force kill any that didn't stop
        for gpu_id, proc, log_handle in processes:
            if proc.poll() is None:  # Still running
                print(f"  Force killing GPU {gpu_id} agent...")
                proc.kill()
            log_handle.close()
        print("All agents stopped.")


def print_results(sweep_id: str, project: str):
    """Print summary of sweep results."""
    import wandb
    
    api = wandb.Api()
    entity = api.default_entity
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    print()
    print("=" * 70)
    print("📊 Hyperparameter Sweep Results")
    print("=" * 70)
    print(f"Sweep ID: {sweep_id}")
    print(f"Project:  {project}")
    print(f"State:    {sweep.state}")
    print(f"Runs:     {len(sweep.runs)}")
    print()
    
    # Collect results - try multiple summary keys (HF Trainer vs TRL vs our explicit logs)
    results = []
    for run in sweep.runs:
        eval_loss = (
            run.summary.get("eval/loss")
            or run.summary.get("eval_loss")
            or run.summary.get("final/eval_loss")
        )
        if eval_loss is not None:
            results.append({
                "run_id": run.id,
                "run_name": run.name,
                "eval_loss": eval_loss,
                "train_loss": (
                    run.summary.get("train/loss")
                    or run.summary.get("train_loss")
                    or run.summary.get("final/train_loss")
                ),
                "learning_rate": run.config.get("learning_rate"),
                "num_epochs": run.config.get("num_epochs"),
            })
    
    if not results:
        print("No completed runs with results yet.")
        # Debug: show run states and sample summary keys
        states = {}
        for run in sweep.runs:
            states[run.state] = states.get(run.state, 0) + 1
        print(f"Run states: {states}")
        if sweep.runs:
            sample = sweep.runs[0]
            summary_keys = [k for k in sample.summary.keys() if "loss" in k.lower() or "eval" in k.lower()]
            if summary_keys:
                print(f"Sample run summary keys (loss/eval): {summary_keys[:15]}")
            else:
                print(f"Sample run summary keys: {list(sample.summary.keys())[:15]}")
        return
    
    # Sort by eval loss
    results.sort(key=lambda x: x["eval_loss"])
    
    print("Top 10 Configurations:")
    print("-" * 70)
    print(f"{'#':<3} {'Eval Loss':<12} {'Train Loss':<12} {'LR':<12} {'Epochs':<7}")
    print("-" * 70)
    
    for i, r in enumerate(results[:10], 1):
        lr_str = f"{r['learning_rate']:.2e}" if r['learning_rate'] else "?"
        train_loss = f"{r['train_loss']:.4f}" if r['train_loss'] else "?"
        print(f"{i:<3} {r['eval_loss']:<12.4f} {train_loss:<12} {lr_str:<12} {r['num_epochs']:<7}")
    
    print("-" * 70)
    print()
    
    # Best configuration
    best = results[0]
    print("🏆 Best Configuration:")
    print(f"  Run ID:        {best['run_id']}")
    print(f"  Eval Loss:     {best['eval_loss']:.4f}")
    print(f"  Train Loss:    {best['train_loss']:.4f}" if best['train_loss'] else "  Train Loss:    N/A")
    print(f"  Learning Rate: {best['learning_rate']:.2e}")
    print(f"  Epochs:        {best['num_epochs']}")
    print()
    print("To train with these parameters:")
    print(f"  python scripts/train_sft_trl.py \\")
    print(f"    --model {DEFAULT_MODEL} \\")
    print(f"    --lr {best['learning_rate']:.2e} \\")
    print(f"    --epochs {best['num_epochs']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="SFT Hyperparameter Sweep with Multi-GPU Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize sweep with your dataset
    python scripts/sweep_sft_hparams.py --init \\
        --train-file ./data/filtered_traces/filtered_traces.parquet
    
    # Launch all 8 GPU agents
    python scripts/sweep_sft_hparams.py --sweep-id abc123 --launch-all
    
    # Or run single agent on specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_sft_hparams.py --sweep-id abc123
    
    # View results
    python scripts/sweep_sft_hparams.py --sweep-id abc123 --results
        """
    )
    
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new sweep"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Sweep ID to join as agent"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"W&B project name (default: {DEFAULT_PROJECT})"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to training data file (required for --init)"
    )
    parser.add_argument(
        "--launch-all",
        action="store_true",
        help="Launch agents on all GPUs"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use with --launch-all (default: 8)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Max runs per agent"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=24,
        help="Maximum total runs for sweep (default: 24)"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"Limit samples per run for faster iterations (default: {DEFAULT_SAMPLE_LIMIT})"
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Print sweep results"
    )
    
    args = parser.parse_args()
    
    if args.init:
        if not args.train_file:
            parser.error("--train-file is required with --init")
        if not Path(args.train_file).exists():
            parser.error(f"Training file not found: {args.train_file}")
        init_sweep(args.train_file, args.project, args.max_runs, args.sample_limit)
        
    elif args.results and args.sweep_id:
        print_results(args.sweep_id, args.project)
        
    elif args.sweep_id and args.launch_all:
        launch_all_agents(
            args.sweep_id,
            args.project,
            args.num_gpus,
            args.count,
        )
        
    elif args.sweep_id:
        run_agent(args.sweep_id, args.project, args.count)
        
    else:
        parser.print_help()
        print("\n❌ Must specify --init, --sweep-id, or --results with --sweep-id")
        sys.exit(1)


if __name__ == "__main__":
    main()
