#!/usr/bin/env python3
"""LoRA Hyperparameter Sweep with Parallel GPU Agents and Early Stopping.

Runs hyperparameter optimization for LoRA fine-tuning using W&B Sweeps.
Supports running 8 parallel agents on 8 GPUs, each training one model.

Features:
- Parallel training on multiple GPUs (one model per GPU)
- Early stopping when eval loss stops improving
- Configurable sweep parameters (LoRA rank, learning rate, epochs)
- 16K context size support
- Reusable for any dataset

Usage:
    # 1. Initialize the sweep (run once)
    python scripts/sweep_lora_hparams.py --init --train-file ./data/filtered_traces/filtered_traces.parquet
    # Grid search:
    python scripts/sweep_lora_hparams.py --init --train-file ./data/filtered_traces/filtered_traces.parquet --sweep-method grid

    # 2. Launch individual agents on each GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_lora_hparams.py --sweep-id <SWEEP_ID> &
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_lora_hparams.py --sweep-id <SWEEP_ID> &
    ...

    # 3. View results
    python scripts/sweep_lora_hparams.py --sweep-id <SWEEP_ID> --results
"""

import argparse
import json
import os
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 16384  # 16K context (further reduced to avoid OOM on parallel runs)
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_PROJECT = "orchestrator-lora-sweep"
DEFAULT_SAMPLE_LIMIT = None  # None = use full dataset; set via --sample-limit to cap

# Sweep configuration focused on performance optimization
SWEEP_CONFIG = {
    "name": "lora-8b-hparam-sweep",
    "method": "bayes",  # Bayesian optimization for efficient search
    "metric": {
        "name": "eval/loss",
        "goal": "minimize"
    },
    "parameters": {
        # LoRA Architecture - higher ranks for better performance
        "lora_rank": {
            "values": [64, 128]
        },
        # Learning rate - log uniform for better exploration
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 2e-4
        },
        # Epochs
        "num_epochs": {
            "values": [2, 3]
        },
    },
    # Stop sweep after finding good result
    "run_cap": 8,  # Maximum number of runs
}

# W&B sweep methods: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
VALID_SWEEP_METHODS = ("bayes", "grid", "random")

# Fixed parameters (not part of sweep)
FIXED_PARAMS = {
    "lora_alpha_ratio": 2.0,  # alpha = rank * 2
    "lora_dropout": 0.05,
    "warmup_ratio": 0.05,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,  # Effective batch size = 4 (reduced from 8 to save memory)
    "use_4bit": True,  # 4-bit quantization to reduce memory usage
    "max_grad_norm": 1.0,  # Gradient clipping for stability
}

# Global state for tracking best results
_best_eval_loss = float("inf")
_best_config = None
_consecutive_no_improvement = 0
_max_no_improvement = 5  # Stop after 5 runs with no improvement


def get_sweep_config(train_file: str, sample_limit: Optional[int] = DEFAULT_SAMPLE_LIMIT) -> Dict[str, Any]:
    """Get sweep configuration with train file and sample limit baked in."""
    config = SWEEP_CONFIG.copy()
    config["parameters"] = SWEEP_CONFIG["parameters"].copy()
    config["parameters"]["train_file"] = {"value": train_file}
    config["parameters"]["sample_limit"] = {"value": sample_limit}  # None = no limit
    return config


def _logspace(min_val: float, max_val: float, n: int) -> list[float]:
    """Generate n log-spaced values from min_val..max_val inclusive."""
    if n <= 1:
        return [float(min_val)]
    if min_val <= 0 or max_val <= 0:
        raise ValueError("logspace requires positive min/max")
    lo = math.log10(min_val)
    hi = math.log10(max_val)
    step = (hi - lo) / (n - 1)
    return [10 ** (lo + i * step) for i in range(n)]


def _coerce_sweep_config_for_method(config: Dict[str, Any]) -> Dict[str, Any]:
    """Make sweep config compatible with the selected W&B method.

    W&B grid sweeps require parameters to be categorical/constant-ish. Our default
    LR spec is log-uniform (good for bayes/random), so for grid we convert it to
    a small set of representative values.
    """
    method = config.get("method")
    if method != "grid":
        return config

    params = config.get("parameters", {})
    lr = params.get("learning_rate", {})
    if isinstance(lr, dict) and lr.get("distribution") == "log_uniform_values":
        lr_min = float(lr["min"])
        lr_max = float(lr["max"])
        # 6-point grid across the configured range (inclusive)
        lr_values = _logspace(lr_min, lr_max, n=6)
        # Keep values human-ish (avoid long floats)
        lr_values = [float(f"{v:.2g}") for v in lr_values]
        params["learning_rate"] = {"values": lr_values}
        config["parameters"] = params

    return config


def init_sweep(train_file: str, project: str = DEFAULT_PROJECT, 
               max_runs: int = 12, sample_limit: Optional[int] = DEFAULT_SAMPLE_LIMIT,
               output_dir: Optional[str] = None, sweep_method: Optional[str] = None) -> str:
    """Initialize a W&B sweep and return the sweep ID."""
    import wandb
    
    config = get_sweep_config(train_file, sample_limit)
    if sweep_method:
        config["method"] = sweep_method
    config["run_cap"] = max_runs
    config = _coerce_sweep_config_for_method(config)
    
    sweep_id = wandb.sweep(
        sweep=config,
        project=project,
    )
    
    # Save sweep_id to output dir for pipeline resumability
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "sweep_id.txt").write_text(sweep_id)
    
    print("=" * 70)
    print("🚀 LoRA Hyperparameter Sweep Initialized!")
    print("=" * 70)
    print(f"Sweep ID: {sweep_id}")
    print(f"Project:  {project}")
    print(f"Method:   {config['method']}")
    print(f"Max Runs: {max_runs}")
    print()
    print("Parameters being optimized:")
    print(f"  LoRA Rank:     {SWEEP_CONFIG['parameters']['lora_rank']['values']}")
    print(f"  Learning Rate: {SWEEP_CONFIG['parameters']['learning_rate']['min']:.0e} - {SWEEP_CONFIG['parameters']['learning_rate']['max']:.0e}")
    print(f"  Epochs:        {SWEEP_CONFIG['parameters']['num_epochs']['values']}")
    print()
    print("Fixed parameters:")
    for k, v in FIXED_PARAMS.items():
        print(f"  {k}: {v}")
    print()
    print("Data:")
    print(f"  Train file:    {train_file}")
    print(f"  Sample limit:  {f'{sample_limit:,}' if sample_limit else 'None (full dataset)'}")
    print(f"  Model:         {DEFAULT_MODEL}")
    print(f"  Context size:  {DEFAULT_MAX_SEQ_LENGTH:,} tokens (16K)")
    print("=" * 70)
    print()
    print("To launch agents, run:")
    for i in range(8):
        print(f"  CUDA_VISIBLE_DEVICES={i} python scripts/sweep_lora_hparams.py --sweep-id {sweep_id} &")
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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    global _best_eval_loss, _best_config, _consecutive_no_improvement

    # Resolve base model (env var overrides default)
    base_model = os.environ.get("SWEEP_BASE_MODEL", DEFAULT_MODEL)
    
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config
    
    # Get parameters from sweep
    lora_rank = config.lora_rank
    lora_alpha = int(lora_rank * FIXED_PARAMS["lora_alpha_ratio"])
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    train_file = config.train_file
    sample_limit = config.get("sample_limit", DEFAULT_SAMPLE_LIMIT)
    
    # Get GPU info
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    # Print configuration
    print()
    print("=" * 70)
    print(f"🔧 LoRA Training Run - GPU {gpu_id}")
    print("=" * 70)
    print(f"Run ID:   {run.id}")
    print(f"Run Name: {run.name}")
    print()
    print("Sweep Parameters:")
    print(f"  LoRA Rank:     {lora_rank}")
    print(f"  LoRA Alpha:    {lora_alpha}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Epochs:        {num_epochs}")
    print()
    print("Fixed Parameters:")
    print(f"  Dropout:       {FIXED_PARAMS['lora_dropout']}")
    print(f"  Batch Size:    {FIXED_PARAMS['batch_size']} x {FIXED_PARAMS['gradient_accumulation_steps']} = {FIXED_PARAMS['batch_size'] * FIXED_PARAMS['gradient_accumulation_steps']}")
    print(f"  4-bit:         {FIXED_PARAMS['use_4bit']}")
    print(f"  Max Length:    {DEFAULT_MAX_SEQ_LENGTH:,} (16K)")
    print(f"  Sample Limit:  {f'{sample_limit:,}' if sample_limit else 'None (full dataset)'}")
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
    print(f"Loading tokenizer for {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
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
    print(f"Loading model: {base_model}...")
    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
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
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0},  # Single GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    
    if FIXED_PARAMS["use_4bit"]:
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=FIXED_PARAMS["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / all_params
    print(f"Trainable: {trainable_params:,} / {all_params:,} ({trainable_pct:.2f}%)")
    
    wandb.log({
        "model/trainable_params": trainable_params,
        "model/total_params": all_params,
        "model/trainable_pct": trainable_pct,
        "model/lora_rank": lora_rank,
        "model/lora_alpha": lora_alpha,
    })
    
    # Create output directory (use SWEEP_OUTPUT_DIR env var if set by pipeline)
    run_name = f"r{lora_rank}-lr{learning_rate:.0e}-e{num_epochs}"
    base_output = Path(os.environ.get("SWEEP_OUTPUT_DIR", "checkpoints/sweep"))
    output_dir = base_output / run_name
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
        max_grad_norm=FIXED_PARAMS["max_grad_norm"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,  # Keep only 1 checkpoint to save disk space
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
        dataloader_pin_memory=False,  # Disable pinned memory to reduce memory usage
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
                early_stopping_patience=5,  # Stop if no improvement for 8 evals
                early_stopping_threshold=5e-4,  # Minimum improvement threshold
            )
        ],
    )
    
    # Resume from checkpoint if one exists for this run config
    resume_from_checkpoint = None
    existing_checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if existing_checkpoints:
        resume_from_checkpoint = str(existing_checkpoints[-1])
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # Train
    print()
    print("=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    print()

    start_time = time.time()

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        train_time = time.time() - start_time

        # Get final metrics
        final_eval = trainer.evaluate()
        final_eval_loss = final_eval["eval_loss"]

        # Update best tracking and save best model
        # Guard against NaN: float('nan') < float('inf') is False in Python,
        # which would silently skip saving even the very first run.
        import math
        if math.isnan(final_eval_loss):
            print(f"\n⚠️ WARNING: eval loss is NaN — skipping best-model check for this run.")
            wandb.log({"error": "eval_loss_is_nan"})
            wandb.finish(exit_code=1)
            return
        is_best = final_eval_loss < _best_eval_loss
        if is_best:
            _best_eval_loss = final_eval_loss
            _best_config = {
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "eval_loss": final_eval_loss,
            }
            _consecutive_no_improvement = 0
            print(f"\n🏆 NEW BEST! Eval Loss: {final_eval_loss:.4f}")
            
            # Save best model with descriptive name
            # Format: best-r{rank}-lr{lr}-e{epochs}-loss{loss}
            lr_str = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
            loss_str = f"{final_eval_loss:.4f}".replace(".", "_")
            best_model_name = f"best-r{lora_rank}-lr{lr_str}-e{num_epochs}-loss{loss_str}"
            best_model_dir = base_output / "best" / best_model_name
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving best model to: {best_model_dir}")
            # Use trainer.model which has the best checkpoint loaded (load_best_model_at_end=True)
            trainer.model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            print(f"✅ Best model saved!")
            
            # Write best_model_info.json for pipeline to pick up
            best_info = {
                "path": str(best_model_dir.absolute()),
                "config": _best_config,
                "eval_loss": final_eval_loss,
                "run_id": run.id,
                "timestamp": datetime.now().isoformat(),
            }
            with open(base_output / "best_model_info.json", "w") as f:
                json.dump(best_info, f, indent=2)
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
            "best/lora_rank": _best_config["lora_rank"] if _best_config else None,
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
        print(f"  LoRA Rank:     {lora_rank}")
        print(f"  Learning Rate: {learning_rate:.2e}")
        print(f"  Epochs:        {num_epochs}")
        print(f"  Eval Loss:     {final_eval_loss:.4f}")
        print(f"  Train Loss:    {train_result.training_loss:.4f}")
        print()
        print("🏆 Current Best:")
        if _best_config:
            print(f"  LoRA Rank:     {_best_config['lora_rank']}")
            print(f"  Learning Rate: {_best_config['learning_rate']:.2e}")
            print(f"  Epochs:        {_best_config['num_epochs']}")
            print(f"  Eval Loss:     {_best_config['eval_loss']:.4f}")
        print("=" * 70)
        
        # Save summary
        summary = {
            "run_id": run.id,
            "gpu": gpu_id,
            "config": {
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
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

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print(f"   This run used: rank={lora_rank}, lr={learning_rate:.2e}, epochs={num_epochs}")
            print(f"   Try reducing: max_seq_length, lora_rank, or batch_size")
            wandb.log({"error": "cuda_out_of_memory", "oom_config": {
                "lora_rank": lora_rank, "learning_rate": learning_rate, "num_epochs": num_epochs
            }})
            wandb.finish(exit_code=1)
            return
        else:
            # Re-raise other RuntimeErrors
            raise

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"error": str(e)})
        # Mark run as failed so _write_best_model_from_wandb doesn't
        # mistake it for a successful "finished" run with missing metrics.
        wandb.finish(exit_code=1)
        return

    finally:
        # Critical: Clean up GPU memory after each run to prevent OOM
        # This is essential when running multiple sweep runs on the same GPU
        import gc
        try:
            if 'trainer' in locals():
                del trainer
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'train_dataset' in locals():
                del train_dataset
            if 'eval_dataset' in locals():
                del eval_dataset
            if 'dataset' in locals():
                del dataset
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ GPU memory cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Cleanup failed: {cleanup_error}")

        wandb.finish()


def run_agent(sweep_id: str, project: str, count: Optional[int] = None, retry_failed: bool = False):
    """Run a sweep agent that pulls configs and trains.
    
    Args:
        sweep_id: W&B sweep ID
        project: W&B project name
        count: Maximum number of runs (None = unlimited)
        retry_failed: If True, retry failed runs with same configs
    """
    import wandb
    
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Stagger startup BEFORE connecting to wandb so we don't consume a run slot while sleeping.
    # Each GPU waits (gpu_index * 30s) to prevent all GPUs from loading the model simultaneously.
    gpu_id_for_delay = int(gpu_id.split(",")[0])
    if gpu_id_for_delay > 0:
        delay = gpu_id_for_delay * 30
        print(f"⏳ Staggered startup: waiting {delay}s before connecting (GPU {gpu_id_for_delay})...")
        time.sleep(delay)

    print(f"🤖 Starting sweep agent on GPU {gpu_id}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project:  {project}")
    if count:
        print(f"Max runs: {count}")
    if retry_failed:
        print(f"Mode: Retry failed runs")
    print()
    
    if retry_failed:
        # Get failed runs and retry them with the same config
        api = wandb.Api()
        entity = api.default_entity
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        
        failed_runs = [run for run in sweep.runs if run.state == "failed" or run.state == "crashed"]
        print(f"Found {len(failed_runs)} failed runs to retry")
        
        for run in failed_runs[:count] if count else failed_runs:
            print(f"\n🔄 Retrying failed run: {run.id}")
            print(f"   Config: {run.config}")
            
            # Initialize a new run with the same config
            with wandb.init(project=project, config=run.config, group=f"retry-{sweep_id}") as retry_run:
                # Set the sweep ID for tracking
                retry_run.config.update({"original_run_id": run.id, "is_retry": True})
                train_with_config()
    else:
        wandb.agent(
            sweep_id,
            function=train_with_config,
            project=project,
            count=count,
        )




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
    
    # Collect results
    results = []
    for run in sweep.runs:
        if run.state == "finished":
            eval_loss = run.summary.get("eval/loss") or run.summary.get("final/eval_loss")
            if eval_loss is not None:
                results.append({
                    "run_id": run.id,
                    "run_name": run.name,
                    "eval_loss": eval_loss,
                    "train_loss": run.summary.get("train/loss") or run.summary.get("final/train_loss"),
                    "lora_rank": run.config.get("lora_rank"),
                    "learning_rate": run.config.get("learning_rate"),
                    "num_epochs": run.config.get("num_epochs"),
                })
    
    if not results:
        print("No completed runs with results yet.")
        return
    
    # Sort by eval loss
    results.sort(key=lambda x: x["eval_loss"])
    
    print("Top 10 Configurations:")
    print("-" * 70)
    print(f"{'#':<3} {'Eval Loss':<12} {'Train Loss':<12} {'Rank':<6} {'LR':<12} {'Epochs':<7}")
    print("-" * 70)
    
    for i, r in enumerate(results[:10], 1):
        lr_str = f"{r['learning_rate']:.2e}" if r['learning_rate'] else "?"
        train_loss = f"{r['train_loss']:.4f}" if r['train_loss'] else "?"
        print(f"{i:<3} {r['eval_loss']:<12.4f} {train_loss:<12} {r['lora_rank']:<6} {lr_str:<12} {r['num_epochs']:<7}")
    
    print("-" * 70)
    print()
    
    # Best configuration
    best = results[0]
    print("🏆 Best Configuration:")
    print(f"  Run ID:        {best['run_id']}")
    print(f"  Eval Loss:     {best['eval_loss']:.4f}")
    print(f"  Train Loss:    {best['train_loss']:.4f}" if best['train_loss'] else "  Train Loss:    N/A")
    print(f"  LoRA Rank:     {best['lora_rank']}")
    print(f"  LoRA Alpha:    {int(best['lora_rank'] * FIXED_PARAMS['lora_alpha_ratio'])}")
    print(f"  Learning Rate: {best['learning_rate']:.2e}")
    print(f"  Epochs:        {best['num_epochs']}")
    print()
    print("To train with these parameters:")
    print(f"  python scripts/train_lora_sft.py \\")
    print(f"    --model {DEFAULT_MODEL} \\")
    print(f"    --lora-rank {best['lora_rank']} \\")
    print(f"    --lora-alpha {int(best['lora_rank'] * FIXED_PARAMS['lora_alpha_ratio'])} \\")
    print(f"    --lr {best['learning_rate']:.2e} \\")
    print(f"    --epochs {best['num_epochs']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Hyperparameter Sweep with Multi-GPU Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize sweep with your dataset
    python scripts/sweep_lora_hparams.py --init \\
        --train-file ./data/filtered_traces/filtered_traces.parquet

    # Run single agent on specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_lora_hparams.py --sweep-id abc123

    # View results
    python scripts/sweep_lora_hparams.py --sweep-id abc123 --results

    # Retry failed runs
    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_lora_hparams.py --sweep-id abc123 --retry-failed
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
        "--sweep-method",
        type=str,
        default=None,
        choices=list(VALID_SWEEP_METHODS),
        help="W&B sweep method (default: use script config, currently 'bayes'; options: bayes, grid, random)"
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
        help="Limit training samples per run (default: None = full dataset)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name/path (overrides DEFAULT_MODEL, sets SWEEP_BASE_MODEL env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints and best model (sets SWEEP_OUTPUT_DIR env var)"
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Print sweep results"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed runs from the sweep instead of continuing with new configs"
    )
    
    args = parser.parse_args()
    
    # Set env vars so train_with_config() picks them up
    if args.model:
        os.environ["SWEEP_BASE_MODEL"] = args.model
    if args.output_dir:
        os.environ["SWEEP_OUTPUT_DIR"] = args.output_dir
    
    if args.init:
        if not args.train_file:
            parser.error("--train-file is required with --init")
        if not Path(args.train_file).exists():
            parser.error(f"Training file not found: {args.train_file}")
        init_sweep(
            args.train_file,
            args.project,
            args.max_runs,
            args.sample_limit,
            args.output_dir,
            sweep_method=args.sweep_method,
        )
        
    elif args.results and args.sweep_id:
        print_results(args.sweep_id, args.project)

    elif args.sweep_id:
        run_agent(args.sweep_id, args.project, args.count, args.retry_failed)
        
    else:
        parser.print_help()
        print("\n❌ Must specify --init, --sweep-id, or --results with --sweep-id")
        sys.exit(1)


if __name__ == "__main__":
    main()
