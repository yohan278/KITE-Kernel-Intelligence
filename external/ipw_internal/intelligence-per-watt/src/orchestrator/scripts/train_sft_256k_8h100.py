#!/usr/bin/env python3
"""Full SFT training script for orchestrator (256K context, 8x H100).

Trains Qwen3-8B with full fine-tuning on successful trajectories from Moonlight-16B-A3B teacher.
Uses 256K context window on 8x H100 80GB GPUs.

Features:
- Full SFT (no LoRA) - all parameters trained
- Gradient checkpointing for 256K context
- GenerationMonitorCallback to check quality every 500 steps
- Filter to successful trajectories only

Usage:
    # Filter successful trajectories first:
    python -c "
    import json
    with open('data/moonlight_trajectories_50k/trajectories.jsonl') as f, \
         open('data/moonlight_trajectories_50k/successful_trajectories.jsonl', 'w') as out:
        for line in f:
            item = json.loads(line)
            if item.get('success', False):
                out.write(line)
    "

    # Launch training on 8 GPUs:
    accelerate launch \
        --config_file src/orchestrator/configs/accelerate_4h100_fsdp.yaml \
        src/orchestrator/scripts/train_sft_256k_8h100.py
"""

import gc
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Reduce CUDA memory fragmentation
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer import Trainer
from trl import SFTConfig, SFTTrainer


class ChunkedLossSFTTrainer(SFTTrainer):
    """SFTTrainer compatible with chunked loss (minimal logits in model output).

    The chunked forward returns only 1 token of logits to save ~14 GB of GPU
    memory.  TRL's default ``compute_loss`` tries to compute per-token accuracy
    and entropy from full-sequence logits and crashes when logits are minimal.

    This subclass calls the grandparent (HF ``Trainer.compute_loss``) to get
    the loss from the model, then tracks training tokens, but skips the
    logits-derived metrics (entropy, token accuracy) that need the full tensor.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        # Disable KV cache during training (same as TRL)
        inputs["use_cache"] = False

        # Call Trainer.compute_loss (grandparent), which just calls
        # model(**inputs) and reads outputs.loss.  This skips TRL's
        # logits-based metrics that require the full (seq_len, vocab) tensor.
        (loss, outputs) = Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Track training tokens (same as TRL's SFTTrainer)
        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens = self.accelerator.gather_for_metrics(
                    inputs["attention_mask"].sum()
                ).sum().item()
            elif "position_ids" in inputs:
                local = torch.tensor(
                    inputs["position_ids"].size(1), device=inputs["position_ids"].device
                )
                num_tokens = self.accelerator.gather_for_metrics(local).sum().item()
            else:
                num_tokens = 0
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # NOTE: entropy and per-token accuracy are skipped because our
        # chunked forward returns only 1 logit to save ~14 GB GPU memory.
        # Loss (logged by the Trainer) is the primary training signal.

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Chunked LM-head + Cross-Entropy Loss
# ---------------------------------------------------------------------------
# The default Qwen3ForCausalLM.forward() materializes the FULL logits tensor:
#   logits = lm_head(hidden_states)  → (batch, seq_len, vocab_size)
# For seq_len=50K and vocab=152K in bf16, that's 14.2 GB.
# Then ForCausalLMLoss upcasts to float32 → 28.3 GB.
# Both coexist during the upcast = 42.5 GB on ONE GPU, just for logits.
# During backward, add another 28 GB for the gradient → ~56 GB total.
#
# Fix: compute lm_head + CE in chunks of `chunk_size` tokens at a time.
# Each chunk's float32 logits ≈ 2.4 GB (for chunk_size=4096).
# Gradient-checkpoint each chunk so autograd doesn't keep all of them.
# Peak logits memory drops from ~56 GB to ~5 GB.
# ---------------------------------------------------------------------------

CHUNKED_LOSS_CHUNK_SIZE = 4096


def _compute_chunk_ce(hidden_chunk, lm_weight, lm_bias, label_chunk, vocab_size, ignore_index):
    """Compute cross-entropy loss for a single chunk of the sequence.

    This function is called under torch.utils.checkpoint so its intermediates
    (the float32 logits) are NOT saved during forward — they are recomputed
    during backward.  This means only ONE chunk's logits live on GPU at a time.
    """
    logits = F.linear(hidden_chunk, lm_weight, lm_bias).float()
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        label_chunk.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss


def _chunked_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    cache_position=None,
    logits_to_keep=0,
    **kwargs,
):
    """Drop-in replacement for Qwen3ForCausalLM.forward that computes
    lm_head + CE loss in chunks, never materializing the full logits tensor.

    Inference mode (labels=None) falls back to the original forward.
    """
    # Pop Trainer-injected keys that the base model doesn't understand
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)

    # Inference mode — use the original (un-patched) forward
    if labels is None:
        # Re-inject num_items_in_batch for the original forward
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return self._original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    # ---- Training mode: chunked lm_head + loss ----

    # 1. Run the transformer backbone (all layers), no lm_head yet
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state  # (B, T, H) — kept in bf16

    # 2. Causal LM shift: position i predicts token i+1
    shift_hidden = hidden_states[:, :-1, :]       # (B, T-1, H)
    shift_labels = labels[:, 1:].contiguous()      # (B, T-1)

    vocab_size = self.config.vocab_size
    seq_len = shift_hidden.shape[1]
    chunk_size = CHUNKED_LOSS_CHUNK_SIZE
    ignore_index = -100

    lm_weight = self.lm_head.weight
    lm_bias = getattr(self.lm_head, "bias", None)

    # 3. Count valid tokens (no gradient needed)
    with torch.no_grad():
        total_valid = (shift_labels != ignore_index).sum().item()

    if total_valid == 0:
        loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
        loss.requires_grad_(True)
    else:
        # 4. Chunked lm_head + CE with gradient checkpointing per chunk
        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

        for i in range(0, seq_len, chunk_size):
            j = min(i + chunk_size, seq_len)
            chunk_labels = shift_labels[:, i:j]

            # Skip all-padding chunks
            with torch.no_grad():
                if (chunk_labels == ignore_index).all():
                    continue

            chunk_hidden = shift_hidden[:, i:j, :]

            # grad_checkpoint: forward computes & discards logits;
            # backward recomputes them.  Only 1 chunk on GPU at a time.
            chunk_loss = grad_checkpoint(
                _compute_chunk_ce,
                chunk_hidden,
                lm_weight,
                lm_bias,
                chunk_labels,
                vocab_size,
                ignore_index,
                use_reentrant=False,
            )
            total_loss = total_loss + chunk_loss

        # 5. Proper loss averaging
        if num_items_in_batch is not None:
            num_items = num_items_in_batch
            if torch.is_tensor(num_items):
                num_items = num_items.to(total_loss.device)
            loss = total_loss / num_items
        else:
            loss = total_loss / total_valid

    # 6. Return minimal logits (1 token) — the Trainer only uses `loss`
    with torch.no_grad():
        last_logits = self.lm_head(hidden_states[:, -1:, :])

    return CausalLMOutputWithPast(
        loss=loss,
        logits=last_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_model_for_chunked_loss(model, chunk_size=4096):
    """Monkey-patch the model's forward to use chunked lm_head + CE loss.

    Must be called BEFORE the model is wrapped by FSDP (i.e., before
    creating the Trainer).  FSDP will call the patched forward correctly
    because it invokes ``module.forward()`` on the inner model.
    """
    global CHUNKED_LOSS_CHUNK_SIZE
    CHUNKED_LOSS_CHUNK_SIZE = chunk_size

    # Save the original forward (with decorators) so inference can use it
    model._original_forward = model.forward

    # Replace with chunked version
    model.forward = types.MethodType(_chunked_forward, model)

    vocab_size = model.config.vocab_size
    peak_chunk_gb = chunk_size * vocab_size * 4 / 1024**3
    full_gb = 50_000 * vocab_size * 4 / 1024**3
    print(f"\n[ChunkedLoss] Patched model forward for chunked lm_head + CE loss")
    print(f"  chunk_size     = {chunk_size:,} tokens")
    print(f"  vocab_size     = {vocab_size:,}")
    print(f"  Peak per chunk = {peak_chunk_gb:.2f} GB (float32)")
    print(f"  Without patch  = {full_gb:.1f} GB (float32, full sequence)")
    print(f"  Savings        ≈ {full_gb - peak_chunk_gb:.1f} GB\n")

class MemoryProfilerCallback(TrainerCallback):
    """Callback to monitor GPU memory usage every step and detect leaks.

    Logs allocated, reserved, and peak memory. Detects monotonic growth
    (likely leak) after a warmup period.
    """

    def __init__(
        self,
        log_every: int = 1,
        warn_growth_threshold_mb: float = 100.0,
        warmup_steps: int = 20,
        empty_cache_every: int = 10,
        snapshot_dir: str | None = None,
    ):
        """Initialize the memory profiler.

        Args:
            log_every: Log memory stats every N steps.
            warn_growth_threshold_mb: Warn if memory grows more than this per step (avg).
            warmup_steps: Steps to ignore before leak detection (memory is unstable early).
            empty_cache_every: Call torch.cuda.empty_cache() every N steps.
            snapshot_dir: If set, save CUDA memory snapshots here for offline analysis.
        """
        self.log_every = log_every
        self.warn_growth_threshold_mb = warn_growth_threshold_mb
        self.warmup_steps = warmup_steps
        self.empty_cache_every = empty_cache_every
        self.snapshot_dir = snapshot_dir

        # Tracking
        self.memory_history: list[dict] = []
        self.peak_allocated_mb = 0.0
        self._snapshot_active = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Reset CUDA peak stats and optionally start memory recording."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            # Enable CUDA memory history for snapshot debugging
            if self.snapshot_dir:
                Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
                try:
                    torch.cuda.memory._record_memory_history(max_entries=100_000)
                    self._snapshot_active = True
                    print(f"[MemoryProfiler] CUDA memory history recording enabled -> {self.snapshot_dir}")
                except Exception as e:
                    print(f"[MemoryProfiler] Could not start memory history: {e}")

            self._log_memory(0, tag="TRAIN_BEGIN")

    def on_step_end(self, args, state, control, **kwargs):
        """Log memory after each step, detect leaks, periodically clear cache."""
        if not torch.cuda.is_available():
            return

        step = state.global_step

        # Periodic cache clearing to reduce fragmentation
        if step > 0 and step % self.empty_cache_every == 0:
            torch.cuda.empty_cache()

        # Log memory stats
        if step % self.log_every == 0:
            stats = self._log_memory(step)

            # Leak detection after warmup
            if step > self.warmup_steps and len(self.memory_history) >= 10:
                self._check_for_leak(step, stats)

    def on_train_end(self, args, state, control, **kwargs):
        """Print summary and save snapshot if enabled."""
        if not torch.cuda.is_available():
            return

        self._log_memory(state.global_step, tag="TRAIN_END")
        self._print_summary()

        # Save final memory snapshot
        if self._snapshot_active and self.snapshot_dir:
            try:
                snapshot_path = Path(self.snapshot_dir) / "memory_snapshot_final.pickle"
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
                print(f"[MemoryProfiler] Final snapshot saved to {snapshot_path}")
                print(f"  -> Visualize: python -m torch.cuda.memory._snapshot {snapshot_path}")
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as e:
                print(f"[MemoryProfiler] Could not save snapshot: {e}")

    def _log_memory(self, step: int, tag: str = "") -> dict:
        """Collect and log memory statistics."""
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        free_mb = free_mem / 1024**2
        total_mb = total_mem / 1024**2
        used_pct = (1.0 - free_mem / total_mem) * 100.0

        stats = {
            "step": step,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_mb": peak,
            "free_mb": free_mb,
            "total_mb": total_mb,
            "used_pct": used_pct,
            "time": time.time(),
        }
        self.memory_history.append(stats)
        self.peak_allocated_mb = max(self.peak_allocated_mb, allocated)

        prefix = f"[MemoryProfiler{f' {tag}' if tag else ''}]"
        print(
            f"{prefix} Step {step:>5d} | "
            f"Alloc: {allocated:>8.1f} MB | "
            f"Reserved: {reserved:>8.1f} MB | "
            f"Peak: {peak:>8.1f} MB | "
            f"GPU Used: {used_pct:>5.1f}% ({total_mb - free_mb:.0f}/{total_mb:.0f} MB)"
        )
        return stats

    def _check_for_leak(self, step: int, current_stats: dict):
        """Check if memory is growing monotonically (leak indicator)."""
        # Compare last 10 entries
        recent = self.memory_history[-10:]
        deltas = [
            recent[i + 1]["allocated_mb"] - recent[i]["allocated_mb"]
            for i in range(len(recent) - 1)
        ]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0

        # All positive deltas = monotonic growth = likely leak
        all_growing = all(d > 0 for d in deltas)
        growth_too_fast = avg_delta > self.warn_growth_threshold_mb

        if all_growing and growth_too_fast:
            print(
                f"[MemoryProfiler WARNING] Possible memory leak detected at step {step}!\n"
                f"  Memory grew every step for last {len(deltas)} steps.\n"
                f"  Average growth: {avg_delta:.1f} MB/step\n"
                f"  Current allocated: {current_stats['allocated_mb']:.1f} MB\n"
                f"  If this continues, OOM in ~{(current_stats['free_mb']) / max(avg_delta, 0.1):.0f} steps"
            )

            # Save emergency snapshot on leak detection
            if self._snapshot_active and self.snapshot_dir:
                try:
                    snapshot_path = Path(self.snapshot_dir) / f"memory_snapshot_leak_step{step}.pickle"
                    torch.cuda.memory._dump_snapshot(str(snapshot_path))
                    print(f"  -> Emergency snapshot saved to {snapshot_path}")
                except Exception:
                    pass

    def _print_summary(self):
        """Print end-of-training memory summary."""
        if not self.memory_history:
            return

        print(f"\n{'=' * 70}")
        print("[MemoryProfiler] Training Memory Summary")
        print(f"{'=' * 70}")

        all_alloc = [s["allocated_mb"] for s in self.memory_history]
        print(f"  Peak allocated:  {max(all_alloc):>8.1f} MB")
        print(f"  Min allocated:   {min(all_alloc):>8.1f} MB")
        print(f"  Final allocated: {all_alloc[-1]:>8.1f} MB")

        # Growth analysis
        if len(self.memory_history) > self.warmup_steps:
            post_warmup = [s["allocated_mb"] for s in self.memory_history[self.warmup_steps:]]
            if len(post_warmup) > 1:
                total_growth = post_warmup[-1] - post_warmup[0]
                steps = self.memory_history[-1]["step"] - self.memory_history[self.warmup_steps]["step"]
                print(f"  Post-warmup growth: {total_growth:>+8.1f} MB over {steps} steps")
                if steps > 0:
                    print(f"  Avg growth/step:    {total_growth / steps:>+8.2f} MB/step")

        print(f"{'=' * 70}\n")


class MemoryCleanupCallback(TrainerCallback):
    """Aggressively cleans up GPU memory to prevent fragmentation and leaks.

    This runs every N steps and does a full gc + cache clear cycle.
    """

    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class GenerationMonitorCallback(TrainerCallback):
    """Callback to monitor generation quality during training.

    Generates sample outputs every N steps to verify the model is learning
    the correct THOUGHT/TOOL/INPUT format.

    IMPORTANT: Does NOT store a reference to the model. Uses the model from
    the trainer's kwargs to avoid holding a stale pre-FSDP reference (which
    would prevent memory from being freed and cause a massive leak).
    """

    def __init__(
        self,
        tokenizer,
        interval: int = 500,
        num_samples: int = 3,
        prompts: list = None,
    ):
        """Initialize the callback.

        Args:
            tokenizer: Model tokenizer
            interval: Generate samples every N steps
            num_samples: Number of samples to generate per check
            prompts: List of test prompts (uses defaults if None)
        """
        self.tokenizer = tokenizer
        # NOTE: We intentionally do NOT store a model reference here.
        # The trainer passes the current (FSDP-wrapped) model via kwargs.
        self.interval = interval
        self.num_samples = num_samples
        self.prompts = prompts or [
            "What is the integral of x^2?",
            "Write a Python function to check if a number is prime.",
            "Explain the difference between TCP and UDP.",
        ]

    def on_step_end(self, args, state, control, **kwargs):
        """Check generation quality every interval steps."""
        if state.global_step > 0 and state.global_step % self.interval == 0:
            # Get the model from the trainer's kwargs -- this is the live
            # FSDP-wrapped model, not a stale pre-wrapping copy.
            model = kwargs.get("model")
            if model is None:
                print("[GenerationMonitor] WARNING: No model in kwargs, skipping generation check")
                return
            self._generate_samples(state.global_step, model)
            # Clean up after generation to avoid lingering tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_samples(self, step: int, model):
        """Generate and print sample outputs."""
        print(f"\n{'=' * 70}")
        print(f"[Generation Quality Check @ Step {step}]")
        print(f"{'=' * 70}")

        model.eval()

        try:
            device = next(model.parameters()).device
        except StopIteration:
            # FSDP may not expose parameters directly
            device = torch.device("cuda")

        for i, prompt in enumerate(self.prompts[: self.num_samples]):
            print(f"\n--- Sample {i + 1}: {prompt[:50]}... ---")

            # Format as chat message
            messages = [
                {"role": "system", "content": "You are an intelligent orchestrator that solves tasks by delegating to the most appropriate tools."},
                {"role": "user", "content": prompt},
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate -- wrapped in no_grad to avoid graph retention
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode and print
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                # Check for expected format markers
                has_thought = "THOUGHT:" in response.upper()
                has_tool = "TOOL:" in response.upper()
                has_input = "INPUT:" in response.upper()
                has_final = "FINAL_ANSWER:" in response.upper()

                format_check = []
                if has_thought:
                    format_check.append("THOUGHT")
                if has_tool:
                    format_check.append("TOOL")
                if has_input:
                    format_check.append("INPUT")
                if has_final:
                    format_check.append("FINAL_ANSWER")

                print(f"Format markers found: {format_check}")
                print(f"Response (first 500 chars):\n{response[:500]}")
            except Exception as e:
                print(f"  Generation failed (may be FSDP limitation): {e}")
            finally:
                # Explicitly delete intermediate tensors
                del inputs
                if "outputs" in dir():
                    del outputs

        model.train()
        print(f"\n{'=' * 70}\n")


def load_jsonl(path: str) -> list:
    """Load data from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def filter_successful_trajectories(input_path: str, output_path: str) -> int:
    """Filter to only successful trajectories.

    Args:
        input_path: Path to input JSONL with all trajectories
        output_path: Path to output JSONL with successful only

    Returns:
        Number of successful trajectories
    """
    count = 0
    with open(input_path, "r") as f, open(output_path, "w") as out:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("success", False):
                    out.write(line)
                    count += 1
    return count


def print_gpu_memory_summary():
    """Print a detailed GPU memory summary across all visible devices."""
    if not torch.cuda.is_available():
        return
    print(f"\n{'=' * 70}")
    print("[GPU Memory Summary at Launch]")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"  GPU {i}: {(total - free) / 1024**3:.1f} / {total / 1024**3:.1f} GB used "
            f"({(1 - free / total) * 100:.1f}%)"
        )
    print(f"{'=' * 70}\n")


def main():
    # Configuration
    model_name = "Qwen/Qwen3-8B"
    output_dir = "checkpoints/qwen3-8b-orchestrator-sft-256k-8h100"
    memory_snapshot_dir = "checkpoints/qwen3-8b-orchestrator-sft-256k-8h100/memory_snapshots"
    data_path = Path("/home/ubuntu/lambda-stanford/herumb/ipw_internal/intelligence-per-watt/src/orchestrator/data/filtered_traces/filtered_traces.parquet")

    # Parse env var to enable deep memory debugging (set MEMORY_DEBUG=1)
    deep_memory_debug = os.environ.get("MEMORY_DEBUG", "0") == "1"

    print("=" * 70)
    print("Full SFT Training for Orchestrator (256K context, 8x H100)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Data: {data_path}")
    print(f"Context: 256K tokens (262144)")
    print(f"GPUs: 8 (expected via accelerate)")
    print(f"Memory debug: {'ENABLED (snapshots will be saved)' if deep_memory_debug else 'disabled (set MEMORY_DEBUG=1 to enable snapshots)'}")
    print("=" * 70)

    print_gpu_memory_summary()

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run filter_dataset.py first.")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from parquet
    print(f"\nLoading dataset from {data_path}...")
    ds = Dataset.from_parquet(str(data_path))
    print(f"Loaded {len(ds)} samples")

    if len(ds) == 0:
        print("Error: No samples found!")
        sys.exit(1)

    # Create train/val split
    print("\nCreating 90/10 train/val split...")
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    def preprocess(example):
        """Convert trajectory to chat format."""
        # Support both 'conversations' and 'messages' keys
        messages = example.get("conversations") or example.get("messages")
        if messages is None:
            raise ValueError("Example must have 'conversations' or 'messages' key")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("\nPreprocessing datasets...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    # Filter out sequences that exceed context length to prevent OOM
    max_tokens = 262144
    print(f"\nFiltering sequences exceeding {max_tokens} tokens...")

    def get_token_length(example):
        return {"_num_tokens": len(tokenizer.encode(example["text"], add_special_tokens=False))}

    train_ds = train_ds.map(get_token_length, num_proc=8, desc="Counting tokens (train)")
    val_ds = val_ds.map(get_token_length, num_proc=8, desc="Counting tokens (val)")

    # Log token length distribution to understand memory pressure
    train_lengths = train_ds["_num_tokens"]
    print(f"\n[Data Stats] Token length distribution (train):")
    print(f"  Min: {min(train_lengths):,}, Max: {max(train_lengths):,}")
    print(f"  Mean: {statistics.mean(train_lengths):,.0f}, Median: {statistics.median(train_lengths):,.0f}")
    print(f"  Stdev: {statistics.stdev(train_lengths):,.0f}")
    # Print percentiles
    sorted_lengths = sorted(train_lengths)
    for pct in [50, 75, 90, 95, 99]:
        idx = int(len(sorted_lengths) * pct / 100)
        print(f"  P{pct}: {sorted_lengths[idx]:,} tokens")

    train_before = len(train_ds)
    val_before = len(val_ds)
    train_ds = train_ds.filter(lambda x: x["_num_tokens"] <= max_tokens, num_proc=8)
    val_ds = val_ds.filter(lambda x: x["_num_tokens"] <= max_tokens, num_proc=8)

    train_ds = train_ds.remove_columns(["_num_tokens"])
    val_ds = val_ds.remove_columns(["_num_tokens"])

    print(f"Train: {train_before} -> {len(train_ds)} ({train_before - len(train_ds)} removed)")
    print(f"Val: {val_before} -> {len(val_ds)} ({val_before - len(val_ds)} removed)")

    # Free intermediate dataset references
    del split, ds
    gc.collect()

    # Load model - full precision for full SFT
    print("\nLoading model (full bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # SDPA uses PyTorch's fused attention (Flash Attention on H100/A100) without
        # the separate flash-attn package, avoiding ABI mismatch with PyTorch.
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,  # Reduce peak CPU memory during loading
    )
    # Gradient checkpointing handled by SFTConfig (not FSDP's activation_checkpointing)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    param_memory_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"Model parameter memory: {param_memory_gb:.2f} GB (bf16)")
    print(f"Expected memory per GPU with FSDP FULL_SHARD (8 GPUs): ~{param_memory_gb / 8:.2f} GB params + optimizer states")

    # ---- CRITICAL: Patch forward to use chunked lm_head + CE loss ----
    # Without this, the full logits tensor (seq_len × 151936 × 4 bytes)
    # is 28 GB for 50K tokens in float32.  With chunking, peak is ~2.4 GB.
    patch_model_for_chunked_loss(model, chunk_size=4096)

    # Training configuration
    print("\nConfiguring training...")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        # Batch size settings for 256K context
        per_device_train_batch_size=1,  # CRITICAL: Must be 1 for 256K
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 1 * 8 * 16 = 128
        # Learning rate (lower for full SFT)
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # Context length
        max_length=262144,  # 256K context
        # Logging and saving
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        # Precision
        bf16=True,
        # Gradient checkpointing - ESSENTIAL for 256K context memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Gradient clipping for stability
        max_grad_norm=1.0,
        # Other settings
        report_to=["wandb"],
        run_name="qwen3-8b-orchestrator-sft-256k",
        dataset_text_field="text",
        packing=False,  # Don't pack - each trajectory is independent
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Dataloader settings to reduce memory
        dataloader_pin_memory=False,  # Avoid pinning with FSDP CPU offload
        dataloader_num_workers=2,  # Lower to reduce memory overhead
    )

    # Create callbacks
    # NOTE: GenerationMonitorCallback does NOT take a model reference --
    # it reads the live (FSDP-wrapped) model from kwargs at callback time.
    callbacks = [
        MemoryProfilerCallback(
            log_every=1,               # Log every step for diagnosis
            warn_growth_threshold_mb=50.0,
            warmup_steps=20,
            empty_cache_every=10,
            snapshot_dir=memory_snapshot_dir if deep_memory_debug else None,
        ),
        MemoryCleanupCallback(every_n_steps=50),
        GenerationMonitorCallback(
            tokenizer=tokenizer,
            interval=500,
            num_samples=3,
        ),
    ]

    # Create trainer
    print("\nInitializing trainer...")
    print(f"Effective batch size: 1 x 8 x 16 = 128")
    trainer = ChunkedLossSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # After trainer creation, the model has been wrapped by accelerate/FSDP.
    # Delete our local reference to avoid holding the pre-FSDP model.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Start training
    print("\nStarting training...")
    print("=" * 70)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config info
    config_info = {
        "base_model": model_name,
        "max_length": 262144,
        "training_epochs": 3,
        "effective_batch_size": 128,
    }
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"\nTraining complete! Model saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
