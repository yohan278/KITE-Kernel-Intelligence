"""Profile causal-LM inference on Apple Silicon (M4) or CPU/CUDA.

Runs prefill + token-by-token decode under ``torch.profiler`` and
exports a Chrome trace plus a summary table.  No extra profiling
libraries required — only ``torch`` and ``transformers``.

Usage examples:

    # Default: tiny-gpt2 on MPS, 32 decode tokens
    python -m ipw.simulator.profile_lm --prompt "The quick brown fox"

    # Sync each step so trace aligns with real device work
    python -m ipw.simulator.profile_lm --prompt "Hello world" --sync-each-step

    # Full stack traces + MPS Instruments signposts
    python -m ipw.simulator.profile_lm --prompt "Energy" --with-stack \
        --mps-signposts --mps-wait-until-completed

    # Use a different model on CPU
    python -m ipw.simulator.profile_lm --model gpt2 --device cpu \
        --prompt "Once upon a time" --max-new-tokens 16
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def synchronize(device: str) -> None:
    """Barrier for async device work so profiler ranges are accurate."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def resolve_device(requested: str) -> torch.device:
    """Return a ``torch.device``, validating availability."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            sys.exit("ERROR: --device mps requested but MPS is not available on this system.")
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            sys.exit("ERROR: --device cuda requested but CUDA is not available on this system.")
        return torch.device("cuda")

    return torch.device("cpu")


def greedy_decode_step(model, input_ids, past_key_values):
    """Single greedy-decode forward; returns (next_token, new_past)."""
    out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    return next_token, out.past_key_values


# ------------------------------------------------------------------
# Warmup
# ------------------------------------------------------------------


def warmup(model, input_ids, device: str, n_iters: int = 2, decode_tokens: int = 8) -> None:
    """Run a few throw-away iterations to stabilise caches / JIT."""
    for _ in range(n_iters):
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(decode_tokens - 1):
            tok, past = greedy_decode_step(model, tok, past)
    synchronize(device)


# ------------------------------------------------------------------
# Profiled inference
# ------------------------------------------------------------------


def profiled_inference(
    model,
    input_ids,
    device: str,
    max_new_tokens: int,
    sync_each_step: bool,
    with_stack: bool,
    trace_path: str,
) -> None:
    """Run prefill + decode under ``torch.profiler`` and export trace."""
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        # -- Prefill --
        with record_function("prefill_forward"):
            out = model(input_ids=input_ids, use_cache=True)
            if sync_each_step:
                synchronize(device)

        past = out.past_key_values
        next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [next_token]

        # -- Decode loop --
        for step in range(max_new_tokens - 1):
            with record_function(f"decode_step_{step:03d}"):
                next_token, past = greedy_decode_step(model, next_token, past)
                if sync_each_step:
                    synchronize(device)
            generated.append(next_token)

    # Final sync before export
    synchronize(device)

    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to: {trace_path}")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=40,
        )
    )


# ------------------------------------------------------------------
# MPS signpost context manager
# ------------------------------------------------------------------


def mps_signpost_context(enabled: bool, wait_until_completed: bool):
    """Return the MPS Instruments signpost context (or a no-op).

    MPS signposts appear in Xcode Instruments under the *Metal System
    Trace* template.  They do **not** affect ``trace.json`` — the
    Chrome trace is driven entirely by ``torch.profiler``.
    """
    if not enabled:
        return contextlib.nullcontext()
    # torch.mps.profiler.profile is available from PyTorch 2.1+
    return torch.mps.profiler.profile(
        mode="interval",
        wait_until_completed=wait_until_completed,
    )


# ------------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Profile causal-LM inference (prefill + decode) with torch.profiler",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name or path")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to decode")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device (default: auto → mps if available else cpu)",
    )
    parser.add_argument("--trace", default="trace.json", help="Chrome trace output path")
    parser.add_argument("--with-stack", action="store_true", help="Capture Python call stacks")
    parser.add_argument(
        "--sync-each-step",
        action="store_true",
        help="Synchronize device after every forward pass",
    )
    parser.add_argument(
        "--mps-signposts",
        action="store_true",
        help="Enable MPS Instruments signposts (Xcode only, no effect on trace.json)",
    )
    parser.add_argument(
        "--mps-wait-until-completed",
        action="store_true",
        help="Block until each MPS command buffer completes (use with --mps-signposts)",
    )
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    device_str = str(device)
    print(f"Device : {device}")
    print(f"Model  : {args.model}")

    # -- Load model & tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"Prompt : {args.prompt!r}  ({input_ids.shape[1]} tokens)")
    print(f"Decode : {args.max_new_tokens} tokens, greedy")

    # -- Warmup --
    print("Warming up …")
    with torch.no_grad():
        warmup(model, input_ids, device_str)

    # -- Profile --
    print("Profiling …")
    with torch.no_grad(), mps_signpost_context(
        enabled=args.mps_signposts and device_str == "mps",
        wait_until_completed=args.mps_wait_until_completed,
    ):
        profiled_inference(
            model=model,
            input_ids=input_ids,
            device=device_str,
            max_new_tokens=args.max_new_tokens,
            sync_each_step=args.sync_each_step,
            with_stack=args.with_stack,
            trace_path=args.trace,
        )


if __name__ == "__main__":
    main()
