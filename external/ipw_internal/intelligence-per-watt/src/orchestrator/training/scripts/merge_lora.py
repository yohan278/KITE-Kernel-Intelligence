#!/usr/bin/env python3
"""Merge LoRA adapter weights into base model for efficient serving."""

import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base", default="Qwen/Qwen3-8B", help="Base model name/path")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    
    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
