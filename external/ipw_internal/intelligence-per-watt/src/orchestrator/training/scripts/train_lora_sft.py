#!/usr/bin/env python3
"""LoRA SFT training script for orchestrator.

Trains Qwen3-8B with LoRA on successful trajectories.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/train_lora_sft.py
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def format_conversation(messages):
    """Format messages into a training string using chat template."""
    formatted = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == 'user':
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == 'tool':
            formatted += f"<|im_start|>tool\n{content}<|im_end|>\n"
    return formatted


def main():
    model_name = "Qwen/Qwen3-8B"
    output_dir = "checkpoints/qwen3-8b-orchestrator-lora"
    data_path = "data/sft/successful_trajectories"
    
    print("=" * 70)
    print("LoRA SFT Training for Orchestrator")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Data: {data_path}")
    print("=" * 70)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading dataset...")
    ds = load_from_disk(data_path)
    print(f"Loaded {len(ds)} samples")
    
    def preprocess(example):
        messages = example['conversations']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    print("\nPreprocessing dataset...")
    ds = ds.map(preprocess, remove_columns=ds.column_names)
    print(f"Sample text length: {len(ds[0]['text'])} chars")
    
    print("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\nConfiguring training...")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=32768,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )
    
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
