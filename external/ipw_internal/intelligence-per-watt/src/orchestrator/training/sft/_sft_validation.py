"""Validation logic for SFT trainer."""

from typing import Any, Dict, List

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def compute_validation_loss(
    model: Any,
    tokenizer: Any,
    val_traces: List[Dict[str, Any]],
    device: Any,
    max_seq_length: int = 4096,
    batch_size: int = 8,
) -> float:
    """Compute validation loss on held-out traces.
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        val_traces: Validation traces
        device: Device
        max_seq_length: Max sequence length
        batch_size: Batch size
        
    Returns:
        Average validation loss
    """
    if not HAS_TORCH or model is None:
        return 0.0
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(val_traces), batch_size):
            batch_traces = val_traces[i:i + batch_size]
            
            # Format and tokenize
            batch_examples = []
            for trace in batch_traces:
                text = _format_conversation(trace["conversations"], tokenizer)
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch_examples.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                })
            
            # Stack batch
            input_ids = torch.stack([ex["input_ids"] for ex in batch_examples]).to(device)
            attention_mask = torch.stack([ex["attention_mask"] for ex in batch_examples]).to(device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def _format_conversation(conversations: List[Dict[str, str]], tokenizer: Any) -> str:
    """Format conversation turns into training text."""
    parts = []
    for turn in conversations:
        role = turn.get("role", "")
        content = turn.get("content", "")
        
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        elif role == "tool":
            tool_name = turn.get("name", "unknown")
            parts.append(f"<|user|>\nTool '{tool_name}' returned: {content}")
    
    return "\n".join(parts) + tokenizer.eos_token
