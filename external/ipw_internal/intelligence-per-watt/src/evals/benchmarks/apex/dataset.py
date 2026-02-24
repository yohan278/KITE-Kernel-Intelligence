# benchmarks/apex/dataset.py
"""
APEX-v1-extended dataset loader.
Loads the Mercor APEX dataset from HuggingFace for evaluating AI models
on economically valuable tasks across professional domains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset

HF_DATASET_PATH = "mercor/APEX-v1-extended"

# System prompt for response generation (from response_generation_prompt.txt)
RESPONSE_GENERATION_SYSTEM_PROMPT = """You are an AI assistant that produces final, domain-appropriate deliverables from a given task description and (optionally) attached files. You will be given the following inputs:
Inputs
• Task Domain: <Domain>  (e.g., "Operations")
• Task Prompt: <Prompt>
• Attachments:
  ==== Attached files content: ====
  === <File_1> ===
  <File_1_Contents>
  === <File_2> ===
  <File_2_Contents>
  … (repeat as needed)
Ground Rules
1) You must not ask follow-up questions. Interpret the prompt as best you can and produce the best complete answer given the information provided.
2) Use the attachments as primary sources.
3) Treat each "=== <File_Name> === …" block as the full content of that file.
All of the source files that you need have been added to the prompt."""


@dataclass
class APEXSample:
    """A single APEX evaluation sample."""
    
    task_id: str
    domain: str
    prompt: str
    attachments: Dict[str, str]  # {filename: content}
    rubric: Dict[str, Any]  # Grading rubric/criteria
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_prompt(self) -> str:
        """Format the prompt with domain and attachments for model input."""
        lines = [
            f"• Task Domain: {self.domain}",
            f"• Task Prompt: {self.prompt}",
        ]
        
        if self.attachments:
            lines.append("• Attachments:")
            lines.append("  ==== Attached files content: ====")
            for filename, content in self.attachments.items():
                lines.append(f"  === {filename} ===")
                lines.append(f"  {content}")
        
        return "\n".join(lines)
    
    def get_full_prompt(self) -> str:
        """Get the full prompt including system instructions."""
        return f"{RESPONSE_GENERATION_SYSTEM_PROMPT}\n\nInputs\n{self.format_prompt()}"


def _parse_rubric(rubric_data: Any) -> Dict[str, Any]:
    """Parse rubric from various formats into a standardized dict."""
    if isinstance(rubric_data, str):
        try:
            return json.loads(rubric_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, wrap it in a simple criterion
            return {"criterion_1": {"description": rubric_data, "weight": 1.0}}
    
    if isinstance(rubric_data, dict):
        return rubric_data
    
    if isinstance(rubric_data, list):
        # Convert list of criteria to dict
        result = {}
        for idx, criterion in enumerate(rubric_data, 1):
            if isinstance(criterion, dict):
                key = criterion.get("id", criterion.get("key", f"criterion_{idx}"))
                result[key] = criterion
            else:
                result[f"criterion_{idx}"] = {"description": str(criterion), "weight": 1.0}
        return result
    
    return {}


def _parse_attachments(row: Dict[str, Any]) -> Dict[str, str]:
    """Extract attachments from a dataset row."""
    attachments = {}
    
    # Try different possible column names for attachments
    attachment_columns = [
        "File Attachments",  # APEX dataset uses this
        "attachments",
        "files", 
        "source_documents",
        "documents",
        "reference_files",
    ]
    
    for col in attachment_columns:
        if col in row and row[col]:
            data = row[col]
            if isinstance(data, dict):
                attachments.update(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name", item.get("filename", f"file_{len(attachments)}"))
                        content = item.get("content", item.get("text", str(item)))
                        attachments[name] = content
                    elif isinstance(item, str):
                        attachments[f"file_{len(attachments)}"] = item
            elif isinstance(data, str):
                # APEX dataset stores file paths as newline-separated string
                # For now, store the paths as a reference (actual file content would need separate loading)
                file_paths = [p.strip() for p in data.strip().split('\n') if p.strip()]
                for path in file_paths:
                    filename = path.split('/')[-1] if '/' in path else path
                    attachments[filename] = f"[File reference: {path}]"
            break
    
    return attachments


def load_apex_samples(
    *,
    split: str = "train",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    domains: Optional[List[str]] = None,
) -> Iterator[APEXSample]:
    """
    Load APEX-v1-extended samples from HuggingFace.
    
    Args:
        split: Dataset split to load (default: "train")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        domains: Optional list of domains to filter (e.g., ["Law", "Medicine"])
    
    Yields:
        APEXSample objects ready for evaluation
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)
    
    if shuffle:
        ds = ds.shuffle(seed=seed)
    
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    
    for idx, row in enumerate(ds):
        # Extract task ID (APEX uses "Task ID" with space)
        task_id = row.get("Task ID", row.get("task_id", row.get("id", f"apex_{idx}")))
        
        # Extract domain (APEX uses "Domain" with capital D)
        domain = row.get("Domain", row.get("domain", row.get("category", "General")))
        
        # Filter by domain if specified
        if domains and domain not in domains:
            continue
        
        # Extract prompt (APEX uses "Prompt" with capital P)
        prompt = row.get("Prompt", row.get("prompt", row.get("task", row.get("question", ""))))
        
        # Extract attachments
        attachments = _parse_attachments(row)
        
        # Extract and parse rubric (APEX uses "Rubric JSON")
        rubric_raw = row.get("Rubric JSON", row.get("rubric", row.get("grading_rubric", row.get("criteria", {}))))
        rubric = _parse_rubric(rubric_raw)
        
        # Collect metadata
        metadata = {
            k: v for k, v in row.items()
            if k not in ["Task ID", "task_id", "id", "Domain", "domain", "category", 
                        "Prompt", "prompt", "task", "question", 
                        "Rubric JSON", "rubric", "grading_rubric", "criteria",
                        "File Attachments", "attachments", "files", "source_documents", "documents"]
        }
        
        yield APEXSample(
            task_id=str(task_id),
            domain=domain,
            prompt=prompt,
            attachments=attachments,
            rubric=rubric,
            metadata=metadata,
        )


def get_apex_domains() -> List[str]:
    """Get list of available domains in the APEX dataset."""
    return [
        "Finance",
        "Consulting", 
        "Legal",
        "Medicine",
    ]

