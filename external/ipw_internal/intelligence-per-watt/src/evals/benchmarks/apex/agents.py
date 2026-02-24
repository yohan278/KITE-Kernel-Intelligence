# benchmarks/apex/agents.py
"""
APEX-Agents dataset loader.

Loads the mercor/apex-agents dataset from HuggingFace for evaluating
agentic professional tasks across Investment Banking, Law, and Management Consulting.

This is distinct from APEX-v1-extended (mercor/APEX-v1-extended) which focuses on
single-turn professional tasks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset

HF_DATASET_PATH = "mercor/apex-agents"

# Job categories in APEX-Agents
APEX_AGENTS_JOB_CATEGORIES = [
    "investment_banking",
    "law",
    "management_consulting",
]

# System prompt for APEX-Agents tasks
APEX_AGENTS_SYSTEM_PROMPT = """You are an AI assistant completing professional agentic tasks.

You will be given a task in a specific professional domain (Investment Banking, Law, or Management Consulting). Your goal is to produce a high-quality output that meets all evaluation criteria.

Instructions:
1. Read the task carefully and understand what is being asked.
2. Use the provided context files if available.
3. Produce a complete, professional-quality response.
4. Do not ask follow-up questions - work with the information provided.
5. Format your response appropriately for the expected output type.

Your response will be evaluated against binary criteria (Met/Not Met) by an expert judge."""


@dataclass
class APEXAgentsSample:
    """A single APEX-Agents benchmark sample."""

    task_id: str
    prompt: str                    # Single-turn agent instruction
    rubric: Dict[str, Any]         # Binary criteria dictionary
    gold_outputs: List[str]        # Expert reference outputs
    job_category: str              # investment_banking, law, consulting
    workflow_tags: List[str]       # Task type tags
    output_type: str               # Expected output type
    estimated_hours: float         # Estimated time for human
    world_files: List[str]         # Context file pointers
    world_id: str                  # Identifier for the world/context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_prompt(self) -> str:
        """Format the prompt with context files for model input."""
        lines = [
            f"Job Category: {self.job_category}",
            f"Task: {self.prompt}",
        ]

        if self.world_files:
            lines.append("\nAvailable context files:")
            for f in self.world_files:
                lines.append(f"  - {f}")

        return "\n".join(lines)

    def get_full_prompt(self) -> str:
        """Get the full prompt including system instructions."""
        return f"{APEX_AGENTS_SYSTEM_PROMPT}\n\n{self.format_prompt()}"

    @property
    def num_criteria(self) -> int:
        """Get number of evaluation criteria in rubric."""
        return len(self.rubric)


def _parse_rubric(rubric_data: Any) -> Dict[str, Any]:
    """Parse rubric from various formats into a standardized dict."""
    if isinstance(rubric_data, str):
        try:
            return json.loads(rubric_data)
        except json.JSONDecodeError:
            return {"criterion_1": {"description": rubric_data, "weight": 1.0}}

    if isinstance(rubric_data, dict):
        return rubric_data

    if isinstance(rubric_data, list):
        result = {}
        for idx, criterion in enumerate(rubric_data, 1):
            if isinstance(criterion, dict):
                key = criterion.get("id", criterion.get("key", f"criterion_{idx}"))
                result[key] = criterion
            else:
                result[f"criterion_{idx}"] = {"description": str(criterion), "weight": 1.0}
        return result

    return {}


def _parse_list_field(value: Any) -> List[str]:
    """Parse a field that should be a list of strings."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            return [value]
    return []


def load_apex_agents_samples(
    *,
    split: str = "train",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    job_categories: Optional[List[str]] = None,
    min_criteria: Optional[int] = None,
    max_criteria: Optional[int] = None,
    world_ids: Optional[List[str]] = None,
) -> Iterator[APEXAgentsSample]:
    """
    Load APEX-Agents samples from HuggingFace.

    Args:
        split: Dataset split to load (default: "train")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        job_categories: Optional list of job categories to filter
                       (e.g., ["investment_banking", "law"])
        min_criteria: Minimum number of rubric criteria
        max_criteria: Maximum number of rubric criteria
        world_ids: Optional list of world IDs to filter

    Yields:
        APEXAgentsSample objects ready for evaluation
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for idx, row in enumerate(ds):
        # Extract task ID
        task_id = row.get("task_id", row.get("id", f"apex_agents_{idx}"))

        # Extract job category and filter if specified
        job_category = row.get(
            "job_category",
            row.get("category", row.get("domain", "general"))
        )
        if job_categories and job_category not in job_categories:
            continue

        # Extract world_id and filter if specified
        world_id = row.get("world_id", row.get("world", ""))
        if world_ids and world_id not in world_ids:
            continue

        # Extract and parse rubric
        rubric_raw = row.get(
            "rubric",
            row.get("criteria", row.get("grading_rubric", {}))
        )
        rubric = _parse_rubric(rubric_raw)

        # Filter by criteria count if specified
        num_criteria = len(rubric)
        if min_criteria is not None and num_criteria < min_criteria:
            continue
        if max_criteria is not None and num_criteria > max_criteria:
            continue

        # Extract prompt
        prompt = row.get("prompt", row.get("task", row.get("instruction", "")))

        # Extract gold outputs
        gold_outputs = _parse_list_field(
            row.get("gold_outputs", row.get("reference_outputs", []))
        )

        # Extract workflow tags
        workflow_tags = _parse_list_field(
            row.get("workflow_tags", row.get("tags", []))
        )

        # Extract world files
        world_files = _parse_list_field(
            row.get("world_files", row.get("context_files", row.get("files", [])))
        )

        # Extract other fields
        output_type = row.get("output_type", row.get("expected_output", "text"))
        estimated_hours = float(row.get("estimated_hours", row.get("time_estimate", 0.0)))

        # Collect remaining fields as metadata
        known_fields = {
            "task_id", "id", "prompt", "task", "instruction",
            "rubric", "criteria", "grading_rubric",
            "gold_outputs", "reference_outputs",
            "job_category", "category", "domain",
            "workflow_tags", "tags",
            "output_type", "expected_output",
            "estimated_hours", "time_estimate",
            "world_files", "context_files", "files",
            "world_id", "world",
        }
        metadata = {k: v for k, v in row.items() if k not in known_fields}

        yield APEXAgentsSample(
            task_id=str(task_id),
            prompt=prompt,
            rubric=rubric,
            gold_outputs=gold_outputs,
            job_category=job_category,
            workflow_tags=workflow_tags,
            output_type=output_type,
            estimated_hours=estimated_hours,
            world_files=world_files,
            world_id=world_id,
            metadata=metadata,
        )


def get_apex_agents_job_categories() -> List[str]:
    """Get list of job categories in the APEX-Agents dataset."""
    return APEX_AGENTS_JOB_CATEGORIES.copy()


def get_apex_agents_worlds() -> List[str]:
    """
    Get list of unique world IDs in the APEX-Agents dataset.

    Returns:
        List of unique world identifiers.
    """
    ds = load_dataset(HF_DATASET_PATH, split="train", streaming=False)
    worlds = set()
    for row in ds:
        world_id = row.get("world_id", row.get("world", ""))
        if world_id:
            worlds.add(world_id)
    return sorted(worlds)


def get_apex_agents_sample_count(split: str = "train") -> int:
    """
    Get total number of samples in the dataset.

    Args:
        split: Dataset split

    Returns:
        Number of samples.
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)
    return len(ds)
