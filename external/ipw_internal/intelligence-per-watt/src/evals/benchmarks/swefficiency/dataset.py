# benchmarks/swefficiency/dataset.py
"""
SWEfficiency dataset loader.

Loads the swefficiency/swefficiency dataset from HuggingFace for evaluating
software performance optimization (SWE-bench style).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset

HF_DATASET_PATH = "swefficiency/swefficiency"


@dataclass
class SWEfficiencySample:
    """A single SWEfficiency benchmark sample."""

    instance_id: str
    repo: str
    base_commit: str
    patch: str              # Ground truth optimization patch
    test_patch: str
    problem_statement: str
    workload: str           # Description of the workload
    speedup: float          # Expected speedup from optimization
    test_cmd: str           # Command to run tests
    rebuild_cmd: str        # Command to rebuild after patch
    image_name: str         # Docker image for execution
    covering_tests: List[str]  # Tests that verify correctness
    pass_to_pass: List[str]    # Tests that should remain passing
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def repo_owner(self) -> str:
        """Extract repository owner from repo name."""
        return self.repo.split("/")[0] if "/" in self.repo else ""

    @property
    def repo_name(self) -> str:
        """Extract repository name from repo."""
        return self.repo.split("/")[1] if "/" in self.repo else self.repo

    def get_prompt(self, input_prompt: Optional[str] = None) -> str:
        """Get the formatted prompt for this sample."""
        prompt = input_prompt or DEFAULT_INPUT_PROMPT

        return prompt.format(
            repo=self.repo,
            problem_statement=self.problem_statement,
            workload=self.workload,
            expected_speedup=self.speedup,
        )


def _parse_test_list(value: Any) -> List[str]:
    """Parse test list from various formats."""
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value] if value else []
    return []


def load_swefficiency_samples(
    *,
    split: str = "test",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    repos: Optional[List[str]] = None,
    min_speedup: Optional[float] = None,
    max_speedup: Optional[float] = None,
) -> Iterator[SWEfficiencySample]:
    """
    Load SWEfficiency samples from HuggingFace.

    Args:
        split: Dataset split to load (default: "test")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        repos: Optional list of repos to filter (e.g., ["numpy/numpy"])
        min_speedup: Minimum expected speedup to include
        max_speedup: Maximum expected speedup to include

    Yields:
        SWEfficiencySample objects ready for evaluation
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for row in ds:
        # Extract repo and filter if specified
        repo = row.get("repo", "")
        if repos and repo not in repos:
            continue

        # Extract speedup and filter if specified
        speedup = float(row.get("speedup", row.get("expected_speedup", 1.0)))
        if min_speedup is not None and speedup < min_speedup:
            continue
        if max_speedup is not None and speedup > max_speedup:
            continue

        # Parse test lists
        covering_tests = _parse_test_list(
            row.get("covering_tests", row.get("COVERING_TESTS", []))
        )
        pass_to_pass = _parse_test_list(
            row.get("pass_to_pass", row.get("PASS_TO_PASS", []))
        )

        # Collect remaining fields as metadata
        known_fields = {
            "instance_id", "repo", "base_commit", "patch", "test_patch",
            "problem_statement", "workload", "speedup", "expected_speedup",
            "test_cmd", "rebuild_cmd", "image_name",
            "covering_tests", "COVERING_TESTS", "pass_to_pass", "PASS_TO_PASS"
        }
        metadata = {k: v for k, v in row.items() if k not in known_fields}

        yield SWEfficiencySample(
            instance_id=row.get("instance_id", ""),
            repo=repo,
            base_commit=row.get("base_commit", ""),
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            problem_statement=row.get("problem_statement", ""),
            workload=row.get("workload", ""),
            speedup=speedup,
            test_cmd=row.get("test_cmd", ""),
            rebuild_cmd=row.get("rebuild_cmd", ""),
            image_name=row.get("image_name", ""),
            covering_tests=covering_tests,
            pass_to_pass=pass_to_pass,
            metadata=metadata,
        )


def get_swefficiency_repos() -> List[str]:
    """
    Get list of repositories in the SWEfficiency dataset.

    Returns:
        List of unique repository names.
    """
    ds = load_dataset(HF_DATASET_PATH, split="test", streaming=False)
    repos = set()
    for row in ds:
        repo = row.get("repo", "")
        if repo:
            repos.add(repo)
    return sorted(repos)


def get_sample_count(split: str = "test") -> int:
    """
    Get total number of samples in the dataset.

    Args:
        split: Dataset split

    Returns:
        Number of samples.
    """
    ds = load_dataset(HF_DATASET_PATH, split=split, streaming=False)
    return len(ds)


DEFAULT_INPUT_PROMPT = """You are a software performance engineer. Your task is to optimize the code in the repository to improve performance.

Repository: {repo}

## Problem Statement
{problem_statement}

## Workload Description
{workload}

## Expected Speedup
The optimization should achieve approximately {expected_speedup:.1f}x speedup.

## Instructions
1. Analyze the codebase to identify performance bottlenecks
2. Implement optimizations that improve performance
3. Ensure all existing tests still pass
4. Generate a git patch with your changes

Please provide your optimization patch in unified diff format."""
