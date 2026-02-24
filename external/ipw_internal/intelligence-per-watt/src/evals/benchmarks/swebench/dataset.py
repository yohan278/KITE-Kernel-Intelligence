# benchmarks/swebench/dataset.py
"""
SWE-bench dataset loader.

Supports two dataset variants:
- verified: princeton-nlp/SWE-bench_Verified (500 tasks)
- verified_mini: MariusHobbhahn/swe-bench-verified-mini (50 tasks)

Note: This module provides raw data only. Prompt formatting (e.g., wrapping
problem_statement in <issue> tags) is handled by custom_runner.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional

from huggingface_hub import hf_hub_download

DATASET_PATHS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "verified_mini": "MariusHobbhahn/swe-bench-verified-mini",
}

DatasetVariant = Literal["verified", "verified_mini"]


@dataclass
class SWEBenchSample:
    """A single SWE-bench evaluation sample."""
    
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    version: str
    patch: str  # Ground truth patch
    test_patch: str
    created_at: str
    environment_setup_commit: str
    fail_to_pass: List[str]  # Tests that should go from fail to pass
    pass_to_pass: List[str]  # Tests that should remain passing
    difficulty: Optional[str] = None  # Only in verified dataset
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def repo_owner(self) -> str:
        """Extract repository owner from repo name."""
        return self.repo.split("/")[0] if "/" in self.repo else ""
    
    @property
    def repo_name(self) -> str:
        """Extract repository name from repo."""
        return self.repo.split("/")[1] if "/" in self.repo else self.repo


def _parse_test_list(value: str) -> List[str]:
    """Parse JSON-encoded test list from dataset."""
    import json
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return [value] if value else []


def load_swebench_samples(
    *,
    dataset: DatasetVariant = "verified_mini",
    split: str = "test",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    repos: Optional[List[str]] = None,
) -> Iterator[SWEBenchSample]:
    """
    Load SWE-bench samples from HuggingFace.
    
    Args:
        dataset: Dataset variant to load:
            - "verified": Full 500-task dataset (princeton-nlp/SWE-bench_Verified)
            - "verified_mini": 50-task subset (MariusHobbhahn/swe-bench-verified-mini)
        split: Dataset split to load (default: "test")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        repos: Optional list of repos to filter (e.g., ["django/django", "pytest-dev/pytest"])
    
    Yields:
        SWEBenchSample objects ready for evaluation
    """
    import random
    import pandas as pd
    
    dataset_path = DATASET_PATHS[dataset]
    
    # Download parquet file directly using huggingface_hub
    # This avoids the dill/pickle issues with datasets library on Python 3.14
    parquet_file = hf_hub_download(
        repo_id=dataset_path,
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    
    df = pd.read_parquet(parquet_file)
    rows = df.to_dict('records')
    
    if shuffle:
        random.seed(seed)
        random.shuffle(rows)
    
    if limit is not None:
        rows = rows[:limit]
    
    for row in rows:
        # Extract repo and filter if specified
        repo = row.get("repo", "")
        if repos and repo not in repos:
            continue
        
        # Parse JSON-encoded test lists
        fail_to_pass = _parse_test_list(row.get("FAIL_TO_PASS", ""))
        pass_to_pass = _parse_test_list(row.get("PASS_TO_PASS", ""))
        
        # Collect remaining fields as metadata
        known_fields = {
            "instance_id", "repo", "base_commit", "problem_statement",
            "hints_text", "version", "patch", "test_patch", "created_at",
            "environment_setup_commit", "FAIL_TO_PASS", "PASS_TO_PASS", "difficulty"
        }
        metadata = {k: v for k, v in row.items() if k not in known_fields}
        
        yield SWEBenchSample(
            instance_id=row.get("instance_id", ""),
            repo=repo,
            base_commit=row.get("base_commit", ""),
            problem_statement=row.get("problem_statement", ""),
            hints_text=row.get("hints_text", ""),
            version=row.get("version", ""),
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            created_at=row.get("created_at", ""),
            environment_setup_commit=row.get("environment_setup_commit", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            difficulty=row.get("difficulty"),  # Only in verified dataset
            metadata=metadata,
        )


def get_swebench_repos(dataset: DatasetVariant = "verified_mini") -> List[str]:
    """
    Get list of repositories in the SWE-bench dataset.
    
    Args:
        dataset: Dataset variant ("verified" or "verified_mini")
    
    Returns:
        List of unique repository names.
    """
    import pandas as pd
    
    dataset_path = DATASET_PATHS[dataset]
    parquet_file = hf_hub_download(
        repo_id=dataset_path,
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_file)
    return sorted(df["repo"].unique().tolist())


def get_sample_count(dataset: DatasetVariant = "verified_mini") -> int:
    """
    Get total number of samples in the dataset.
    
    Args:
        dataset: Dataset variant ("verified" or "verified_mini")
    
    Returns:
        Number of samples.
    """
    import pandas as pd
    
    dataset_path = DATASET_PATHS[dataset]
    parquet_file = hf_hub_download(
        repo_id=dataset_path,
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_file)
    return len(df)

