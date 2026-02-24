# benchmarks/deepresearch/dataset.py
"""
DeepResearchBench dataset loader.

Loads the deep_research_bench dataset from GitHub for evaluating
research report generation with citations.

Reference: https://github.com/Ayanami0730/deep_research_bench
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# GitHub raw URLs for the dataset
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main"
QUERY_FILE_URL = f"{GITHUB_RAW_BASE}/data/prompt_data/query.jsonl"

# Local cache directory
CACHE_DIR = Path.home() / ".cache" / "ipw" / "deepresearch"


@dataclass
class DeepResearchSample:
    """A single DeepResearchBench sample."""

    task_id: str
    query: str                # Research question
    domain: str               # Physics, Chemistry, Finance, etc.
    language: str             # "en" or "zh"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_prompt(self, input_prompt: Optional[str] = None) -> str:
        """Get the formatted prompt for this sample."""
        prompt = input_prompt or DEFAULT_INPUT_PROMPT
        return prompt.format(
            query=self.query,
            domain=self.domain,
            language=self.language,
        )


@dataclass
class DeepResearchResult:
    """Result from a deep research task."""

    task_id: str
    query: str              # Original query
    article: str            # Generated research report with citations
    metadata: Dict[str, Any] = field(default_factory=dict)


def _download_dataset(force_download: bool = False) -> Path:
    """Download the dataset from GitHub if not cached.

    Args:
        force_download: Force re-download even if cached

    Returns:
        Path to the local query.jsonl file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / "query.jsonl"

    if local_path.exists() and not force_download:
        return local_path

    try:
        import urllib.request

        urllib.request.urlretrieve(QUERY_FILE_URL, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download DeepResearchBench dataset from {QUERY_FILE_URL}: {e}\n"
            "Please ensure you have internet access or manually download the dataset."
        ) from e


def load_deepresearch_samples(
    *,
    language: Optional[str] = None,
    domains: Optional[List[str]] = None,
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    force_download: bool = False,
) -> Iterator[DeepResearchSample]:
    """
    Load DeepResearchBench samples from the dataset.

    Args:
        language: Filter by language ("en" or "zh"), None for all
        domains: Optional list of domains to filter (e.g., ["Physics", "Chemistry"])
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        force_download: Force re-download of dataset

    Yields:
        DeepResearchSample objects ready for evaluation
    """
    local_path = _download_dataset(force_download=force_download)

    # Load all samples
    samples = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract fields with flexible key names
            task_id = data.get("task_id", data.get("id", f"task_{line_num}"))
            query = data.get("query", data.get("question", data.get("prompt", "")))
            domain = data.get("domain", data.get("category", "unknown"))
            sample_lang = data.get("language", data.get("lang", "en"))

            # Apply filters
            if language and sample_lang.lower() != language.lower():
                continue
            if domains and domain not in domains:
                continue

            # Collect remaining fields as metadata
            known_fields = {"task_id", "id", "query", "question", "prompt",
                           "domain", "category", "language", "lang"}
            metadata = {k: v for k, v in data.items() if k not in known_fields}

            samples.append(DeepResearchSample(
                task_id=str(task_id),
                query=query,
                domain=domain,
                language=sample_lang,
                metadata=metadata,
            ))

    # Shuffle if requested
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(samples)

    # Apply limit
    if limit is not None:
        samples = samples[:limit]

    yield from samples


def get_deepresearch_domains() -> List[str]:
    """Get list of domains in the DeepResearchBench dataset."""
    # Based on the paper, these are the 22 domains
    return [
        "Physics",
        "Chemistry",
        "Biology",
        "Medicine",
        "Computer Science",
        "Mathematics",
        "Engineering",
        "Earth Science",
        "Astronomy",
        "Psychology",
        "Sociology",
        "Economics",
        "Finance",
        "Law",
        "History",
        "Philosophy",
        "Art",
        "Music",
        "Literature",
        "Education",
        "Politics",
        "Environmental Science",
    ]


DEFAULT_INPUT_PROMPT = """You are a research assistant tasked with writing a comprehensive research report.

**Research Domain:** {domain}

**Research Question:** {query}

**Instructions:**
1. Write a detailed research report answering the question above.
2. Include proper citations using [1], [2], etc. format.
3. At the end, provide a "References" section listing all cited sources with URLs.
4. The report should be comprehensive, well-structured, and academically rigorous.
5. Focus on providing accurate, factual information with proper attribution.

Please write your research report below:"""
