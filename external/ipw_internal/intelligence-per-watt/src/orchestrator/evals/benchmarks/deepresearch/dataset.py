"""DeepResearch-Bench dataset loader.

Loads evaluation data from the DeepResearch-Bench benchmark:
- Research prompts and metadata from query.jsonl (GitHub repo)
- Per-task evaluation criteria from criteria.jsonl (GitHub repo)
- Reference articles from openai-deepresearch.jsonl (HuggingFace)

Follows the original implementation at:
https://github.com/Ayanami0730/deep_research_bench
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Raw GitHub URLs for benchmark data files
QUERY_URL = "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/prompt_data/query.jsonl"
CRITERIA_URL = "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/criteria_data/criteria.jsonl"


@dataclass
class DeepResearchSample:
    """Single DeepResearch-Bench evaluation sample."""

    sample_id: str
    """Unique sample identifier"""

    prompt: str
    """The research prompt/question"""

    reference_article: Optional[str] = None
    """Reference article for comparison (from OpenAI Deep Research)"""

    criteria: Optional[Dict[str, Any]] = None
    """Per-task evaluation criteria with dimension weights and criterion weights"""

    category: str = ""
    """Task category/topic"""

    language: str = "en"
    """Language of the prompt (en or zh)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def get_prompt(self) -> str:
        """Get the formatted prompt for this research task."""
        return (
            f"You are a deep research assistant. Please conduct thorough research "
            f"and write a comprehensive, well-structured report on the following topic:\n\n"
            f"{self.prompt}\n\n"
            f"Your report should be detailed, factually accurate, well-organized with clear sections, "
            f"and supported by evidence. Aim for a comprehensive analysis."
        )


def _download_jsonl(url: str, cache_path: Path) -> List[Dict]:
    """Download a JSONL file from URL, caching locally."""
    if cache_path.exists():
        items = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, str(cache_path))

    items = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"  Downloaded {len(items)} items -> {cache_path}")
    return items


class DeepResearchDataset:
    """Dataset loader for DeepResearch-Bench.

    Loads and merges three data sources (matched by prompt text):
    1. query.jsonl    -- 100 research prompts with id/topic/language
    2. criteria.jsonl  -- per-task evaluation criteria with dimension + criterion weights
    3. openai-deepresearch.jsonl -- reference articles (from HuggingFace)

    Example:
        dataset = DeepResearchDataset(limit=50, seed=42)
        for sample in dataset:
            print(sample.prompt, sample.criteria is not None)
    """

    HF_DATASET = "muset-ai/DeepResearch-Bench-Dataset"

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize DeepResearch-Bench dataset.

        Args:
            split: Dataset split (unused, benchmark has single split)
            limit: Maximum number of samples
            seed: Random seed for sampling
            language: Filter by language ('en' or 'zh')
            cache_dir: Cache directory for downloaded files
        """
        self.split = split
        self.limit = limit
        self.seed = seed
        self.language = language

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "deepresearch_bench"

        self._samples: List[DeepResearchSample] = []
        self.references_precleaned: bool = False
        """True if reference articles were loaded from pre-cleaned cache."""
        self._load_dataset()

    def _load_dataset(self):
        """Load and merge data from all three sources."""
        # 1. Download query.jsonl and criteria.jsonl from GitHub
        query_cache = self.cache_dir / "query.jsonl"
        criteria_cache = self.cache_dir / "criteria.jsonl"

        try:
            queries = _download_jsonl(QUERY_URL, query_cache)
            criteria_list = _download_jsonl(CRITERIA_URL, criteria_cache)
        except Exception as e:
            print(f"Warning: Could not download benchmark data from GitHub: {e}")
            print("Falling back to mock data for development.")
            self._load_mock_data()
            return

        # Build lookup maps keyed by prompt text (this is how the original matches)
        query_by_prompt = {item["prompt"]: item for item in queries}
        criteria_by_prompt = {item["prompt"]: item for item in criteria_list}

        # 2. Load reference articles — prefer pre-cleaned cache, otherwise
        #    download from HuggingFace, auto-clean via Gemini, and cache for future runs
        reference_by_prompt: Dict[str, str] = {}
        from .clean_articles import load_cleaned_references, clean_reference_articles
        cleaned_refs = load_cleaned_references(self.cache_dir)
        if cleaned_refs:
            reference_by_prompt = cleaned_refs
            self.references_precleaned = True
            print(f"Loaded {len(reference_by_prompt)} pre-cleaned reference articles from cache")
        elif HAS_DATASETS:
            try:
                print("No pre-cleaned reference articles found. "
                      "Downloading and cleaning references (one-time setup)...")
                clean_reference_articles(cache_dir=self.cache_dir, max_workers=3)
                cleaned_refs = load_cleaned_references(self.cache_dir)
                if cleaned_refs:
                    reference_by_prompt = cleaned_refs
                    self.references_precleaned = True
                    print(f"Cleaned and cached {len(reference_by_prompt)} reference articles")
                else:
                    print("Warning: Cleaning completed but no references were produced. "
                          "Falling back to raw references.")
                    dataset = load_dataset(
                        self.HF_DATASET,
                        data_files="generated_reports/openai-deepresearch.jsonl",
                        split="train",
                        cache_dir=str(self.cache_dir),
                    )
                    for item in dataset:
                        prompt = item.get("prompt", "")
                        article = item.get("article", "")
                        if prompt and article:
                            reference_by_prompt[prompt] = article
                    print(f"Loaded {len(reference_by_prompt)} raw reference articles from HuggingFace")
            except Exception as e:
                print(f"Warning: Could not load/clean reference articles: {e}")
        else:
            print("Warning: 'datasets' package not installed. Reference articles unavailable.")

        # 3. Merge into samples using query.jsonl as the primary source
        samples = []
        for query_item in queries:
            prompt = query_item.get("prompt", "")
            if not prompt:
                continue

            sample_id = str(query_item.get("id", ""))
            lang = query_item.get("language", "en")
            topic = query_item.get("topic", "")

            # Get matching criteria and reference
            criteria = criteria_by_prompt.get(prompt)
            reference = reference_by_prompt.get(prompt)

            sample = DeepResearchSample(
                sample_id=sample_id,
                prompt=prompt,
                reference_article=reference,
                criteria=criteria,
                category=topic,
                language=lang,
                metadata={"topic": topic},
            )
            samples.append(sample)

        # Filter by language if specified
        if self.language:
            samples = [s for s in samples if s.language == self.language]

        # Random sampling with seed
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(samples)

        # Apply limit
        if self.limit:
            samples = samples[:self.limit]

        self._samples = samples

        # Report stats
        with_criteria = sum(1 for s in self._samples if s.criteria is not None)
        with_reference = sum(1 for s in self._samples if s.reference_article is not None)
        print(f"Loaded {len(self._samples)} samples from DeepResearch-Bench "
              f"({with_criteria} with criteria, {with_reference} with reference articles)")

    def _load_mock_data(self):
        """Load mock data for development/testing."""
        mock_samples = [
            DeepResearchSample(
                sample_id="dr_mock_001",
                prompt="Analyze the current state of quantum computing and its potential impact on cryptography.",
                category="Science & Technology",
                language="en",
            ),
            DeepResearchSample(
                sample_id="dr_mock_002",
                prompt="Provide a comprehensive analysis of the global rare earth elements supply chain.",
                category="Geopolitics & Economics",
                language="en",
            ),
        ]

        if self.language:
            mock_samples = [s for s in mock_samples if s.language == self.language]
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(mock_samples)
        if self.limit:
            mock_samples = mock_samples[:self.limit]

        self._samples = mock_samples
        print(f"Loaded {len(self._samples)} mock samples for development")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[DeepResearchSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> DeepResearchSample:
        return self._samples[idx]

    def get_by_id(self, sample_id: str) -> Optional[DeepResearchSample]:
        """Get sample by ID."""
        for sample in self._samples:
            if sample.sample_id == sample_id:
                return sample
        return None
