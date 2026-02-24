"""GAIA dataset loader.

Loads evaluation data from the GAIA benchmark (gaia-benchmark/GAIA).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class GAIASample:
    """Single GAIA evaluation sample."""

    task_id: str
    """Unique task identifier"""

    question: str
    """The question/task description"""

    answer: str
    """The expected answer (only available for validation split)"""

    level: int
    """Difficulty level (1, 2, or 3)"""

    file_name: Optional[str] = None
    """Associated file name (if any)"""

    file_path: Optional[str] = None
    """Path to associated file"""

    annotator_metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata from annotators"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def get_prompt(self) -> str:
        """Get the formatted prompt for this task."""
        prompt = self.question

        if self.file_name:
            prompt += f"\n\n[Note: This question references a file: {self.file_name}]"

        return prompt


class GAIADataset:
    """Dataset loader for GAIA benchmark.

    Example:
        dataset = GAIADataset(split="validation", limit=100, seed=42)
        for sample in dataset:
            print(sample.question)
    """

    HF_DATASET = "gaia-benchmark/GAIA"

    def __init__(
        self,
        split: str = "validation",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        levels: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize GAIA dataset.

        Args:
            split: Dataset split (validation or test - test has no answers)
            limit: Maximum number of samples
            seed: Random seed for sampling
            levels: Filter by difficulty levels (1, 2, 3)
            cache_dir: Cache directory for dataset
        """
        self.split = split
        self.limit = limit
        self.seed = seed
        self.levels = levels
        self.cache_dir = cache_dir

        self._samples: List[GAIASample] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from HuggingFace."""
        if not HAS_DATASETS:
            print("Warning: 'datasets' package not installed. Using mock data.")
            self._load_mock_data()
            return

        try:
            # GAIA requires access - try loading
            dataset = load_dataset(
                self.HF_DATASET,
                "2023_all",  # Use 2023 version
                split=self.split,
                cache_dir=self.cache_dir,
            )

            samples = []
            for i, item in enumerate(dataset):
                sample = GAIASample(
                    task_id=item.get("task_id", str(i)),
                    question=item.get("Question", ""),
                    answer=item.get("Final answer", ""),
                    level=int(item.get("Level", 1)),  # Ensure level is integer
                    file_name=item.get("file_name"),
                    file_path=item.get("file_path"),
                    annotator_metadata=item.get("Annotator Metadata", {}),
                    metadata={
                        "steps": item.get("Steps"),
                        "number_of_steps": item.get("Number of steps"),
                    },
                )
                samples.append(sample)

            # Filter by levels if specified
            if self.levels:
                samples = [s for s in samples if s.level in self.levels]

            # Random sampling with seed
            if self.seed is not None:
                random.seed(self.seed)
                random.shuffle(samples)

            # Apply limit
            if self.limit:
                samples = samples[:self.limit]

            self._samples = samples
            print(f"Loaded {len(self._samples)} samples from GAIA")

        except Exception as e:
            print(f"Warning: Could not load GAIA from Hub: {e}")
            print("Note: GAIA requires access approval on HuggingFace.")
            print("Falling back to mock data for development.")
            self._load_mock_data()

    def _load_mock_data(self):
        """Load mock data for development/testing."""
        mock_samples = [
            GAIASample(
                task_id="gaia_mock_001",
                question="What is the capital city of the country that hosted the 2024 Summer Olympics?",
                answer="Paris",
                level=1,
            ),
            GAIASample(
                task_id="gaia_mock_002",
                question="Calculate the sum of the first 100 prime numbers.",
                answer="24133",
                level=2,
            ),
            GAIASample(
                task_id="gaia_mock_003",
                question="What is the molecular weight of caffeine in g/mol?",
                answer="194.19",
                level=1,
            ),
            GAIASample(
                task_id="gaia_mock_004",
                question="Who directed the highest-grossing film of 2023?",
                answer="Christopher Nolan",
                level=1,
            ),
            GAIASample(
                task_id="gaia_mock_005",
                question="What is the population of Tokyo according to the most recent census?",
                answer="13.96 million",
                level=2,
            ),
        ]

        if self.levels:
            mock_samples = [s for s in mock_samples if s.level in self.levels]

        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(mock_samples)

        if self.limit:
            mock_samples = mock_samples[:self.limit]

        self._samples = mock_samples
        print(f"Loaded {len(self._samples)} mock samples for development")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[GAIASample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> GAIASample:
        return self._samples[idx]

    def get_by_id(self, task_id: str) -> Optional[GAIASample]:
        """Get sample by task ID."""
        for sample in self._samples:
            if sample.task_id == task_id:
                return sample
        return None
