"""Humanity's Last Exam (HLE) dataset loader.

Loads evaluation data from the HLE benchmark (cais/hle).
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
class HLESample:
    """Single HLE evaluation sample."""

    question_id: str
    """Unique question identifier"""

    question: str
    """The question text"""

    answer: str
    """The correct answer"""

    subject: str
    """Subject area (e.g., math, physics, chemistry)"""

    question_type: str
    """Type: multiple_choice or short_answer"""

    choices: Optional[List[str]] = None
    """Multiple choice options (if applicable)"""

    image: Optional[Any] = None
    """Image data (for multimodal questions)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def get_prompt(self) -> str:
        """Get the formatted prompt for this question."""
        prompt = self.question

        if self.choices:
            prompt += "\n\nOptions:\n"
            for i, choice in enumerate(self.choices):
                letter = chr(ord('A') + i)
                prompt += f"{letter}. {choice}\n"
            prompt += "\nAnswer with the letter of the correct option."
        else:
            prompt += "\n\nProvide a concise answer."

        return prompt


class HLEDataset:
    """Dataset loader for Humanity's Last Exam.

    Example:
        dataset = HLEDataset(split="test", limit=100, seed=42)
        for sample in dataset:
            print(sample.question)
    """

    HF_DATASET = "cais/hle"

    def __init__(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize HLE dataset.

        Args:
            split: Dataset split (test)
            limit: Maximum number of samples
            seed: Random seed for sampling
            subjects: Filter by specific subjects
            cache_dir: Cache directory for dataset
        """
        self.split = split
        self.limit = limit
        self.seed = seed
        self.subjects = subjects
        self.cache_dir = cache_dir

        self._samples: List[HLESample] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from HuggingFace."""
        if not HAS_DATASETS:
            print("Warning: 'datasets' package not installed. Using mock data.")
            self._load_mock_data()
            return

        try:
            dataset = load_dataset(
                self.HF_DATASET,
                split=self.split,
                cache_dir=self.cache_dir,
            )

            samples = []
            for i, item in enumerate(dataset):
                # Extract question type
                question_type = "short_answer"
                choices = None
                answer = item.get("answer", "")
                
                # Check multiple ways to detect MC questions
                if item.get("answer_type") == "multiple_choice" or item.get("choices"):
                    question_type = "multiple_choice"
                    choices = item.get("choices", [])
                # Also check if answer is a single letter (A-I) - likely MC
                elif len(answer.strip()) == 1 and answer.strip().upper() in "ABCDEFGHI":
                    question_type = "multiple_choice"

                sample = HLESample(
                    question_id=item.get("id", str(i)),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    subject=item.get("subject", "general"),
                    question_type=question_type,
                    choices=choices,
                    image=item.get("image"),
                    metadata={
                        "difficulty": item.get("difficulty"),
                        "source": item.get("source"),
                        "author": item.get("author"),
                    },
                )
                samples.append(sample)

            # Filter by subjects if specified
            if self.subjects:
                samples = [s for s in samples if s.subject.lower() in [sub.lower() for sub in self.subjects]]

            # Random sampling with seed
            if self.seed is not None:
                random.seed(self.seed)
                random.shuffle(samples)

            # Apply limit
            if self.limit:
                samples = samples[:self.limit]

            self._samples = samples
            print(f"Loaded {len(self._samples)} samples from HLE")

        except Exception as e:
            print(f"Warning: Could not load HLE from Hub: {e}")
            print("Falling back to mock data for development.")
            self._load_mock_data()

    def _load_mock_data(self):
        """Load mock data for development/testing."""
        mock_samples = [
            HLESample(
                question_id="hle_mock_001",
                question="What is the primary function of mitochondria in eukaryotic cells?",
                answer="ATP production through cellular respiration",
                subject="biology",
                question_type="short_answer",
            ),
            HLESample(
                question_id="hle_mock_002",
                question="Which of the following is NOT a property of a continuous function?",
                answer="B",
                subject="mathematics",
                question_type="multiple_choice",
                choices=[
                    "The intermediate value property",
                    "Can have jump discontinuities",
                    "Preserves limits",
                    "Is bounded on closed intervals"
                ],
            ),
            HLESample(
                question_id="hle_mock_003",
                question="In quantum mechanics, what does the Heisenberg uncertainty principle state?",
                answer="The position and momentum of a particle cannot both be precisely determined simultaneously",
                subject="physics",
                question_type="short_answer",
            ),
        ]

        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(mock_samples)

        if self.limit:
            mock_samples = mock_samples[:self.limit]

        self._samples = mock_samples
        print(f"Loaded {len(self._samples)} mock samples for development")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[HLESample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> HLESample:
        return self._samples[idx]

    def get_by_id(self, question_id: str) -> Optional[HLESample]:
        """Get sample by question ID."""
        for sample in self._samples:
            if sample.question_id == question_id:
                return sample
        return None
