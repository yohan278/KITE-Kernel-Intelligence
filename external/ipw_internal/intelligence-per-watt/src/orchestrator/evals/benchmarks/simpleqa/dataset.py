"""SimpleQA Verified dataset loader.

Loads evaluation data from the SimpleQA Verified benchmark (google/simpleqa-verified).
A 1,000-prompt factuality benchmark from Google DeepMind and Google Research.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class SimpleQASample:
    """Single SimpleQA Verified evaluation sample."""

    original_index: int
    """Index from the original SimpleQA benchmark"""

    problem: str
    """The question/problem prompt"""

    answer: str
    """The gold answer"""

    topic: str
    """Topic category (e.g., Politics, Art, Science and technology, etc.)"""

    answer_type: str
    """Answer type (Number, Person, Date, Place, Other)"""

    multi_step: bool
    """Whether the question requires information from multiple sources"""

    requires_reasoning: bool
    """Whether the question requires more complex reasoning"""

    urls: str
    """Comma-separated list of URLs supporting the gold answer"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def get_prompt(self) -> str:
        """Get the formatted prompt for this question."""
        return self.problem


class SimpleQADataset:
    """Dataset loader for SimpleQA Verified benchmark.

    Example:
        dataset = SimpleQADataset(limit=100, seed=42)
        for sample in dataset:
            print(sample.problem)
    """

    HF_DATASET = "google/simpleqa-verified"

    def __init__(
        self,
        split: str = "eval",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        topics: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize SimpleQA Verified dataset.

        Args:
            split: Dataset split (eval)
            limit: Maximum number of samples
            seed: Random seed for sampling
            topics: Filter by specific topics
            cache_dir: Cache directory for dataset
        """
        self.split = split
        self.limit = limit
        self.seed = seed
        self.topics = topics
        self.cache_dir = cache_dir

        self._samples: List[SimpleQASample] = []
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
                sample = SimpleQASample(
                    original_index=item.get("original_index", i),
                    problem=item.get("problem", ""),
                    answer=item.get("answer", ""),
                    topic=item.get("topic", ""),
                    answer_type=item.get("answer_type", "Other"),
                    multi_step=bool(item.get("multi_step", False)),
                    requires_reasoning=bool(item.get("requires_reasoning", False)),
                    urls=item.get("urls", ""),
                    metadata={
                        "index": i,
                    },
                )
                samples.append(sample)

            # Filter by topics if specified
            if self.topics:
                topics_lower = [t.lower() for t in self.topics]
                samples = [s for s in samples if s.topic.lower() in topics_lower]

            # Random sampling with seed
            if self.seed is not None:
                random.seed(self.seed)
                random.shuffle(samples)

            # Apply limit
            if self.limit:
                samples = samples[:self.limit]

            self._samples = samples
            print(f"Loaded {len(self._samples)} samples from SimpleQA Verified")

        except Exception as e:
            print(f"Warning: Could not load SimpleQA Verified from Hub: {e}")
            print("Falling back to mock data for development.")
            self._load_mock_data()

    def _load_mock_data(self):
        """Load mock data for development/testing."""
        mock_samples = [
            SimpleQASample(
                original_index=5,
                problem="How much money, in euros, was the surgeon held responsible for Stella Obasanjo's death ordered to pay her son?",
                answer="120,000 euros",
                topic="Politics",
                answer_type="Number",
                multi_step=True,
                requires_reasoning=False,
                urls="https://en.wikipedia.org/wiki/Stella_Obasanjo",
            ),
            SimpleQASample(
                original_index=13,
                problem="In which year did Melbourne's Monash Gallery of Art (MGA) rebrand and become the Museum of Australian Photography (MAPh)?",
                answer="2023",
                topic="Art",
                answer_type="Date",
                multi_step=False,
                requires_reasoning=False,
                urls="https://maph.org.au/about/",
            ),
            SimpleQASample(
                original_index=32,
                problem="What day, month, and year was Carrie Underwood's album 'Cry Pretty' certified Gold by the RIAA?",
                answer="Oct 23, 2018",
                topic="Music",
                answer_type="Date",
                multi_step=False,
                requires_reasoning=False,
                urls="https://en.wikipedia.org/wiki/Cry_Pretty",
            ),
            SimpleQASample(
                original_index=397,
                problem="Which city in Nepal is known as the 'City of Nine Hills?'",
                answer="Nuwakot",
                topic="Geography",
                answer_type="Place",
                multi_step=False,
                requires_reasoning=False,
                urls="https://en.wikipedia.org/wiki/Nuwakot",
            ),
            SimpleQASample(
                original_index=394,
                problem="Which team won the Coppa Italia Serie C in the 1981-82 season?",
                answer="Vicenza",
                topic="Sports",
                answer_type="Other",
                multi_step=False,
                requires_reasoning=False,
                urls="https://en.wikipedia.org/wiki/Coppa_Italia_Serie_C",
            ),
        ]

        if self.topics:
            topics_lower = [t.lower() for t in self.topics]
            mock_samples = [s for s in mock_samples if s.topic.lower() in topics_lower]

        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(mock_samples)

        if self.limit:
            mock_samples = mock_samples[:self.limit]

        self._samples = mock_samples
        print(f"Loaded {len(self._samples)} mock samples for development")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[SimpleQASample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> SimpleQASample:
        return self._samples[idx]

    def get_by_index(self, original_index: int) -> Optional[SimpleQASample]:
        """Get sample by original index."""
        for sample in self._samples:
            if sample.original_index == original_index:
                return sample
        return None
