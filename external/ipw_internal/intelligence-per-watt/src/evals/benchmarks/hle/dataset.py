"""HLE (Humanity's Last Exam) dataset loader.

HLE is a challenging benchmark from the Center for AI Safety (CAIS) that
tests expert-level knowledge across many academic disciplines.

Dataset source: https://huggingface.co/datasets/cais/hle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class HLESample:
    """Single HLE benchmark sample.

    Attributes:
        task_id: Unique identifier for the task
        question: The question/task to solve
        answer: Ground truth answer
        category: Task category (e.g., math, coding, reasoning)
        difficulty: Difficulty level (if available)
        has_image: Whether the sample contains image data
        has_audio: Whether the sample contains audio data
        metadata: Additional metadata
    """
    task_id: str
    question: str
    answer: str
    category: str = "general"
    difficulty: Optional[str] = None
    has_image: bool = False
    has_audio: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_text_only(self) -> bool:
        """Check if sample is text-only (no image or audio)."""
        return not self.has_image and not self.has_audio


def load_hle_dataset(
    split: str = "test",
    limit: Optional[int] = None,
    category_filter: Optional[str] = None,
    dataset_path: Optional[str] = None,
    text_only: bool = False,
) -> List[HLESample]:
    """Load HLE dataset.

    Args:
        split: Dataset split ("train", "validation", "test")
        limit: Maximum number of samples to load
        category_filter: Filter by category (if None, load all)
        dataset_path: Path to local dataset or HuggingFace dataset name
        text_only: If True, filter to only text-only samples (no images/audio)

    Returns:
        List of HLE samples
    """
    # Try to load from HuggingFace
    try:
        from datasets import load_dataset

        # Official HLE dataset from CAIS
        if dataset_path is None:
            dataset_path = "cais/hle"

        dataset = load_dataset(dataset_path, split=split)

        samples = []
        for idx, item in enumerate(dataset):
            if limit and len(samples) >= limit:
                break

            # Extract fields (adjust based on actual dataset schema)
            task_id = item.get("id") or item.get("task_id") or f"hle_{idx}"
            question = item.get("question") or item.get("instruction") or item.get("prompt")
            answer = item.get("answer") or item.get("gold_answer") or item.get("response")

            # Skip samples where question or answer is missing/None
            if question is None or answer is None:
                continue

            category = item.get("category") or item.get("subject") or item.get("type") or "general"
            difficulty = item.get("difficulty") or item.get("level")

            # Detect multimodal content
            has_image = bool(
                item.get("image") or item.get("image_path") or item.get("images")
            )
            has_audio = bool(
                item.get("audio") or item.get("audio_path") or item.get("audios")
            )

            # Filter by category if specified
            if category_filter and category != category_filter:
                continue

            sample = HLESample(
                task_id=str(task_id),
                question=str(question),
                answer=str(answer),
                category=str(category),
                difficulty=str(difficulty) if difficulty else None,
                has_image=has_image,
                has_audio=has_audio,
                metadata={k: v for k, v in item.items() if k not in [
                    "id", "task_id", "question", "instruction", "prompt",
                    "answer", "gold_answer", "response", "category", "type",
                    "difficulty", "level", "image", "image_path", "images",
                    "audio", "audio_path", "audios"
                ]}
            )

            # Apply text_only filter
            if text_only and not sample.is_text_only:
                continue

            samples.append(sample)

        return samples

    except ImportError:
        raise ImportError(
            "datasets package required for HLE. Install with: pip install datasets"
        )
    except Exception as e:
        # Fallback: try loading from local file
        import json
        from pathlib import Path

        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path) as f:
                data = json.load(f)

            samples = []
            for idx, item in enumerate(data):
                if limit and len(samples) >= limit:
                    break

                task_id = item.get("id", f"hle_{idx}")
                question = item.get("question", item.get("instruction"))
                answer = item.get("answer", item.get("gold_answer"))

                # Skip samples where question or answer is missing/None
                if question is None or answer is None:
                    continue

                category = item.get("category", "general")

                # Detect multimodal content
                has_image = bool(
                    item.get("image") or item.get("image_path") or item.get("images")
                )
                has_audio = bool(
                    item.get("audio") or item.get("audio_path") or item.get("audios")
                )

                if category_filter and category != category_filter:
                    continue

                sample = HLESample(
                    task_id=str(task_id),
                    question=str(question),
                    answer=str(answer),
                    category=str(category),
                    has_image=has_image,
                    has_audio=has_audio,
                )

                # Apply text_only filter
                if text_only and not sample.is_text_only:
                    continue

                samples.append(sample)

            return samples

        raise RuntimeError(
            f"Failed to load HLE dataset: {e}\n"
            "Either install datasets package or provide dataset_path to local JSON file."
        )


def iter_hle_samples(
    split: str = "test",
    limit: Optional[int] = None,
    category_filter: Optional[str] = None,
    dataset_path: Optional[str] = None,
    text_only: bool = False,
) -> Iterator[HLESample]:
    """Iterate over HLE dataset samples.

    Args:
        split: Dataset split
        limit: Maximum samples
        category_filter: Category filter
        dataset_path: Dataset path
        text_only: If True, filter to only text-only samples

    Yields:
        HLE samples
    """
    samples = load_hle_dataset(
        split=split,
        limit=limit,
        category_filter=category_filter,
        dataset_path=dataset_path,
        text_only=text_only,
    )
    yield from samples


__all__ = [
    "HLESample",
    "load_hle_dataset",
    "iter_hle_samples",
]
