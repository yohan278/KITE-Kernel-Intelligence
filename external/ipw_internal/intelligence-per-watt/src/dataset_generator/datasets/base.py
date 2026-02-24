"""Base classes for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSample:
    """A single sample from a dataset for profiling workloads."""

    query: str
    expected_answer: Optional[str] = None
    workload_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDatasetLoader(ABC):
    """Abstract base for loading datasets into DatasetSample lists."""

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> List[DatasetSample]:
        """Load samples from the dataset.

        Args:
            limit: Maximum number of samples to load. None means all.

        Returns:
            List of DatasetSample instances.
        """
        raise NotImplementedError

    @abstractmethod
    def workload_type(self) -> str:
        """Return the workload type string for this dataset."""
        raise NotImplementedError

    @abstractmethod
    def dataset_name(self) -> str:
        """Return the canonical name of this dataset."""
        raise NotImplementedError
