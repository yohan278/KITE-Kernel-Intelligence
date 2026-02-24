"""Base class for dataset characterizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from inference_simulator.types.workload_profile import WorkloadProfile


class BaseCharacterizer(ABC):
    """Analyze a dataset and produce a WorkloadProfile with fitted distributions."""

    @abstractmethod
    def characterize(self, limit: Optional[int] = None) -> "WorkloadProfile":
        """Load data, tokenize, fit distributions, return a WorkloadProfile.

        Args:
            limit: Maximum number of samples to analyze. None means all.

        Returns:
            A WorkloadProfile summarizing the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def workload_type(self) -> str:
        """Return the workload type string (e.g. 'chat', 'reasoning')."""
        raise NotImplementedError

    @abstractmethod
    def dataset_name(self) -> str:
        """Return the canonical dataset name."""
        raise NotImplementedError
