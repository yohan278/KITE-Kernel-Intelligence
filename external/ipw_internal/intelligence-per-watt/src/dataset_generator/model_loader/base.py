"""Base class for model loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from inference_simulator.types.model_spec import ModelSpec


class BaseModelLoader(ABC):
    """Abstract base for loading model architectures into ModelSpec."""

    @abstractmethod
    def load(self, model_id: str) -> ModelSpec:
        """Load model configuration and return a ModelSpec.

        Args:
            model_id: Model identifier (e.g., HuggingFace repo ID).

        Returns:
            Frozen ModelSpec describing the model architecture.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_architectures(self) -> List[str]:
        """List of architecture names this loader handles."""
        raise NotImplementedError
