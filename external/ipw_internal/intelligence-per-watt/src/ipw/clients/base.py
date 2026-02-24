from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Sequence, Tuple

from ipw.core.types import Response


class InferenceClient(ABC):
    """
    Base class for inference service integrations.

    Subclasses must be registered with ``ClientRegistry`` to become discoverable.
    """

    client_id: str
    client_name: str

    def __init__(self, base_url: str, **config: Any) -> None:
        self.base_url = base_url
        self._config = config

    @abstractmethod
    def run_concurrent(
        self,
        model: str,
        prompt_iter: Iterable[Tuple[int, str]],
        max_in_flight: int,
        **params: Any,
    ) -> Iterator[Tuple[int, Response]]:
        """Run prompts concurrently and yield (dataset_index, Response) tuples."""

    @abstractmethod
    def list_models(self) -> Sequence[str]:
        """Return the list of models exposed by the client."""

    @abstractmethod
    def health(self) -> bool:
        """Return True when the client is healthy and reachable."""

    def prepare(self, model: str) -> None:
        """Optional hook to perform warmup before serving requests."""
        return None


__all__ = ["InferenceClient"]
