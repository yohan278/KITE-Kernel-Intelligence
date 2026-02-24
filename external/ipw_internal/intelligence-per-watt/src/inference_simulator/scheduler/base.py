"""Abstract base class for inference schedulers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

from inference_simulator.request.request import Batch, Request
from inference_simulator.request.kv_cache import KVCacheManager


@dataclass
class ScheduleResult:
    """Result of a scheduling decision.

    Attributes:
        new_batches: Batches to execute in the next step.
        preempted_requests: Requests evicted to free resources.
    """

    new_batches: List[Batch] = field(default_factory=list)
    preempted_requests: List[Request] = field(default_factory=list)


class BaseScheduler(ABC):
    """Abstract scheduler that decides which requests to batch together."""

    @abstractmethod
    def schedule(
        self,
        waiting: Sequence[Request],
        running_batches: Sequence[Batch],
        kv_cache: KVCacheManager,
    ) -> ScheduleResult:
        """Determine what to execute next.

        Args:
            waiting: Requests waiting to be scheduled (in arrival order).
            running_batches: Currently active decode batches.
            kv_cache: KV cache manager for memory checks.

        Returns:
            ScheduleResult with new batches and any preempted requests.
        """
        raise NotImplementedError
