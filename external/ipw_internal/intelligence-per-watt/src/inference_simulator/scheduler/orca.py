"""ORCA-style iteration-level scheduling."""

from __future__ import annotations

from typing import List, Sequence

from inference_simulator.request.request import Batch, Request, RequestState
from inference_simulator.request.kv_cache import KVCacheManager
from inference_simulator.scheduler.base import BaseScheduler, ScheduleResult


class OrcaScheduler(BaseScheduler):
    """Iteration-level scheduler inspired by the ORCA system.

    Uses a simpler scheduling policy: each iteration processes a batch
    that mixes prefill and decode requests up to max_batch_size. New
    requests are greedily admitted in FCFS order.

    Attributes:
        max_batch_size: Maximum number of requests per iteration.
    """

    def __init__(self, max_batch_size: int = 64) -> None:
        self._max_batch_size = max_batch_size
        self._batch_counter = 0

    def _next_batch_id(self) -> int:
        self._batch_counter += 1
        return self._batch_counter

    def schedule(
        self,
        waiting: Sequence[Request],
        running_batches: Sequence[Batch],
        kv_cache: KVCacheManager,
    ) -> ScheduleResult:
        result = ScheduleResult()

        # Collect all actively decoding requests
        decode_requests: List[Request] = []
        for batch in running_batches:
            decode_requests.extend(
                r for r in batch.requests if r.state == RequestState.DECODING
            )

        remaining_slots = self._max_batch_size - len(decode_requests)

        # Re-batch decode requests
        if decode_requests:
            decode_batch = Batch(
                batch_id=self._next_batch_id(),
                requests=list(decode_requests),
                is_prefill=False,
            )
            result.new_batches.append(decode_batch)

        # Admit new prefill requests
        if remaining_slots <= 0:
            return result

        prefill_requests: List[Request] = []
        for request in waiting:
            if request.state != RequestState.WAITING:
                continue
            if remaining_slots <= 0:
                break
            if not kv_cache.can_allocate(request.input_tokens):
                continue

            prefill_requests.append(request)
            remaining_slots -= 1

        if prefill_requests:
            prefill_batch = Batch(
                batch_id=self._next_batch_id(),
                requests=prefill_requests,
                is_prefill=True,
            )
            result.new_batches.append(prefill_batch)

        return result
