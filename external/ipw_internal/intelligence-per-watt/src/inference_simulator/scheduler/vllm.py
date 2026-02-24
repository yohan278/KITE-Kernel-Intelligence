"""vLLM-style continuous batching scheduler."""

from __future__ import annotations

from typing import List, Sequence

from inference_simulator.request.request import Batch, Request, RequestState
from inference_simulator.request.kv_cache import KVCacheManager
from inference_simulator.scheduler.base import BaseScheduler, ScheduleResult


class VLLMScheduler(BaseScheduler):
    """Continuous batching scheduler inspired by vLLM.

    Prioritizes running decode batches to minimize latency for in-flight
    requests, then admits new prefill requests up to token and sequence budgets.
    Supports chunked prefill for mixing prefill and decode in the same step.

    Attributes:
        max_num_seqs: Maximum number of sequences in a batch.
        max_num_batched_tokens: Maximum total tokens per batch step.
        enable_chunked_prefill: Allow mixing prefill and decode.
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        enable_chunked_prefill: bool = True,
    ) -> None:
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_chunked_prefill = enable_chunked_prefill
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

        # Count sequences and tokens from running decode batches
        running_seqs: List[Request] = []
        for batch in running_batches:
            running_seqs.extend(
                r for r in batch.requests if r.state == RequestState.DECODING
            )

        num_running = len(running_seqs)
        # Each running decode request contributes 1 token per step
        decode_tokens = num_running

        # Re-batch all running decode requests into a single batch
        if running_seqs:
            decode_batch = Batch(
                batch_id=self._next_batch_id(),
                requests=list(running_seqs),
                is_prefill=False,
            )
            result.new_batches.append(decode_batch)

        # Schedule prefills from waiting queue
        remaining_seqs = self._max_num_seqs - num_running
        remaining_tokens = self._max_num_batched_tokens - decode_tokens

        if remaining_seqs <= 0 or remaining_tokens <= 0:
            return result

        # If chunked prefill is disabled and we have running decodes,
        # don't mix prefill with decode
        if not self._enable_chunked_prefill and num_running > 0:
            return result

        prefill_requests: List[Request] = []
        for request in waiting:
            if request.state != RequestState.WAITING:
                continue
            if remaining_seqs <= 0:
                break

            tokens_needed = request.input_tokens
            if tokens_needed > remaining_tokens:
                # Cannot fit this request
                continue

            # Check KV cache availability
            if not kv_cache.can_allocate(request.input_tokens):
                continue

            prefill_requests.append(request)
            remaining_seqs -= 1
            remaining_tokens -= tokens_needed

        if prefill_requests:
            prefill_batch = Batch(
                batch_id=self._next_batch_id(),
                requests=prefill_requests,
                is_prefill=True,
            )
            result.new_batches.append(prefill_batch)

        return result
