"""KV cache memory manager for inference simulation."""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Optional

from inference_simulator.request.request import Request
from inference_simulator.types.model_spec import ModelSpec


class KVCacheRetentionPolicy(str, Enum):
    """Policy for KV cache blocks during tool execution.

    RETAIN: Keep blocks in GPU memory, marked non-evictable.
    EVICT: Free blocks immediately (must re-prefill after tool call).
    OFFLOAD_CPU: Move block metadata to CPU tracking (simulated offload).
    """

    RETAIN = "retain"
    EVICT = "evict"
    OFFLOAD_CPU = "offload_cpu"


def compute_token_size_bytes(model_spec: ModelSpec, precision_bytes: float = 2.0) -> int:
    """Compute the KV cache size per token in bytes.

    Each token stores K and V vectors for every layer and KV head.
    Size = 2 (K+V) * num_layers * num_kv_heads * head_dim * precision_bytes
    """
    # For SSM-hybrid models, only attention layers need KV cache.
    # Use attention_layer_count which correctly returns num_layers for
    # pure transformers and the actual attention layer count for hybrids.
    kv_layers = model_spec.attention_layer_count
    return int(
        2
        * kv_layers
        * model_spec.num_kv_heads
        * model_spec.head_dim
        * precision_bytes
    )


class KVCacheManager:
    """Block-based KV cache memory manager.

    Manages GPU memory for KV caches using a block allocator,
    similar to vLLM's PagedAttention approach.

    Attributes:
        total_memory_bytes: Total memory budget for KV cache.
        block_size: Number of tokens per block.
        token_size_bytes: Bytes per token in the KV cache.
    """

    def __init__(
        self,
        total_memory_bytes: int,
        block_size: int = 16,
        token_size_bytes: int | None = None,
        model_spec: ModelSpec | None = None,
        precision_bytes: float = 2.0,
    ) -> None:
        if token_size_bytes is not None:
            self._token_size_bytes = token_size_bytes
        elif model_spec is not None:
            self._token_size_bytes = compute_token_size_bytes(model_spec, precision_bytes)
        else:
            # Reasonable default for a ~7B model: 2 * 32 * 8 * 128 * 2 = 131072
            self._token_size_bytes = 131072

        self._block_size = block_size
        self._bytes_per_block = self._token_size_bytes * block_size
        self._total_blocks = max(1, total_memory_bytes // self._bytes_per_block)
        self._allocated: Dict[int, int] = {}  # request_id -> num_blocks
        self._retained: Dict[int, int] = {}  # request_id -> num_retained_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def token_size_bytes(self) -> int:
        return self._token_size_bytes

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def used_blocks(self) -> int:
        return sum(self._allocated.values())

    @property
    def retained_blocks(self) -> int:
        """Number of blocks held by retained (tool-executing) requests."""
        return sum(self._retained.values())

    @property
    def available_blocks(self) -> int:
        # Retained blocks are tracked separately from _allocated.
        # Both must be subtracted from total to get truly available blocks.
        return self._total_blocks - self.used_blocks - self.retained_blocks

    @property
    def utilization(self) -> float:
        """Fraction of blocks in use."""
        if self._total_blocks == 0:
            return 0.0
        return self.used_blocks / self._total_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        """Compute the number of blocks needed for a given token count."""
        return math.ceil(num_tokens / self._block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if there is enough free memory for the given token count."""
        return self.blocks_needed(num_tokens) <= self.available_blocks

    def allocate(self, request: Request, num_tokens: int) -> bool:
        """Allocate KV cache blocks for a request.

        Args:
            request: The request to allocate for.
            num_tokens: Number of tokens to allocate cache for.

        Returns:
            True if allocation succeeded, False if insufficient memory.
        """
        needed = self.blocks_needed(num_tokens)
        if needed > self.available_blocks:
            return False
        current = self._allocated.get(request.request_id, 0)
        self._allocated[request.request_id] = current + needed
        request.kv_cache_blocks = self._allocated[request.request_id]
        return True

    def free(self, request: Request) -> None:
        """Free all KV cache blocks for a request."""
        self._allocated.pop(request.request_id, None)
        request.kv_cache_blocks = 0

    def get_allocated_blocks(self, request_id: int) -> int:
        """Get number of blocks allocated for a request."""
        return self._allocated.get(request_id, 0)

    def allocate_with_prefix(
        self, request: Request, num_tokens: int, prefix_matched_tokens: int
    ) -> bool:
        """Allocate KV cache blocks, reusing prefix-matched tokens.

        Only allocates blocks for tokens beyond the prefix match,
        since prefix tokens already have cached KV blocks.

        Args:
            request: The request to allocate for.
            num_tokens: Total number of tokens (including prefix).
            prefix_matched_tokens: Tokens covered by prefix cache hit.

        Returns:
            True if allocation succeeded, False if insufficient memory.
        """
        new_tokens = max(0, num_tokens - prefix_matched_tokens)
        needed = self.blocks_needed(new_tokens)
        if needed > self.available_blocks:
            return False
        # Account for both prefix blocks (already cached) and new blocks
        prefix_blocks = self.blocks_needed(prefix_matched_tokens)
        current = self._allocated.get(request.request_id, 0)
        self._allocated[request.request_id] = current + needed + prefix_blocks
        request.kv_cache_blocks = self._allocated[request.request_id]
        request.prefix_matched_tokens = prefix_matched_tokens
        return True

    def retain(self, request: Request) -> None:
        """Mark a request's KV cache blocks as retained (non-evictable).

        Moves blocks from the normal allocated pool to the retained pool,
        preventing the scheduler from evicting them during tool execution.
        """
        blocks = self._allocated.pop(request.request_id, 0)
        if blocks > 0:
            self._retained[request.request_id] = blocks

    def unretain(self, request: Request) -> None:
        """Remove retained flag, returning blocks to normal management.

        Moves blocks back from the retained pool to the allocated pool.
        """
        blocks = self._retained.pop(request.request_id, 0)
        if blocks > 0:
            self._allocated[request.request_id] = blocks

    def offload(self, request: Request) -> Optional[dict]:
        """Offload a request's KV cache blocks to CPU (simulated).

        Removes blocks from GPU tracking and returns a handle that can
        be used to reload them later.

        Returns:
            Offload handle dict, or None if no blocks to offload.
        """
        blocks = self._allocated.pop(request.request_id, 0)
        if blocks == 0:
            blocks = self._retained.pop(request.request_id, 0)
        else:
            self._retained.pop(request.request_id, None)
        if blocks == 0:
            return None
        return {
            "request_id": request.request_id,
            "blocks": blocks,
            "token_size_bytes": self._token_size_bytes,
        }

    def reload(self, offload_handle: dict) -> bool:
        """Reload offloaded KV cache blocks back to GPU.

        Args:
            offload_handle: Handle returned by offload().

        Returns:
            True if reload succeeded, False if insufficient GPU memory.
        """
        blocks = offload_handle["blocks"]
        if blocks > self.available_blocks:
            return False
        request_id = offload_handle["request_id"]
        self._allocated[request_id] = self._allocated.get(request_id, 0) + blocks
        return True
