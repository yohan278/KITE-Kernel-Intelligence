"""RadixAttention-style prefix tree for KV cache sharing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PrefixNode:
    """A node in the prefix tree representing a cached token sequence.

    Attributes:
        token_ids: Token sequence for this node.
        kv_blocks: Number of KV-cache blocks for this prefix.
        ref_count: Number of active requests sharing this prefix.
        children: Child nodes keyed by next token ID.
        last_access_time_ns: Timestamp for LRU eviction.
    """

    token_ids: Tuple[int, ...]
    kv_blocks: int
    ref_count: int
    children: Dict[int, PrefixNode] = field(default_factory=dict)
    last_access_time_ns: int = 0


class PrefixTree:
    """Radix tree for prefix caching of KV cache blocks.

    Enables sharing of KV cache blocks across requests that share
    common prompt prefixes, reducing redundant prefill computation.
    """

    def __init__(self) -> None:
        self._root = PrefixNode(
            token_ids=(),
            kv_blocks=0,
            ref_count=0,
            children={},
            last_access_time_ns=0,
        )
        self._total_cached_blocks = 0

    @property
    def total_cached_blocks(self) -> int:
        return self._total_cached_blocks

    def match(self, tokens: List[int]) -> Tuple[Optional[PrefixNode], int]:
        """Find the longest matching prefix in the tree.

        Walks the tree following token IDs and returns the deepest
        matching node along with the number of tokens matched.

        Args:
            tokens: Token sequence to match against the tree.

        Returns:
            Tuple of (matched node or None, number of matched tokens).
        """
        if not tokens:
            return None, 0

        node = self._root
        matched = 0
        best_node: Optional[PrefixNode] = None
        best_matched = 0

        for token_id in tokens:
            if token_id in node.children:
                node = node.children[token_id]
                matched += 1
                if node.kv_blocks > 0:
                    best_node = node
                    best_matched = matched
            else:
                break

        return best_node, best_matched

    def insert(self, tokens: List[int], kv_blocks: int, time_ns: int) -> PrefixNode:
        """Insert a prefix after a request completes, caching its KV blocks.

        Creates nodes along the token path if they don't exist, and
        updates the leaf node with the block count and timestamp.

        Args:
            tokens: Token sequence to insert.
            kv_blocks: Number of KV cache blocks to associate with this prefix.
            time_ns: Current simulation time for LRU tracking.

        Returns:
            The leaf node representing the inserted prefix.
        """
        node = self._root
        for token_id in tokens:
            if token_id not in node.children:
                node.children[token_id] = PrefixNode(
                    token_ids=node.token_ids + (token_id,),
                    kv_blocks=0,
                    ref_count=0,
                    children={},
                    last_access_time_ns=time_ns,
                )
            node = node.children[token_id]

        # Update the leaf with block info
        if node.kv_blocks == 0:
            self._total_cached_blocks += kv_blocks
        else:
            # Update existing entry: adjust total by difference
            self._total_cached_blocks += kv_blocks - node.kv_blocks

        node.kv_blocks = kv_blocks
        node.ref_count += 1
        node.last_access_time_ns = time_ns
        return node

    def evict_lru(self, blocks_needed: int) -> int:
        """Evict least-recently-used prefixes to free blocks.

        Performs a breadth-first collection of leaf nodes (or nodes
        with no active references), sorted by last access time,
        and evicts them until enough blocks are freed.

        Args:
            blocks_needed: Number of blocks to free.

        Returns:
            Number of blocks actually freed.
        """
        if blocks_needed <= 0:
            return 0

        # Collect all evictable nodes (leaf nodes with ref_count == 0 and blocks > 0)
        evictable: List[Tuple[int, PrefixNode, PrefixNode, int]] = []
        self._collect_evictable(self._root, evictable)

        # Sort by last access time (oldest first = LRU)
        evictable.sort(key=lambda x: x[0])

        freed = 0
        for _, node, parent, token_id in evictable:
            if freed >= blocks_needed:
                break
            freed += node.kv_blocks
            self._total_cached_blocks -= node.kv_blocks
            node.kv_blocks = 0
            # Remove leaf nodes with no children
            if not node.children:
                del parent.children[token_id]

        return freed

    def _collect_evictable(
        self,
        node: PrefixNode,
        result: List[Tuple[int, PrefixNode, PrefixNode, int]],
    ) -> None:
        """Recursively collect evictable nodes."""
        for token_id, child in list(node.children.items()):
            self._collect_evictable(child, result)
            # A node is evictable if it has no active references and has blocks
            if child.ref_count == 0 and child.kv_blocks > 0 and not child.children:
                result.append((child.last_access_time_ns, child, node, token_id))
