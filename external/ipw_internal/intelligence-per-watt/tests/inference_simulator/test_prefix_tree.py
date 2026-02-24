"""Tests for RadixAttention-style prefix tree."""

import pytest

from inference_simulator.request.prefix_tree import PrefixNode, PrefixTree


class TestPrefixNode:
    def test_creation(self):
        node = PrefixNode(
            token_ids=(1, 2, 3),
            kv_blocks=4,
            ref_count=1,
        )
        assert node.token_ids == (1, 2, 3)
        assert node.kv_blocks == 4
        assert node.ref_count == 1
        assert node.children == {}
        assert node.last_access_time_ns == 0

    def test_default_children(self):
        node = PrefixNode(token_ids=(), kv_blocks=0, ref_count=0)
        assert len(node.children) == 0


class TestPrefixTreeMatch:
    def test_empty_tree_no_match(self):
        tree = PrefixTree()
        node, matched = tree.match([1, 2, 3])
        assert node is None
        assert matched == 0

    def test_empty_tokens_no_match(self):
        tree = PrefixTree()
        node, matched = tree.match([])
        assert node is None
        assert matched == 0

    def test_exact_match(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        node, matched = tree.match([1, 2, 3])
        assert node is not None
        assert matched == 3
        assert node.kv_blocks == 2

    def test_prefix_match(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        node, matched = tree.match([1, 2, 3, 4, 5])
        assert node is not None
        assert matched == 3

    def test_no_match_different_tokens(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        node, matched = tree.match([4, 5, 6])
        assert node is None
        assert matched == 0

    def test_partial_match(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3, 4, 5], kv_blocks=3, time_ns=100)
        node, matched = tree.match([1, 2, 3])
        # The intermediate nodes (1), (1,2), (1,2,3) have kv_blocks=0
        # Only the leaf (1,2,3,4,5) has blocks
        assert node is None
        assert matched == 0

    def test_multiple_prefixes(self):
        tree = PrefixTree()
        tree.insert([1, 2], kv_blocks=1, time_ns=100)
        tree.insert([1, 2, 3, 4], kv_blocks=2, time_ns=200)
        # Should match the longer prefix
        node, matched = tree.match([1, 2, 3, 4, 5])
        assert matched == 4
        assert node.kv_blocks == 2


class TestPrefixTreeInsert:
    def test_insert_creates_path(self):
        tree = PrefixTree()
        node = tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        assert node.token_ids == (1, 2, 3)
        assert node.kv_blocks == 2
        assert node.ref_count == 1
        assert node.last_access_time_ns == 100

    def test_insert_increments_ref_count(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=200)
        node, _ = tree.match([1, 2, 3])
        assert node.ref_count == 2
        assert node.last_access_time_ns == 200

    def test_insert_updates_total_cached_blocks(self):
        tree = PrefixTree()
        assert tree.total_cached_blocks == 0
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        assert tree.total_cached_blocks == 2
        tree.insert([4, 5, 6], kv_blocks=3, time_ns=200)
        assert tree.total_cached_blocks == 5

    def test_insert_same_prefix_updates_blocks(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        assert tree.total_cached_blocks == 2
        tree.insert([1, 2, 3], kv_blocks=5, time_ns=200)
        assert tree.total_cached_blocks == 5  # Updated, not accumulated

    def test_branching_prefixes(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        tree.insert([1, 2, 4], kv_blocks=3, time_ns=200)
        # Both share [1, 2] prefix
        node3, matched3 = tree.match([1, 2, 3])
        node4, matched4 = tree.match([1, 2, 4])
        assert matched3 == 3
        assert matched4 == 3
        assert node3.kv_blocks == 2
        assert node4.kv_blocks == 3


class TestPrefixTreeEviction:
    def test_evict_lru_frees_blocks(self):
        tree = PrefixTree()
        node1 = tree.insert([1, 2], kv_blocks=2, time_ns=100)
        node2 = tree.insert([3, 4], kv_blocks=3, time_ns=200)
        # Simulate requests completing (ref_count must be 0 to evict)
        node1.ref_count = 0
        node2.ref_count = 0
        freed = tree.evict_lru(2)
        assert freed >= 2
        # Oldest (time_ns=100) should be evicted first
        assert tree.total_cached_blocks <= 3

    def test_evict_lru_zero_needed(self):
        tree = PrefixTree()
        tree.insert([1, 2], kv_blocks=2, time_ns=100)
        freed = tree.evict_lru(0)
        assert freed == 0
        assert tree.total_cached_blocks == 2

    def test_evict_respects_ref_count(self):
        tree = PrefixTree()
        tree.insert([1, 2], kv_blocks=2, time_ns=100)
        # Insert same prefix again to bump ref_count
        tree.insert([1, 2], kv_blocks=2, time_ns=200)
        # ref_count=2, so not evictable
        freed = tree.evict_lru(2)
        assert freed == 0

    def test_evict_oldest_first(self):
        tree = PrefixTree()
        n1 = tree.insert([1], kv_blocks=1, time_ns=100)
        n2 = tree.insert([2], kv_blocks=1, time_ns=200)
        n3 = tree.insert([3], kv_blocks=1, time_ns=300)
        # Simulate requests completing
        n1.ref_count = 0
        n2.ref_count = 0
        n3.ref_count = 0
        # Evict 1 block - should take oldest (token [1])
        freed = tree.evict_lru(1)
        assert freed == 1
        # [1] should be gone
        node, matched = tree.match([1])
        assert node is None
        # [2] and [3] should remain
        node2, _ = tree.match([2])
        assert node2 is not None

    def test_evict_removes_leaf_nodes(self):
        tree = PrefixTree()
        node = tree.insert([1, 2, 3], kv_blocks=2, time_ns=100)
        node.ref_count = 0  # Simulate request completing
        freed = tree.evict_lru(2)
        assert freed == 2
        assert tree.total_cached_blocks == 0
