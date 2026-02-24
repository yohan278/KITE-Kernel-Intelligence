"""Tests for KV-cache retention policies during tool execution."""

import pytest

from inference_simulator.request.request import Request
from inference_simulator.request.kv_cache import (
    KVCacheManager,
    KVCacheRetentionPolicy,
)


def _make_request(request_id: int = 0, input_tokens: int = 100, policy: str = "retain") -> Request:
    return Request(
        request_id=request_id,
        arrival_time_ns=0,
        input_tokens=input_tokens,
        max_output_tokens=50,
        retention_policy=policy,
    )


class TestKVCacheRetentionPolicy:
    def test_enum_values(self):
        assert KVCacheRetentionPolicy.RETAIN.value == "retain"
        assert KVCacheRetentionPolicy.EVICT.value == "evict"
        assert KVCacheRetentionPolicy.OFFLOAD_CPU.value == "offload_cpu"


class TestRetain:
    def test_retain_moves_blocks_from_allocated(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=1, input_tokens=32)
        mgr.allocate(req, 32)
        assert mgr.used_blocks == 2

        mgr.retain(req)
        # Blocks moved from allocated to retained
        assert mgr.used_blocks == 0
        assert mgr.retained_blocks == 2

    def test_retained_blocks_excluded_from_available(self):
        mgr = KVCacheManager(
            total_memory_bytes=3200,  # 2 blocks
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=1, input_tokens=16)
        mgr.allocate(req, 16)  # 1 block
        assert mgr.available_blocks == 1

        mgr.retain(req)
        # Retained blocks are also not available
        assert mgr.available_blocks == 1

    def test_retain_then_unretain(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=1, input_tokens=32)
        mgr.allocate(req, 32)
        initial_available = mgr.available_blocks

        mgr.retain(req)
        assert mgr.retained_blocks == 2

        mgr.unretain(req)
        assert mgr.retained_blocks == 0
        assert mgr.used_blocks == 2
        assert mgr.available_blocks == initial_available


class TestOffload:
    def test_offload_returns_handle(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=1, input_tokens=32)
        mgr.allocate(req, 32)

        handle = mgr.offload(req)
        assert handle is not None
        assert handle["request_id"] == 1
        assert handle["blocks"] == 2
        # Blocks freed from GPU
        assert mgr.used_blocks == 0
        assert mgr.retained_blocks == 0

    def test_offload_no_blocks_returns_none(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=99, input_tokens=32)
        handle = mgr.offload(req)
        assert handle is None

    def test_reload_restores_blocks(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=1, input_tokens=32)
        mgr.allocate(req, 32)
        handle = mgr.offload(req)

        success = mgr.reload(handle)
        assert success
        assert mgr.used_blocks == 2


class TestEvict:
    def test_evict_frees_blocks(self):
        """Evict policy is the legacy free() behavior."""
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        req = _make_request(request_id=1, input_tokens=32, policy="evict")
        mgr.allocate(req, 32)
        assert mgr.used_blocks == 2

        mgr.free(req)
        assert mgr.used_blocks == 0
        assert mgr.retained_blocks == 0


class TestRequestRetentionPolicy:
    def test_default_retain(self):
        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=100,
            max_output_tokens=50,
        )
        assert req.retention_policy == "retain"

    def test_custom_policy(self):
        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=100,
            max_output_tokens=50,
            retention_policy="offload_cpu",
        )
        assert req.retention_policy == "offload_cpu"

    def test_offload_handle_default_none(self):
        req = Request(
            request_id=0,
            arrival_time_ns=0,
            input_tokens=100,
            max_output_tokens=50,
        )
        assert req._offload_handle is None
