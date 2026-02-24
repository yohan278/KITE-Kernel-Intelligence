"""Tests for KVCacheManager allocate/free."""

import pytest

from inference_simulator.request.request import Request
from inference_simulator.request.kv_cache import KVCacheManager, compute_token_size_bytes
from inference_simulator.types.model_spec import (
    ArchitectureType,
    AttentionType,
    ModelSpec,
)


def _make_request(request_id: int = 0, input_tokens: int = 100) -> Request:
    return Request(
        request_id=request_id,
        arrival_time_ns=0,
        input_tokens=input_tokens,
        max_output_tokens=50,
    )


class TestComputeTokenSizeBytes:
    def test_basic(self):
        spec = ModelSpec(
            model_id="test",
            architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.GQA,
            num_layers=32,
            hidden_dim=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=11008,
            vocab_size=32000,
        )
        # 2 * 32 * 8 * 128 * 2 = 131072 bytes per token
        size = compute_token_size_bytes(spec, precision_bytes=2.0)
        assert size == 2 * 32 * 8 * 128 * 2


class TestKVCacheManager:
    def test_creation(self):
        mgr = KVCacheManager(
            total_memory_bytes=1_000_000,
            block_size=16,
            token_size_bytes=1000,
        )
        # 1000 bytes/token * 16 tokens/block = 16000 bytes/block
        # 1_000_000 / 16000 = 62 blocks
        assert mgr.total_blocks == 62
        assert mgr.available_blocks == 62
        assert mgr.used_blocks == 0

    def test_allocate_and_free(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        # 100 * 16 = 1600 bytes/block, 100000/1600 = 62 blocks

        req = _make_request(request_id=1, input_tokens=32)
        # 32 tokens -> ceil(32/16) = 2 blocks
        assert mgr.allocate(req, 32)
        assert mgr.used_blocks == 2
        assert mgr.available_blocks == 60
        assert req.kv_cache_blocks == 2

        mgr.free(req)
        assert mgr.used_blocks == 0
        assert mgr.available_blocks == 62
        assert req.kv_cache_blocks == 0

    def test_allocate_fails_when_full(self):
        mgr = KVCacheManager(
            total_memory_bytes=1600,  # Exactly 1 block
            block_size=16,
            token_size_bytes=100,
        )
        assert mgr.total_blocks == 1

        req1 = _make_request(request_id=1, input_tokens=16)
        assert mgr.allocate(req1, 16)
        assert mgr.available_blocks == 0

        req2 = _make_request(request_id=2, input_tokens=16)
        assert not mgr.allocate(req2, 16)

    def test_can_allocate(self):
        mgr = KVCacheManager(
            total_memory_bytes=3200,  # 2 blocks
            block_size=16,
            token_size_bytes=100,
        )
        assert mgr.can_allocate(16)  # 1 block
        assert mgr.can_allocate(32)  # 2 blocks
        assert not mgr.can_allocate(33)  # 3 blocks needed

    def test_blocks_needed(self):
        mgr = KVCacheManager(
            total_memory_bytes=100_000,
            block_size=16,
            token_size_bytes=100,
        )
        assert mgr.blocks_needed(1) == 1
        assert mgr.blocks_needed(16) == 1
        assert mgr.blocks_needed(17) == 2
        assert mgr.blocks_needed(32) == 2
        assert mgr.blocks_needed(33) == 3

    def test_utilization(self):
        mgr = KVCacheManager(
            total_memory_bytes=3200,  # 2 blocks
            block_size=16,
            token_size_bytes=100,
        )
        assert mgr.utilization == pytest.approx(0.0)
        req = _make_request(request_id=1, input_tokens=16)
        mgr.allocate(req, 16)
        assert mgr.utilization == pytest.approx(0.5)

    def test_multiple_requests(self):
        mgr = KVCacheManager(
            total_memory_bytes=10_000,
            block_size=16,
            token_size_bytes=100,
        )
        # 6 blocks total

        r1 = _make_request(request_id=1, input_tokens=16)
        r2 = _make_request(request_id=2, input_tokens=32)

        assert mgr.allocate(r1, 16)  # 1 block
        assert mgr.allocate(r2, 32)  # 2 blocks
        assert mgr.used_blocks == 3

        mgr.free(r1)
        assert mgr.used_blocks == 2

        mgr.free(r2)
        assert mgr.used_blocks == 0

    def test_incremental_allocation(self):
        mgr = KVCacheManager(
            total_memory_bytes=10_000,
            block_size=16,
            token_size_bytes=100,
        )

        req = _make_request(request_id=1, input_tokens=16)
        assert mgr.allocate(req, 16)  # 1 block
        assert mgr.allocate(req, 16)  # +1 block
        assert mgr.get_allocated_blocks(1) == 2
        assert req.kv_cache_blocks == 2

    def test_with_model_spec(self):
        spec = ModelSpec(
            model_id="test",
            architecture_type=ArchitectureType.DENSE_TRANSFORMER,
            attention_type=AttentionType.GQA,
            num_layers=32,
            hidden_dim=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            intermediate_dim=11008,
            vocab_size=32000,
        )
        mgr = KVCacheManager(
            total_memory_bytes=10_000_000_000,  # 10 GB
            model_spec=spec,
        )
        assert mgr.token_size_bytes == 2 * 32 * 8 * 128 * 2
        assert mgr.total_blocks > 0
