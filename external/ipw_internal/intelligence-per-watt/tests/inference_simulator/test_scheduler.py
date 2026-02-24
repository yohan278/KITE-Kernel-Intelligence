"""Tests for VLLMScheduler and OrcaScheduler."""

from inference_simulator.request.request import Batch, Request, RequestState
from inference_simulator.request.kv_cache import KVCacheManager
from inference_simulator.scheduler.vllm import VLLMScheduler
from inference_simulator.scheduler.orca import OrcaScheduler


def _make_request(request_id: int, input_tokens: int = 100) -> Request:
    return Request(
        request_id=request_id,
        arrival_time_ns=0,
        input_tokens=input_tokens,
        max_output_tokens=50,
    )


def _make_kv_cache(total_mb: int = 1000) -> KVCacheManager:
    return KVCacheManager(
        total_memory_bytes=total_mb * 1_000_000,
        block_size=16,
        token_size_bytes=1000,
    )


class TestVLLMScheduler:
    def test_schedule_empty(self):
        scheduler = VLLMScheduler()
        kv = _make_kv_cache()
        result = scheduler.schedule([], [], kv)
        assert len(result.new_batches) == 0
        assert len(result.preempted_requests) == 0

    def test_schedule_single_prefill(self):
        scheduler = VLLMScheduler(max_num_seqs=256, max_num_batched_tokens=8192)
        kv = _make_kv_cache()
        waiting = [_make_request(0, input_tokens=100)]
        result = scheduler.schedule(waiting, [], kv)

        # Should create one prefill batch
        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        assert prefill_batches[0].size == 1

    def test_schedule_multiple_prefills(self):
        scheduler = VLLMScheduler(max_num_seqs=256, max_num_batched_tokens=8192)
        kv = _make_kv_cache()
        waiting = [_make_request(i, input_tokens=100) for i in range(5)]
        result = scheduler.schedule(waiting, [], kv)

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        assert prefill_batches[0].size == 5

    def test_seq_limit(self):
        scheduler = VLLMScheduler(max_num_seqs=3, max_num_batched_tokens=8192)
        kv = _make_kv_cache()
        waiting = [_make_request(i, input_tokens=100) for i in range(5)]
        result = scheduler.schedule(waiting, [], kv)

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        assert prefill_batches[0].size == 3

    def test_token_budget(self):
        scheduler = VLLMScheduler(max_num_seqs=256, max_num_batched_tokens=250)
        kv = _make_kv_cache()
        waiting = [_make_request(i, input_tokens=100) for i in range(5)]
        result = scheduler.schedule(waiting, [], kv)

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        # Only 2 requests fit: 2 * 100 = 200 <= 250 < 300 = 3 * 100
        assert prefill_batches[0].size == 2

    def test_running_decode_rebatched(self):
        scheduler = VLLMScheduler(max_num_seqs=256, max_num_batched_tokens=8192)
        kv = _make_kv_cache()

        decode_reqs = [_make_request(i, input_tokens=100) for i in range(3)]
        for r in decode_reqs:
            r.state = RequestState.DECODING

        running = [Batch(batch_id=0, requests=decode_reqs, is_prefill=False)]

        result = scheduler.schedule([], running, kv)
        decode_batches = [b for b in result.new_batches if not b.is_prefill]
        assert len(decode_batches) == 1
        assert decode_batches[0].size == 3

    def test_chunked_prefill_disabled(self):
        scheduler = VLLMScheduler(
            max_num_seqs=256,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=False,
        )
        kv = _make_kv_cache()

        decode_reqs = [_make_request(i) for i in range(2)]
        for r in decode_reqs:
            r.state = RequestState.DECODING

        running = [Batch(batch_id=0, requests=decode_reqs, is_prefill=False)]
        waiting = [_make_request(10)]

        result = scheduler.schedule(waiting, running, kv)
        # With chunked prefill disabled and running decodes, no prefill
        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 0

    def test_kv_cache_constraint(self):
        scheduler = VLLMScheduler(max_num_seqs=256, max_num_batched_tokens=8192)
        # Very small cache: can only hold about 1 request
        kv = KVCacheManager(
            total_memory_bytes=20_000,  # ~1 block at 16000 bytes/block
            block_size=16,
            token_size_bytes=1000,
        )
        waiting = [_make_request(i, input_tokens=100) for i in range(3)]
        result = scheduler.schedule(waiting, [], kv)

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        # Should only schedule what fits in KV cache
        if prefill_batches:
            total_scheduled = sum(b.size for b in prefill_batches)
            assert total_scheduled <= 1


class TestOrcaScheduler:
    def test_schedule_empty(self):
        scheduler = OrcaScheduler()
        kv = _make_kv_cache()
        result = scheduler.schedule([], [], kv)
        assert len(result.new_batches) == 0

    def test_schedule_prefills(self):
        scheduler = OrcaScheduler(max_batch_size=64)
        kv = _make_kv_cache()
        waiting = [_make_request(i) for i in range(5)]
        result = scheduler.schedule(waiting, [], kv)

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        assert prefill_batches[0].size == 5

    def test_batch_size_limit(self):
        scheduler = OrcaScheduler(max_batch_size=3)
        kv = _make_kv_cache()
        waiting = [_make_request(i) for i in range(5)]
        result = scheduler.schedule(waiting, [], kv)

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        assert prefill_batches[0].size == 3

    def test_with_running_decodes(self):
        scheduler = OrcaScheduler(max_batch_size=5)
        kv = _make_kv_cache()

        decode_reqs = [_make_request(i) for i in range(3)]
        for r in decode_reqs:
            r.state = RequestState.DECODING

        running = [Batch(batch_id=0, requests=decode_reqs, is_prefill=False)]
        waiting = [_make_request(i + 10) for i in range(5)]

        result = scheduler.schedule(waiting, running, kv)

        decode_batches = [b for b in result.new_batches if not b.is_prefill]
        assert len(decode_batches) == 1
        assert decode_batches[0].size == 3

        prefill_batches = [b for b in result.new_batches if b.is_prefill]
        assert len(prefill_batches) == 1
        # 5 - 3 = 2 remaining slots
        assert prefill_batches[0].size == 2
