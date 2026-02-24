"""Tests for Request state transitions and Batch creation."""

from inference_simulator.request.request import Batch, Request, RequestState


class TestRequestState:
    def test_enum_values(self):
        assert RequestState.WAITING == "waiting"
        assert RequestState.PREFILLING == "prefilling"
        assert RequestState.DECODING == "decoding"
        assert RequestState.COMPLETED == "completed"


class TestRequest:
    def test_create(self):
        r = Request(
            request_id=0,
            arrival_time_ns=1000,
            input_tokens=100,
            max_output_tokens=50,
        )
        assert r.request_id == 0
        assert r.arrival_time_ns == 1000
        assert r.input_tokens == 100
        assert r.max_output_tokens == 50
        assert r.state == RequestState.WAITING
        assert r.tokens_generated == 0
        assert r.prefill_start_ns is None
        assert r.first_token_ns is None
        assert r.completion_ns is None

    def test_total_tokens(self):
        r = Request(request_id=0, arrival_time_ns=0, input_tokens=100, max_output_tokens=50)
        assert r.total_tokens == 100
        r.tokens_generated = 20
        assert r.total_tokens == 120

    def test_is_complete(self):
        r = Request(request_id=0, arrival_time_ns=0, input_tokens=100, max_output_tokens=50)
        assert not r.is_complete
        r.state = RequestState.COMPLETED
        assert r.is_complete

    def test_ttft(self):
        r = Request(request_id=0, arrival_time_ns=1000, input_tokens=100, max_output_tokens=50)
        assert r.ttft_ns is None
        r.first_token_ns = 5000
        assert r.ttft_ns == 4000

    def test_e2e_latency(self):
        r = Request(request_id=0, arrival_time_ns=1000, input_tokens=100, max_output_tokens=50)
        assert r.e2e_latency_ns is None
        r.completion_ns = 10000
        assert r.e2e_latency_ns == 9000

    def test_state_transitions(self):
        r = Request(request_id=0, arrival_time_ns=0, input_tokens=100, max_output_tokens=50)
        assert r.state == RequestState.WAITING

        r.state = RequestState.PREFILLING
        r.prefill_start_ns = 1000
        assert r.state == RequestState.PREFILLING

        r.state = RequestState.DECODING
        r.first_token_ns = 2000
        assert r.state == RequestState.DECODING

        r.tokens_generated = 50
        r.state = RequestState.COMPLETED
        r.completion_ns = 10000
        assert r.is_complete
        assert r.ttft_ns == 2000
        assert r.e2e_latency_ns == 10000


class TestBatch:
    def test_create_empty(self):
        b = Batch(batch_id=0)
        assert b.batch_id == 0
        assert b.size == 0
        assert not b.is_prefill

    def test_with_requests(self):
        requests = [
            Request(request_id=i, arrival_time_ns=0, input_tokens=100, max_output_tokens=50)
            for i in range(4)
        ]
        b = Batch(batch_id=1, requests=requests, is_prefill=True)
        assert b.size == 4
        assert b.is_prefill
        assert b.total_tokens == 400  # 4 * 100 for prefill

    def test_decode_total_tokens(self):
        requests = [
            Request(request_id=i, arrival_time_ns=0, input_tokens=100, max_output_tokens=50)
            for i in range(4)
        ]
        b = Batch(batch_id=1, requests=requests, is_prefill=False)
        assert b.total_tokens == 4  # 1 token per request for decode
