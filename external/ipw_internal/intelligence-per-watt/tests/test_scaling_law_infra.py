"""Tests for scaling-law experiment infrastructure.

Covers:
1. SyntheticDataset: controlled-length prompt generation
2. VLLMClient engine-config passthrough and _coerce_value helper
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# SyntheticDataset tests
# ---------------------------------------------------------------------------


class TestSyntheticDataset:
    """Validate the synthetic dataset provider."""

    def _make(self, **kwargs):
        from ipw.data_loaders.synthetic import SyntheticDataset
        return SyntheticDataset(**kwargs)

    def test_default_params_produce_20_records(self):
        ds = self._make()
        assert ds.size() == 20
        assert len(list(ds.iter_records())) == 20

    def test_custom_num_samples(self):
        ds = self._make(num_samples=5)
        assert ds.size() == 5

    def test_custom_input_tokens(self):
        ds = self._make(input_tokens=1024, num_samples=2)
        records = list(ds.iter_records())
        assert len(records) == 2
        # With more input_tokens the prompt should be longer
        ds_short = self._make(input_tokens=64, num_samples=1)
        short_prompt = list(ds_short.iter_records())[0].problem
        long_prompt = records[0].problem
        assert len(long_prompt.split()) > len(short_prompt.split())

    def test_string_params_coerced(self):
        """CLI passes params as strings; the constructor should accept them."""
        ds = self._make(input_tokens="256", num_samples="3")
        assert ds.size() == 3

    def test_record_structure(self):
        ds = self._make(input_tokens=128, num_samples=2)
        for rec in ds.iter_records():
            assert isinstance(rec.problem, str) and rec.problem
            assert rec.answer == "N/A"
            assert rec.subject == "synthetic"
            assert "target_input_tokens" in rec.dataset_metadata
            assert "sample_index" in rec.dataset_metadata

    def test_sample_index_increments(self):
        ds = self._make(num_samples=5)
        indices = [r.dataset_metadata["sample_index"] for r in ds.iter_records()]
        assert indices == [0, 1, 2, 3, 4]

    def test_prompt_word_count_scales_with_input_tokens(self):
        ds_small = self._make(input_tokens=128, num_samples=1)
        ds_large = self._make(input_tokens=2048, num_samples=1)
        words_small = len(list(ds_small.iter_records())[0].problem.split())
        words_large = len(list(ds_large.iter_records())[0].problem.split())
        # The ratio of words should roughly track the ratio of tokens
        assert words_large > words_small * 4  # 2048/128 = 16x; >4x is lenient

    def test_iterable_via_dunder_iter(self):
        ds = self._make(num_samples=3)
        records = list(ds)
        assert len(records) == 3

    def test_exact_token_count_with_tokenizer(self, monkeypatch):
        from ipw.data_loaders import synthetic as synthetic_mod

        class DummyTokenizer:
            all_special_ids = []

            def encode(self, text, add_special_tokens=False):
                text = text.strip()
                return [] if not text else [1] * len(text.split())

            def decode(self, token_ids, **kwargs):
                return " ".join("tok" for _ in token_ids)

        monkeypatch.setattr(
            synthetic_mod,
            "_load_tokenizer",
            lambda *args, **kwargs: DummyTokenizer(),
        )

        ds = synthetic_mod.SyntheticDataset(
            input_tokens=37,
            num_samples=1,
            tokenizer_model="dummy/model",
        )
        record = list(ds.iter_records())[0]
        assert record.dataset_metadata["actual_input_tokens"] == 37
        assert record.dataset_metadata["tokenizer_mode"] == "exact"

    def test_heuristic_mode_when_tokenizer_unavailable(self, monkeypatch):
        from ipw.data_loaders import synthetic as synthetic_mod

        monkeypatch.setattr(
            synthetic_mod,
            "_load_tokenizer",
            lambda *args, **kwargs: None,
        )

        ds = synthetic_mod.SyntheticDataset(
            input_tokens=256,
            num_samples=1,
            tokenizer_model="missing/model",
        )
        record = list(ds.iter_records())[0]
        assert record.dataset_metadata["actual_input_tokens"] is None
        assert record.dataset_metadata["tokenizer_mode"] == "heuristic"

    def test_strict_token_count_raises_when_not_exact(self, monkeypatch):
        from ipw.data_loaders import synthetic as synthetic_mod

        class ApproxTokenizer:
            all_special_ids = []

            def encode(self, text, add_special_tokens=False):
                text = text.strip()
                return [] if not text else [1] * len(text.split())

            def decode(self, token_ids, **kwargs):
                # Always collapses to 1 token, so exact-length correction fails.
                return "tok"

        monkeypatch.setattr(
            synthetic_mod,
            "_load_tokenizer",
            lambda *args, **kwargs: ApproxTokenizer(),
        )

        with pytest.raises(RuntimeError, match="exact token count"):
            synthetic_mod.SyntheticDataset(
                input_tokens=16,
                num_samples=1,
                tokenizer_model="dummy/model",
                strict_token_count=True,
            )
    def test_registry_lookup(self):
        # Ensure the module has been imported (triggers registration)
        import ipw.data_loaders.synthetic  # noqa: F401
        from ipw.core.registry import DatasetRegistry
        cls = DatasetRegistry.get("synthetic")
        from ipw.data_loaders.synthetic import SyntheticDataset
        assert cls is SyntheticDataset


# ---------------------------------------------------------------------------
# vLLM engine-config passthrough tests
# ---------------------------------------------------------------------------


# We cannot import the real vllm module in CI, so we stub it out.
# The stubs are installed once at module level so that VLLMClient is only
# registered in the ClientRegistry a single time.
_vllm_stub = mock.MagicMock()
_vllm_stub.SamplingParams = mock.MagicMock
_vllm_stub.engine.arg_utils.AsyncEngineArgs = mock.MagicMock
_vllm_stub.sampling_params.RequestOutputKind.DELTA = "DELTA"
_vllm_stub.v1.engine.async_llm.AsyncLLM = mock.MagicMock()

_VLLM_MODS = {
    "vllm": _vllm_stub,
    "vllm.engine": _vllm_stub.engine,
    "vllm.engine.arg_utils": _vllm_stub.engine.arg_utils,
    "vllm.sampling_params": _vllm_stub.sampling_params,
    "vllm.v1": _vllm_stub.v1,
    "vllm.v1.engine": _vllm_stub.v1.engine,
    "vllm.v1.engine.async_llm": _vllm_stub.v1.engine.async_llm,
}

# Only install stubs if vllm is not already importable (avoids clobbering a
# real install when running the full suite on a GPU box).
if "vllm" not in sys.modules:
    try:
        import vllm as _real_vllm  # noqa: F401
    except ImportError:
        sys.modules.update(_VLLM_MODS)

# Now we can safely import; VLLMClient registers exactly once.
from ipw.clients.vllm import VLLMClient, _coerce_value  # noqa: E402


class TestCoerceValue:
    """Test the module-level _coerce_value helper."""

    def test_int_string(self):
        assert _coerce_value("42") == 42

    def test_float_string(self):
        assert _coerce_value("0.95") == 0.95

    def test_bool_true(self):
        assert _coerce_value("true") is True

    def test_bool_false(self):
        assert _coerce_value("false") is False

    def test_plain_string(self):
        assert _coerce_value("float16") == "float16"

    def test_empty_string(self):
        assert _coerce_value("") == ""

    def test_passthrough_int(self):
        assert _coerce_value(42) == 42

    def test_passthrough_none(self):
        assert _coerce_value(None) is None


class TestVLLMEngineConfig:
    """Test that VLLMClient populates _engine_kwargs from config."""

    @staticmethod
    def _make_client(**config):
        return VLLMClient(**config)

    def test_engine_kwargs_from_config(self):
        client = self._make_client(
            dtype="float16", quantization="awq", max_model_len="4096"
        )
        assert client._engine_kwargs["dtype"] == "float16"
        assert client._engine_kwargs["quantization"] == "awq"
        assert client._engine_kwargs["max_model_len"] == 4096

    def test_sampling_params_not_in_engine_kwargs(self):
        client = self._make_client(
            temperature="0.7", top_p="0.9", dtype="bfloat16"
        )
        assert "temperature" not in client._engine_kwargs
        assert "top_p" not in client._engine_kwargs
        assert client._engine_kwargs["dtype"] == "bfloat16"

    def test_unknown_keys_not_in_engine_kwargs(self):
        client = self._make_client(
            dtype="float16", some_random_key="hello"
        )
        assert "some_random_key" not in client._engine_kwargs
        assert client._engine_kwargs["dtype"] == "float16"

    def test_gpu_memory_utilization_coerced(self):
        client = self._make_client(gpu_memory_utilization="0.85")
        assert client._engine_kwargs["gpu_memory_utilization"] == 0.85

    def test_enforce_eager_coerced(self):
        client = self._make_client(enforce_eager="true")
        assert client._engine_kwargs["enforce_eager"] is True

    def test_empty_config_no_engine_kwargs(self):
        client = self._make_client()
        assert client._engine_kwargs == {}
