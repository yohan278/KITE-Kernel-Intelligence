"""Synthetic dataset provider for controlled-length prompt generation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from ipw.core.registry import DatasetRegistry
from ipw.core.types import DatasetRecord
from .base import DatasetProvider

# Fixed English passage used as filler text.  Repeated / truncated to reach
# the target word count so that every prompt has roughly the same token count.
_FILLER_PASSAGE = (
    "The rapid advancement of large language models has transformed the landscape"
    " of artificial intelligence research and application development. These models"
    " demonstrate remarkable capabilities in natural language understanding,"
    " generation, and reasoning across diverse domains. Researchers continue to"
    " explore the fundamental scaling laws that govern model performance as a"
    " function of compute, data, and parameter count. Energy efficiency has emerged"
    " as a critical consideration, with growing attention to the environmental and"
    " economic costs of training and serving these systems at scale. Hardware"
    " accelerators, quantization techniques, and optimized serving frameworks each"
    " play a role in reducing the energy footprint of inference workloads while"
    " maintaining output quality. Understanding the interplay between throughput,"
    " latency, and power consumption is essential for deploying language models"
    " sustainably. Benchmark suites that measure intelligence per watt provide"
    " a principled framework for evaluating these trade-offs across hardware"
    " configurations, model architectures, and serving policies."
)

_FILLER_WORDS: list[str] = _FILLER_PASSAGE.split()


def _build_prompt(target_words: int) -> str:
    """Return a prompt string of approximately *target_words* words."""
    if target_words <= 0:
        return ""
    full_repeats, remainder = divmod(target_words, len(_FILLER_WORDS))
    parts = _FILLER_WORDS * full_repeats + _FILLER_WORDS[:remainder]
    return " ".join(parts)


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _decode_prompt(tokenizer: Any, token_ids: list[int]) -> str:
    for kwargs in (
        {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
        {"skip_special_tokens": True},
        {},
    ):
        try:
            return tokenizer.decode(token_ids, **kwargs)
        except TypeError:
            continue
    return tokenizer.decode(token_ids)


def _strip_special_ids(tokenizer: Any, token_ids: list[int]) -> list[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if not special_ids:
        return token_ids
    return [token_id for token_id in token_ids if token_id not in special_ids]


def _token_count(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(_strip_special_ids(tokenizer, token_ids))


def _repeat_ids(seed_ids: list[int], target_tokens: int) -> list[int]:
    repeats, remainder = divmod(target_tokens, len(seed_ids))
    return seed_ids * repeats + seed_ids[:remainder]


def _seed_token_ids(tokenizer: Any) -> list[int]:
    candidate_ids = tokenizer.encode(_FILLER_PASSAGE, add_special_tokens=False)
    candidate_ids = _strip_special_ids(tokenizer, candidate_ids)
    if candidate_ids:
        return candidate_ids

    fallback_ids = tokenizer.encode("the", add_special_tokens=False)
    fallback_ids = _strip_special_ids(tokenizer, fallback_ids)
    if fallback_ids:
        return fallback_ids

    raise RuntimeError("Tokenizer could not produce usable token ids for synthetic prompts")


def _build_exact_token_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int, bool]:
    """Build a prompt and report (prompt, actual_token_count, exact_match)."""
    if target_tokens <= 0:
        return "", 0, True

    seed_ids = _seed_token_ids(tokenizer)
    token_ids = _repeat_ids(seed_ids, target_tokens)

    # Decode/re-encode can drift for some tokenizers; iteratively correct.
    for _ in range(12):
        prompt = _decode_prompt(tokenizer, token_ids)
        encoded_ids = tokenizer.encode(prompt, add_special_tokens=False)
        encoded_ids = _strip_special_ids(tokenizer, encoded_ids)
        actual_tokens = len(encoded_ids)

        if actual_tokens == target_tokens:
            return prompt, actual_tokens, True

        if actual_tokens <= 0:
            break

        if actual_tokens > target_tokens:
            token_ids = encoded_ids[:target_tokens]
        else:
            token_ids = encoded_ids + _repeat_ids(seed_ids, target_tokens - actual_tokens)

    prompt = _decode_prompt(tokenizer, token_ids)
    actual_tokens = _token_count(tokenizer, prompt)
    return prompt, actual_tokens, actual_tokens == target_tokens


@lru_cache(maxsize=16)
def _load_tokenizer(model_name: str, trust_remote_code: bool) -> Any | None:
    if not model_name:
        return None

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None

    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return None


@DatasetRegistry.register("synthetic")
class SyntheticDataset(DatasetProvider):
    """Generate prompts of a controlled token length for energy-scaling experiments."""

    dataset_id = "synthetic"
    dataset_name = "Synthetic"

    def __init__(
        self,
        input_tokens: int | str = 512,
        num_samples: int | str = 20,
        tokenizer_model: str | None = None,
        trust_remote_code: bool | str = True,
        strict_token_count: bool | str = False,
        **_extra: Any,
    ) -> None:
        self._input_tokens = int(input_tokens)
        self._num_samples = int(num_samples)
        self._tokenizer_model = (tokenizer_model or "").strip()
        self._trust_remote_code = _coerce_bool(trust_remote_code)
        self._strict_token_count = _coerce_bool(strict_token_count)
        (
            self._prompt,
            self._actual_input_tokens,
            self._tokenizer_mode,
        ) = self._build_prompt_payload()
        self._records = tuple(self._generate_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def _build_prompt_payload(self) -> tuple[str, int | None, str]:
        tokenizer = _load_tokenizer(self._tokenizer_model, self._trust_remote_code)
        if tokenizer is not None:
            prompt, actual_tokens, exact = _build_exact_token_prompt(
                tokenizer, self._input_tokens
            )
            if self._strict_token_count and not exact:
                raise RuntimeError(
                    "Failed to generate a synthetic prompt with exact token count "
                    f"(target={self._input_tokens}, actual={actual_tokens}) for "
                    f"tokenizer '{self._tokenizer_model}'"
                )
            return prompt, actual_tokens, "exact" if exact else "approximate"

        # ~1.3 tokens per English word is a commonly used heuristic.
        target_words = max(1, round(self._input_tokens / 1.3))
        return _build_prompt(target_words), None, "heuristic"

    def _generate_records(self) -> Iterable[DatasetRecord]:
        for i in range(self._num_samples):
            yield DatasetRecord(
                problem=self._prompt,
                answer="N/A",
                subject="synthetic",
                dataset_metadata={
                    "target_input_tokens": self._input_tokens,
                    "actual_input_tokens": self._actual_input_tokens,
                    "tokenizer_model": self._tokenizer_model or None,
                    "tokenizer_mode": self._tokenizer_mode,
                    "sample_index": i,
                },
            )
