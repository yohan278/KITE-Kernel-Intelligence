"""Patches for vLLM 0.11.0 compatibility with transformers 5.x.

Applies two fixes:
1. Registers model configs missing from transformers' CONFIG_MAPPING
   (e.g., glm4_moe_lite) so AutoConfig can resolve them.
2. Monkey-patches vLLM's get_cached_tokenizer to handle the removal of
   `all_special_tokens_extended` in transformers 5.x.
"""

from __future__ import annotations


def register_missing_configs() -> None:
    """Register model configs missing from transformers' CONFIG_MAPPING.

    Some model architectures (e.g., glm4_moe_lite) are supported by vLLM
    but not yet in the installed transformers' CONFIG_MAPPING. Registering
    them in CONFIG_MAPPING allows AutoConfig.from_pretrained() with
    trust_remote_code=True to resolve the config class.

    Note: We intentionally do NOT add to vLLM's _CONFIG_REGISTRY because
    it's a LazyConfigDict that expects string values (class names resolvable
    via vllm.transformers_utils.configs), not class objects. Letting models
    fall through to the AutoConfig path with trust_remote_code=True works.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "glm4_moe_lite" not in CONFIG_MAPPING:
            from transformers import Glm4MoeConfig

            CONFIG_MAPPING.register("glm4_moe_lite", Glm4MoeConfig)
    except (ImportError, AttributeError):
        pass


def patch_tokenizer_compat() -> None:
    """Patch vLLM's get_cached_tokenizer for transformers 5.x compatibility.

    transformers 5.x removed `all_special_tokens_extended` from tokenizers.
    vLLM 0.11.0-0.15.x's `get_cached_tokenizer` accesses this attribute,
    causing an AttributeError. This patch replaces the function with a version
    that falls back to `all_special_tokens` when the extended variant is missing.

    vLLM 0.16+ removed `get_cached_tokenizer` entirely, so the patch is
    skipped on newer versions.
    """
    try:
        import vllm.transformers_utils.tokenizer as tok_module
    except ImportError:
        return

    if not hasattr(tok_module, "get_cached_tokenizer"):
        # vLLM 0.16+ no longer has this function — no patch needed
        return

    import copy
    import contextlib

    def patched_get_cached_tokenizer(tokenizer):
        """Cached tokenizer wrapper tolerant of missing all_special_tokens_extended."""
        cached_tokenizer = copy.copy(tokenizer)

        tokenizer_all_special_ids = tokenizer.all_special_ids
        tokenizer_all_special_tokens = tokenizer.all_special_tokens
        # Fallback: use all_special_tokens if extended variant is missing
        tokenizer_all_special_tokens_extended = getattr(
            tokenizer, "all_special_tokens_extended", tokenizer_all_special_tokens
        )
        tokenizer_vocab = tokenizer.get_vocab()
        tokenizer_len = len(tokenizer)

        max_token_id = max(tokenizer_vocab.values())
        if hasattr(tokenizer, "vocab_size"):
            with contextlib.suppress(NotImplementedError):
                max_token_id = max(max_token_id, tokenizer.vocab_size)

        class CachedTokenizer(tokenizer.__class__):

            @property
            def all_special_ids(self):
                return tokenizer_all_special_ids

            @property
            def all_special_tokens(self):
                return tokenizer_all_special_tokens

            @property
            def all_special_tokens_extended(self):
                return tokenizer_all_special_tokens_extended

            def get_vocab(self):
                return tokenizer_vocab

            def __len__(self):
                return tokenizer_len

            @property
            def max_token_id(self):
                return max_token_id

        CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

        cached_tokenizer.__class__ = CachedTokenizer
        return cached_tokenizer

    tok_module.get_cached_tokenizer = patched_get_cached_tokenizer
