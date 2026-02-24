"""Fast token counting for dataset characterization."""

from __future__ import annotations


class FastTokenCounter:
    """Token counter using tiktoken cl100k_base, with word-count fallback."""

    def __init__(self):
        self._encoder = None
        self._tried_import = False

    def count(self, text: str) -> int:
        """Count tokens in text.

        Uses tiktoken cl100k_base if available, otherwise estimates
        as ``max(1, int(word_count * 1.3))``.
        """
        if not text:
            return 0

        if not self._tried_import:
            self._tried_import = True
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoder = None

        if self._encoder is not None:
            return len(self._encoder.encode(text))

        return max(1, int(len(text.split()) * 1.3))
