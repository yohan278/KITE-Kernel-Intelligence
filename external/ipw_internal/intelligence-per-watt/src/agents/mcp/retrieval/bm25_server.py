"""BM25 sparse retrieval server.

Implements BM25 (Best Matching 25) for fast, CPU-only keyword-based retrieval.
Suitable for:
- Keyword-heavy queries
- Low-latency requirements (~10ms)
- Resource-constrained environments
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..base import MCPToolResult
from .base import BaseRetrievalServer, Document, RetrievalResult


class BM25RetrievalServer(BaseRetrievalServer):
    """BM25 sparse retrieval server.

    Uses the BM25 algorithm for keyword-based document retrieval.
    Fast, CPU-only, and requires no external dependencies beyond rank-bm25.

    BM25 Parameters:
    - k1: Term frequency saturation parameter (default: 1.5)
    - b: Length normalization parameter (default: 0.75)

    Example:
        server = BM25RetrievalServer()
        server.index_documents([
            Document(id="1", content="Machine learning is a subset of AI."),
            Document(id="2", content="Deep learning uses neural networks."),
        ])

        result = server.execute("machine learning neural networks", top_k=5)
    """

    def __init__(
        self,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[str] = None,
    ):
        """Initialize BM25 retrieval server.

        Args:
            telemetry_collector: Energy monitor collector for telemetry
            event_recorder: EventRecorder for per-action tracking
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter
            tokenizer: Tokenizer to use ("simple" or None for default)
        """
        super().__init__(
            name="retrieval:bm25",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[Any] = None  # Lazy import

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercased words)
        """
        # Simple whitespace + punctuation tokenization
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def index_documents(self, documents: List[Document]) -> int:
        """Index documents for BM25 search.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents indexed
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25RetrievalServer. "
                "Install with: pip install rank-bm25"
            )

        self._documents = list(documents)
        self._tokenized_corpus = [self._tokenize(doc.content) for doc in documents]

        # Create BM25 index
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        self._document_count = len(self._documents)
        return self._document_count

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None
        self._document_count = 0

    def _search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for documents matching query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        if self._bm25 is None or not self._documents:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # Create results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                doc = self._documents[idx]

                # Generate highlights from matching tokens
                highlights = self._generate_highlights(doc.content, query_tokens)

                results.append(
                    RetrievalResult(
                        document=doc,
                        score=float(scores[idx]),
                        highlights=highlights,
                    )
                )

        return results

    def _generate_highlights(
        self, content: str, query_tokens: List[str], max_highlights: int = 3
    ) -> List[str]:
        """Generate highlighted snippets showing query matches.

        Args:
            content: Document content
            query_tokens: Query tokens to highlight
            max_highlights: Maximum number of highlights

        Returns:
            List of highlighted snippets
        """
        highlights = []
        sentences = re.split(r"[.!?]\s+", content)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(token in sentence_lower for token in query_tokens):
                # Truncate long sentences
                if len(sentence) > 150:
                    sentence = sentence[:150] + "..."
                highlights.append(sentence.strip())
                if len(highlights) >= max_highlights:
                    break

        return highlights

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute BM25 search.

        Args:
            prompt: Search query
            **params: Additional parameters:
                - top_k: Number of results (default: 5)
                - include_scores: Include scores in output (default: True)
                - include_metadata: Include metadata in output (default: False)

        Returns:
            MCPToolResult with retrieved documents
        """
        top_k = params.get("top_k", 5)
        include_scores = params.get("include_scores", True)
        include_metadata = params.get("include_metadata", False)

        if self._bm25 is None:
            return MCPToolResult(
                content="No documents indexed. Call index_documents() first.",
                cost_usd=0.0,
                metadata={"tool": "retrieval:bm25", "error": "no_index"},
            )

        # Perform search
        results = self._search(prompt, top_k=top_k)

        # Format results
        content = self._format_results(
            results,
            include_scores=include_scores,
            include_metadata=include_metadata,
        )

        return MCPToolResult(
            content=content,
            cost_usd=0.0,
            metadata={
                "tool": "retrieval:bm25",
                "query": prompt,
                "num_results": len(results),
                "top_k": top_k,
                "indexed_documents": self._document_count,
            },
        )
