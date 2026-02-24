"""Base classes for retrieval MCP servers.

All retrieval servers inherit from BaseRetrievalServer, which extends BaseMCPServer
to provide automatic telemetry capture for energy, power, and latency.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseMCPServer, MCPToolResult


@dataclass
class Document:
    """A document for retrieval indexing.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Optional metadata (e.g., title, source, date)
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate document."""
        if not self.id:
            raise ValueError("Document id cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class RetrievalResult:
    """A single retrieval result with score.

    Attributes:
        document: The retrieved document
        score: Relevance score (higher is better)
        highlights: Optional highlighted text snippets
    """

    document: Document
    score: float
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.document.id,
            "content": self.document.content,
            "metadata": self.document.metadata,
            "score": self.score,
            "highlights": self.highlights,
        }


class BaseRetrievalServer(BaseMCPServer):
    """Base class for retrieval servers with automatic telemetry.

    All retrieval servers inherit from this class, which wraps _execute_impl
    with telemetry capture for energy, power, and latency metrics.

    Subclasses must implement:
    - _execute_impl(): Perform the actual retrieval
    - index_documents(): Index a list of documents
    - clear_index(): Clear all indexed documents

    Example:
        class MyRetrievalServer(BaseRetrievalServer):
            def __init__(self, **kwargs):
                super().__init__(name="retrieval:my", **kwargs)
                self._index = {}

            def index_documents(self, documents: List[Document]) -> int:
                for doc in documents:
                    self._index[doc.id] = doc
                return len(documents)

            def clear_index(self) -> None:
                self._index.clear()

            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                # Implement retrieval logic
                results = self._search(prompt, params.get("top_k", 5))
                return MCPToolResult(
                    content=self._format_results(results),
                    metadata={"num_results": len(results)}
                )
    """

    def __init__(
        self,
        name: str,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
    ):
        """Initialize retrieval server.

        Args:
            name: Server name (e.g., "retrieval:bm25")
            telemetry_collector: Energy monitor collector for telemetry
            event_recorder: EventRecorder for per-action tracking
        """
        super().__init__(
            name=name,
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self._document_count = 0

    def _get_backend(self) -> str:
        """Get the backend type for this server."""
        return "retrieval"

    @abstractmethod
    def index_documents(self, documents: List[Document]) -> int:
        """Index a list of documents.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents successfully indexed
        """
        raise NotImplementedError

    @abstractmethod
    def clear_index(self) -> None:
        """Clear all indexed documents."""
        raise NotImplementedError

    @property
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        return self._document_count

    def _format_results(
        self,
        results: List[RetrievalResult],
        include_scores: bool = True,
        include_metadata: bool = False,
    ) -> str:
        """Format retrieval results as text.

        Args:
            results: List of retrieval results
            include_scores: Whether to include relevance scores
            include_metadata: Whether to include document metadata

        Returns:
            Formatted string with results
        """
        if not results:
            return "No results found."

        lines = []
        for i, result in enumerate(results, 1):
            header = f"[{i}] {result.document.id}"
            if include_scores:
                header += f" (score: {result.score:.4f})"
            lines.append(header)

            if include_metadata and result.document.metadata:
                meta_str = ", ".join(
                    f"{k}={v}" for k, v in result.document.metadata.items()
                )
                lines.append(f"  Metadata: {meta_str}")

            # Show highlights if available, otherwise show content snippet
            if result.highlights:
                for highlight in result.highlights:
                    lines.append(f"  ...{highlight}...")
            else:
                content = result.document.content
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"  {content}")

            lines.append("")

        return "\n".join(lines)

    def health_check(self) -> bool:
        """Check if retrieval server is operational.

        Returns:
            True if server is healthy
        """
        try:
            # Check if we have any documents indexed
            return self._document_count >= 0
        except Exception:
            return False
