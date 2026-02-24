"""Dense neural retrieval server using FAISS and sentence-transformers.

Implements semantic search using dense embeddings:
- Uses sentence-transformers for encoding
- FAISS for fast similarity search
- Supports both CPU and GPU indices
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..base import MCPToolResult
from .base import BaseRetrievalServer, Document, RetrievalResult


class DenseRetrievalServer(BaseRetrievalServer):
    """Dense neural retrieval server using FAISS + sentence-transformers.

    Uses dense embeddings for semantic search. Requires:
    - sentence-transformers for encoding
    - faiss-cpu (or faiss-gpu) for similarity search

    Latency: ~50ms per query
    Cost: Zero (local inference)

    Example:
        server = DenseRetrievalServer(model_name="all-MiniLM-L6-v2")
        server.index_documents([
            Document(id="1", content="Machine learning automates data analysis."),
            Document(id="2", content="Neural networks learn from examples."),
        ])

        # Semantic search - finds related concepts
        result = server.execute("AI learns patterns from data", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        use_gpu: bool = False,
        gpu_device: int = 0,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """Initialize dense retrieval server.

        Args:
            model_name: Sentence-transformers model name
            telemetry_collector: Energy monitor collector for telemetry
            event_recorder: EventRecorder for per-action tracking
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
            gpu_device: GPU device index for FAISS and encoder (default: 0).
                Use a different index than the LM to avoid GPU contention.
            normalize_embeddings: Whether to normalize embeddings (for cosine sim)
            batch_size: Batch size for encoding
        """
        super().__init__(
            name="retrieval:dense",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        self._encoder: Optional[Any] = None  # Lazy import
        self._index: Optional[Any] = None  # FAISS index
        self._documents: List[Document] = []
        self._embedding_dim: Optional[int] = None

    def _get_encoder(self) -> Any:
        """Get or create the sentence encoder.

        Returns:
            SentenceTransformer model
        """
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for DenseRetrievalServer. "
                    "Install with: pip install sentence-transformers"
                )

            # Set device based on use_gpu and gpu_device settings
            if self.use_gpu:
                device = f"cuda:{self.gpu_device}"
            else:
                device = "cpu"

            self._encoder = SentenceTransformer(self.model_name, device=device)
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()

        return self._encoder

    def _create_faiss_index(self, dimension: int) -> Any:
        """Create a FAISS index.

        Args:
            dimension: Embedding dimension

        Returns:
            FAISS index
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for DenseRetrievalServer. "
                "Install with: pip install faiss-cpu"
            )

        # Use Inner Product for normalized vectors (equivalent to cosine sim)
        if self.normalize_embeddings:
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                # Use specified GPU device instead of hardcoded GPU 0
                index = faiss.index_cpu_to_gpu(res, self.gpu_device, index)
            except Exception:
                pass  # Fall back to CPU

        return index

    def index_documents(self, documents: List[Document]) -> int:
        """Index documents using dense embeddings.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents indexed
        """
        import numpy as np

        if not documents:
            return 0

        encoder = self._get_encoder()

        # Store documents
        self._documents = list(documents)

        # Encode all documents
        texts = [doc.content for doc in documents]
        embeddings = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Ensure embeddings are float32 numpy array
        embeddings = np.array(embeddings, dtype=np.float32)

        # Create and populate index
        self._index = self._create_faiss_index(embeddings.shape[1])
        self._index.add(embeddings)

        self._document_count = len(self._documents)
        return self._document_count

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._index = None
        self._document_count = 0

    def _search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for documents semantically similar to query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        import numpy as np

        if self._index is None or not self._documents:
            return []

        encoder = self._get_encoder()

        # Encode query
        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search
        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding, k)

        # Create results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self._documents):
                doc = self._documents[idx]

                # Generate highlights (sentences most similar to query)
                highlights = self._generate_highlights(doc.content, query)

                results.append(
                    RetrievalResult(
                        document=doc,
                        score=float(score),
                        highlights=highlights,
                    )
                )

        return results

    def _generate_highlights(
        self, content: str, query: str, max_highlights: int = 3
    ) -> List[str]:
        """Generate highlighted snippets.

        For dense retrieval, we return the first few sentences as highlights
        since the entire document is semantically relevant.

        Args:
            content: Document content
            query: Original query
            max_highlights: Maximum number of highlights

        Returns:
            List of highlighted snippets
        """
        # Split into sentences
        sentences = re.split(r"[.!?]\s+", content)

        highlights = []
        for sentence in sentences[:max_highlights]:
            sentence = sentence.strip()
            if sentence:
                if len(sentence) > 150:
                    sentence = sentence[:150] + "..."
                highlights.append(sentence)

        return highlights

    def save_index(self, path: Union[str, Path]) -> None:
        """Save the FAISS index to disk.

        Args:
            path: Path to save the index
        """
        import faiss
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

        # Save documents as JSON
        docs_data = [
            {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
            for doc in self._documents
        ]
        with open(path / "documents.json", "w") as f:
            json.dump(docs_data, f)

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "document_count": self._document_count,
            "embedding_dim": self._embedding_dim,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load a FAISS index from disk.

        Args:
            path: Path to load the index from
        """
        import faiss
        import json

        path = Path(path)

        # Load FAISS index
        self._index = faiss.read_index(str(path / "index.faiss"))

        # Load documents
        with open(path / "documents.json") as f:
            docs_data = json.load(f)
        self._documents = [
            Document(id=d["id"], content=d["content"], metadata=d.get("metadata", {}))
            for d in docs_data
        ]

        # Load metadata
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        self._document_count = meta["document_count"]
        self._embedding_dim = meta.get("embedding_dim")

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute dense semantic search.

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

        if self._index is None:
            return MCPToolResult(
                content="No documents indexed. Call index_documents() first.",
                cost_usd=0.0,
                metadata={"tool": "retrieval:dense", "error": "no_index"},
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
                "tool": "retrieval:dense",
                "query": prompt,
                "num_results": len(results),
                "top_k": top_k,
                "model": self.model_name,
                "indexed_documents": self._document_count,
            },
        )
