"""Hybrid retrieval server combining BM25 + dense with RRF fusion.

Combines the strengths of sparse (BM25) and dense retrieval:
- BM25 for exact keyword matching
- Dense for semantic understanding
- Reciprocal Rank Fusion (RRF) for combining results

This achieves the best accuracy at the cost of higher latency (~100ms).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..base import MCPToolResult
from .base import BaseRetrievalServer, Document, RetrievalResult
from .bm25_server import BM25RetrievalServer
from .dense_server import DenseRetrievalServer


class HybridRetrievalServer(BaseRetrievalServer):
    """Hybrid BM25 + dense retrieval with RRF fusion.

    Combines sparse and dense retrieval for best accuracy:
    - BM25 finds documents with exact keyword matches
    - Dense finds semantically similar documents
    - RRF (Reciprocal Rank Fusion) combines rankings

    RRF Formula: score(d) = sum(1 / (k + rank(d))) for each retriever
    where k is typically 60.

    Latency: ~100ms per query
    Cost: Zero (local inference)

    Example:
        server = HybridRetrievalServer()
        server.index_documents([
            Document(id="1", content="Machine learning automates data analysis."),
            Document(id="2", content="Neural networks learn from examples."),
        ])

        # Combines keyword and semantic matching
        result = server.execute("ML data patterns", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        use_gpu: bool = False,
        gpu_device: int = 0,
    ):
        """Initialize hybrid retrieval server.

        Args:
            model_name: Sentence-transformers model for dense retrieval
            telemetry_collector: Energy monitor collector for telemetry
            event_recorder: EventRecorder for per-action tracking
            bm25_weight: Weight for BM25 scores in fusion (default: 0.5)
            dense_weight: Weight for dense scores in fusion (default: 0.5)
            rrf_k: RRF constant k (default: 60)
            use_gpu: Whether to use GPU for dense retrieval
            gpu_device: GPU device index for dense retrieval (default: 0).
                Use a different index than the LM to avoid GPU contention.
        """
        super().__init__(
            name="retrieval:hybrid",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        # Initialize sub-retrievers
        self._bm25 = BM25RetrievalServer(
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self._dense = DenseRetrievalServer(
            model_name=model_name,
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        self._documents: Dict[str, Document] = {}

    def index_documents(self, documents: List[Document]) -> int:
        """Index documents for hybrid search.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents indexed
        """
        # Store document mapping
        self._documents = {doc.id: doc for doc in documents}

        # Index in both retrievers
        bm25_count = self._bm25.index_documents(documents)
        dense_count = self._dense.index_documents(documents)

        self._document_count = min(bm25_count, dense_count)
        return self._document_count

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._bm25.clear_index()
        self._dense.clear_index()
        self._document_count = 0

    def _rrf_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF assigns each document a score based on its rank in each result list:
        score(d) = sum(weight / (k + rank(d))) for each retriever

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            top_k: Number of results to return

        Returns:
            Fused and re-ranked results
        """
        # Build rank maps (1-indexed)
        bm25_ranks: Dict[str, int] = {
            r.document.id: i + 1 for i, r in enumerate(bm25_results)
        }
        dense_ranks: Dict[str, int] = {
            r.document.id: i + 1 for i, r in enumerate(dense_results)
        }

        # Collect all document IDs
        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += self.bm25_weight / (self.rrf_k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[doc_id])
            rrf_scores[doc_id] = score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result list
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            # Combine highlights from both retrievers
            highlights = []
            for r in bm25_results:
                if r.document.id == doc_id:
                    highlights.extend(r.highlights)
                    break
            for r in dense_results:
                if r.document.id == doc_id:
                    for h in r.highlights:
                        if h not in highlights:
                            highlights.append(h)
                    break

            results.append(
                RetrievalResult(
                    document=doc,
                    score=rrf_scores[doc_id],
                    highlights=highlights[:3],  # Limit highlights
                )
            )

        return results

    def _search(
        self,
        query: str,
        top_k: int = 5,
        bm25_candidates: int = 20,
        dense_candidates: int = 20,
    ) -> List[RetrievalResult]:
        """Search using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: Number of BM25 candidates to consider
            dense_candidates: Number of dense candidates to consider

        Returns:
            Fused results
        """
        # Get results from both retrievers
        bm25_results = self._bm25._search(query, top_k=bm25_candidates)
        dense_results = self._dense._search(query, top_k=dense_candidates)

        # Fuse results
        return self._rrf_fusion(bm25_results, dense_results, top_k)

    def save_index(self, path: Union[str, Path]) -> None:
        """Save the hybrid index to disk.

        Args:
            path: Path to save the index
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save dense index (BM25 index is rebuilt on load)
        self._dense.save_index(path / "dense")

        # Save metadata
        meta = {
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "document_count": self._document_count,
        }
        with open(path / "hybrid_metadata.json", "w") as f:
            json.dump(meta, f)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load a hybrid index from disk.

        Args:
            path: Path to load the index from
        """
        import json

        path = Path(path)

        # Load dense index
        self._dense.load_index(path / "dense")

        # Load documents and rebuild BM25
        self._documents = {doc.id: doc for doc in self._dense._documents}
        self._bm25.index_documents(list(self._documents.values()))

        # Load metadata
        with open(path / "hybrid_metadata.json") as f:
            meta = json.load(f)
        self._document_count = meta["document_count"]

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute hybrid search.

        Args:
            prompt: Search query
            **params: Additional parameters:
                - top_k: Number of results (default: 5)
                - bm25_candidates: BM25 candidate pool size (default: 20)
                - dense_candidates: Dense candidate pool size (default: 20)
                - include_scores: Include scores in output (default: True)
                - include_metadata: Include metadata in output (default: False)

        Returns:
            MCPToolResult with retrieved documents
        """
        top_k = params.get("top_k", 5)
        bm25_candidates = params.get("bm25_candidates", 20)
        dense_candidates = params.get("dense_candidates", 20)
        include_scores = params.get("include_scores", True)
        include_metadata = params.get("include_metadata", False)

        if self._document_count == 0:
            return MCPToolResult(
                content="No documents indexed. Call index_documents() first.",
                cost_usd=0.0,
                metadata={"tool": "retrieval:hybrid", "error": "no_index"},
            )

        # Perform hybrid search
        results = self._search(
            prompt,
            top_k=top_k,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
        )

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
                "tool": "retrieval:hybrid",
                "query": prompt,
                "num_results": len(results),
                "top_k": top_k,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "indexed_documents": self._document_count,
            },
        )
