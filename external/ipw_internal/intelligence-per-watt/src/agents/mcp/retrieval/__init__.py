"""Retrieval MCP servers for benchmark integration.

This module provides retrieval capabilities (BM25, dense, grep, hybrid) for
benchmarks that need document retrieval, with full energy/power/latency profiling.

Target benchmarks:
- FRAMES: Wikipedia article retrieval for multi-hop QA
- BrowseComp: Local corpus retrieval (supplement existing web search)
- DeepResearch: Domain document retrieval for research synthesis
- SimpleQA: RAG-augmented variant for factual QA
- FinanceBench: Financial document retrieval

Example:
    from agents.mcp.retrieval import HybridRetrievalServer, Document

    # Create server with telemetry
    server = HybridRetrievalServer(telemetry_collector=collector)

    # Index documents
    docs = [Document(id="1", content="..."), Document(id="2", content="...")]
    server.index_documents(docs)

    # Search
    result = server.execute("search query", top_k=5)
    print(result.content)  # Retrieved documents
    print(result.latency_seconds)  # Retrieval latency
"""

from .base import BaseRetrievalServer, Document, RetrievalResult
from .grep_server import GrepRetrievalServer
from .bm25_server import BM25RetrievalServer
from .dense_server import DenseRetrievalServer
from .hybrid_server import HybridRetrievalServer
from .index_manager import IndexManager

__all__ = [
    # Base classes
    "BaseRetrievalServer",
    "Document",
    "RetrievalResult",
    # Server implementations
    "GrepRetrievalServer",
    "BM25RetrievalServer",
    "DenseRetrievalServer",
    "HybridRetrievalServer",
    # Index management
    "IndexManager",
]
