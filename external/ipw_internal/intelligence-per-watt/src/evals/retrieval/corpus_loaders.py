# src/evals/retrieval/corpus_loaders.py
"""
Benchmark-specific corpus loading functions.

Each function loads documents from a benchmark's data source and converts
them to the standard Document format for retrieval indexing.

Loaders:
- load_financebench_corpus: Load evidence from FinanceBench samples
- load_frames_corpus: Load Wikipedia articles from FRAMES wiki_links
- load_simpleqa_corpus: Load from Wikipedia/knowledge base for factual QA
- load_deepresearch_corpus: Load domain papers/sources for research tasks
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agents.mcp.retrieval.base import Document

logger = logging.getLogger(__name__)


def load_financebench_corpus(
    samples: Optional[List[Any]] = None,
    cache_dir: Optional[Path] = None,
    include_full_page: bool = False,
) -> List[Document]:
    """Load corpus from FinanceBench evidence items.

    Converts EvidenceItem objects from FinanceBench samples into Documents
    for retrieval indexing.

    Args:
        samples: List of FinanceBenchSample objects (loads from dataset if None)
        cache_dir: Cache directory for dataset loading
        include_full_page: Whether to include full page text (larger but more context)

    Returns:
        List of Document objects from evidence
    """
    # Load samples if not provided
    if samples is None:
        from evals.benchmarks.financebench.dataset import load_financebench_samples

        samples = load_financebench_samples(cache_dir=cache_dir)

    documents = []
    seen_ids = set()

    for sample in samples:
        # Get evidence items
        evidence = getattr(sample, "evidence", [])
        if not evidence:
            continue

        for ev in evidence:
            # Extract evidence text
            if include_full_page and ev.evidence_text_full_page:
                content = ev.evidence_text_full_page
            else:
                content = ev.evidence_text

            if not content or not content.strip():
                continue

            # Create unique ID
            doc_name = ev.evidence_doc_name or "unknown"
            page_num = ev.evidence_page_num or 0
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            doc_id = f"{doc_name}_p{page_num}_{content_hash}"

            # Skip duplicates
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            # Create document
            documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        "source": doc_name,
                        "page": page_num,
                        "company": getattr(sample, "company", ""),
                        "question_uid": getattr(sample, "uid", ""),
                    },
                )
            )

    logger.info(f"Loaded {len(documents)} documents from FinanceBench evidence")
    return documents


def load_frames_corpus(
    samples: Optional[List[Any]] = None,
    fetch_wikipedia: bool = False,
    max_articles: int = 500,
    cache_dir: Optional[Path] = None,
) -> List[Document]:
    """Load corpus from FRAMES Wikipedia links.

    FRAMES samples contain wiki_links pointing to relevant Wikipedia articles.
    This function can either:
    1. Create document stubs from the URLs (for linking)
    2. Fetch actual article content (if fetch_wikipedia=True)

    Args:
        samples: List of FRAMESSample objects (loads from dataset if None)
        fetch_wikipedia: Whether to fetch actual Wikipedia content
        max_articles: Maximum number of articles to fetch
        cache_dir: Cache directory for fetched articles

    Returns:
        List of Document objects from Wikipedia
    """
    # Load samples if not provided
    if samples is None:
        from evals.benchmarks.frames.dataset import load_frames_samples

        samples = list(load_frames_samples())

    # Collect unique Wikipedia URLs
    wiki_urls = set()
    for sample in samples:
        wiki_links = getattr(sample, "wiki_links", [])
        for url in wiki_links:
            if url and "wikipedia.org" in url:
                wiki_urls.add(url)

    logger.info(f"Found {len(wiki_urls)} unique Wikipedia URLs in FRAMES samples")

    if not fetch_wikipedia:
        # Create stub documents with URLs as content
        documents = []
        for url in list(wiki_urls)[:max_articles]:
            # Extract title from URL
            title = _extract_wiki_title(url)
            doc_id = f"wiki_{hashlib.md5(url.encode()).hexdigest()[:12]}"

            documents.append(
                Document(
                    id=doc_id,
                    content=f"Wikipedia article: {title}\nURL: {url}",
                    metadata={
                        "title": title,
                        "url": url,
                        "source": "wikipedia",
                    },
                )
            )

        logger.info(f"Created {len(documents)} Wikipedia stub documents")
        return documents

    # Fetch actual Wikipedia content
    return _fetch_wikipedia_articles(
        list(wiki_urls)[:max_articles],
        cache_dir=cache_dir,
    )


def _extract_wiki_title(url: str) -> str:
    """Extract article title from Wikipedia URL.

    Args:
        url: Wikipedia URL

    Returns:
        Article title
    """
    # Handle various Wikipedia URL formats
    # https://en.wikipedia.org/wiki/Article_Title
    # https://en.wikipedia.org/wiki/Article_Title#Section
    match = re.search(r"/wiki/([^#?]+)", url)
    if match:
        title = match.group(1).replace("_", " ")
        return title
    return url


def _fetch_wikipedia_articles(
    urls: List[str],
    cache_dir: Optional[Path] = None,
) -> List[Document]:
    """Fetch Wikipedia article content.

    Args:
        urls: List of Wikipedia URLs
        cache_dir: Cache directory for fetched articles

    Returns:
        List of Document objects with article content
    """
    import json

    # Setup cache
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ipw" / "wikipedia"
    cache_dir.mkdir(parents=True, exist_ok=True)

    documents = []

    for url in urls:
        title = _extract_wiki_title(url)
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"

        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                content = cached.get("content", "")
                if content:
                    documents.append(
                        Document(
                            id=f"wiki_{cache_key[:12]}",
                            content=content,
                            metadata={
                                "title": title,
                                "url": url,
                                "source": "wikipedia",
                            },
                        )
                    )
                    continue
            except Exception as e:
                logger.warning(f"Failed to load cached article {title}: {e}")

        # Fetch from Wikipedia API
        try:
            content = _fetch_wikipedia_content(title)
            if content:
                # Cache the content
                with open(cache_path, "w") as f:
                    json.dump({"url": url, "title": title, "content": content}, f)

                documents.append(
                    Document(
                        id=f"wiki_{cache_key[:12]}",
                        content=content,
                        metadata={
                            "title": title,
                            "url": url,
                            "source": "wikipedia",
                        },
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to fetch Wikipedia article {title}: {e}")

    logger.info(f"Loaded {len(documents)} Wikipedia articles")
    return documents


def _fetch_wikipedia_content(title: str, max_chars: int = 50000) -> str:
    """Fetch Wikipedia article content using the API.

    Args:
        title: Article title
        max_chars: Maximum content length

    Returns:
        Article content text
    """
    import requests

    # Use Wikipedia API to get plain text
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,  # Plain text, no HTML
        "format": "json",
    }

    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    for page_id, page in pages.items():
        if page_id == "-1":  # Page not found
            return ""

        extract = page.get("extract", "")
        if len(extract) > max_chars:
            extract = extract[:max_chars] + "..."
        return extract

    return ""


def load_simpleqa_corpus(
    topics: Optional[List[str]] = None,
    max_documents: int = 10000,
    source: str = "wikipedia_simple",
) -> List[Document]:
    """Load corpus for SimpleQA factual knowledge augmentation.

    SimpleQA tests parametric knowledge. For RAG augmentation, we can use:
    - Simple Wikipedia (simpler language, good for factual questions)
    - Full Wikipedia (more comprehensive)
    - Custom knowledge base

    Args:
        topics: Optional topic filter (Science, History, etc.)
        max_documents: Maximum number of documents to load
        source: Source to load from ("wikipedia_simple", "wikipedia", "huggingface")

    Returns:
        List of Document objects
    """
    if source == "wikipedia_simple":
        return _load_simple_wikipedia(max_documents=max_documents)
    elif source == "wikipedia":
        return _load_wikipedia_subset(topics=topics, max_documents=max_documents)
    elif source == "huggingface":
        return _load_huggingface_wiki(max_documents=max_documents)
    else:
        logger.warning(f"Unknown source {source}, returning empty corpus")
        return []


def _load_simple_wikipedia(max_documents: int = 10000) -> List[Document]:
    """Load Simple Wikipedia articles from HuggingFace.

    Args:
        max_documents: Maximum articles to load

    Returns:
        List of Document objects
    """
    try:
        from datasets import load_dataset

        # Load Simple Wikipedia dataset
        logger.info("Loading Simple Wikipedia from HuggingFace...")
        ds = load_dataset(
            "wikipedia",
            "20220301.simple",
            split="train",
            streaming=True,
        )

        documents = []
        for i, item in enumerate(ds):
            if i >= max_documents:
                break

            title = item.get("title", f"doc_{i}")
            text = item.get("text", "")

            if not text or len(text) < 100:
                continue

            # Truncate very long articles
            if len(text) > 10000:
                text = text[:10000] + "..."

            documents.append(
                Document(
                    id=f"simplewiki_{i}",
                    content=text,
                    metadata={"title": title, "source": "simple_wikipedia"},
                )
            )

        logger.info(f"Loaded {len(documents)} Simple Wikipedia articles")
        return documents

    except Exception as e:
        logger.warning(f"Failed to load Simple Wikipedia: {e}")
        return []


def _load_wikipedia_subset(
    topics: Optional[List[str]] = None,
    max_documents: int = 10000,
) -> List[Document]:
    """Load Wikipedia subset, optionally filtered by topic.

    Args:
        topics: Topic keywords to filter by (matches against title/content).
                If None, loads all articles without filtering.
        max_documents: Maximum documents to load

    Returns:
        List of Document objects
    """
    # If no topics specified, use simple wikipedia loader
    if not topics:
        return _load_simple_wikipedia(max_documents=max_documents)

    try:
        from datasets import load_dataset

        # Normalize topics for case-insensitive matching
        topics_lower = [t.lower() for t in topics]

        logger.info(f"Loading Wikipedia articles filtered by topics: {topics}")

        # Use Simple Wikipedia for faster loading (smaller dataset)
        dataset = load_dataset(
            "wikipedia",
            "20220301.simple",
            split="train",
            streaming=True,
        )

        documents = []
        checked = 0
        matched = 0

        for item in dataset:
            checked += 1
            title = item.get("title", "")
            text = item.get("text", "")

            # Skip short articles
            if len(text) < 100:
                continue

            # Check if title or content matches any topic
            title_lower = title.lower()
            text_lower = text[:2000].lower()  # Check first 2000 chars for efficiency

            topic_match = None
            for topic in topics_lower:
                if topic in title_lower or topic in text_lower:
                    topic_match = topic
                    break

            if topic_match is None:
                continue

            matched += 1
            documents.append(
                Document(
                    id=f"wiki_topic_{len(documents)}",
                    content=text[:10000],  # Truncate to 10k chars
                    metadata={
                        "title": title,
                        "source": "wikipedia_filtered",
                        "matched_topic": topic_match,
                    },
                )
            )

            if len(documents) >= max_documents:
                break

            # Log progress periodically
            if checked % 10000 == 0:
                logger.debug(f"Checked {checked} articles, matched {matched}")

        logger.info(
            f"Loaded {len(documents)} Wikipedia articles matching topics "
            f"(checked {checked} total)"
        )
        return documents

    except Exception as e:
        logger.warning(f"Failed to load filtered Wikipedia: {e}")
        logger.warning("Falling back to unfiltered Simple Wikipedia")
        return _load_simple_wikipedia(max_documents=max_documents)


def _load_huggingface_wiki(max_documents: int = 10000) -> List[Document]:
    """Load Wikipedia from HuggingFace datasets.

    Args:
        max_documents: Maximum documents

    Returns:
        List of Document objects
    """
    from agents.mcp.retrieval.index_manager import load_corpus_from_huggingface

    documents = list(
        load_corpus_from_huggingface(
            "wikipedia",
            split="train",
            id_field="id",
            content_field="text",
            max_documents=max_documents,
        )
    )

    logger.info(f"Loaded {len(documents)} Wikipedia documents")
    return documents


def load_deepresearch_corpus(
    samples: Optional[List[Any]] = None,
    domains: Optional[List[str]] = None,
    fetch_sources: bool = False,
    max_documents: int = 1000,
) -> List[Document]:
    """Load corpus for DeepResearch benchmark.

    DeepResearch evaluates research report generation. The corpus can include:
    - Domain-specific academic sources
    - Reference materials from samples
    - Web sources if fetch_sources=True

    Args:
        samples: List of DeepResearchSample objects
        domains: Filter to specific domains
        fetch_sources: Whether to fetch external sources
        max_documents: Maximum documents to load

    Returns:
        List of Document objects
    """
    # Load samples if not provided
    if samples is None:
        from evals.benchmarks.deepresearch.dataset import load_deepresearch_samples

        samples = list(load_deepresearch_samples(domains=domains))

    documents = []

    # Extract any embedded reference materials from samples
    for sample in samples:
        metadata = getattr(sample, "metadata", {})

        # Check for reference materials in metadata
        references = metadata.get("references", [])
        for i, ref in enumerate(references):
            if isinstance(ref, str) and ref.strip():
                doc_id = f"deepresearch_{sample.task_id}_ref_{i}"
                documents.append(
                    Document(
                        id=doc_id,
                        content=ref,
                        metadata={
                            "task_id": sample.task_id,
                            "domain": sample.domain,
                            "language": sample.language,
                            "source": "reference",
                        },
                    )
                )

    # If no references found and fetch_sources enabled, try to fetch domain sources
    if not documents and fetch_sources:
        logger.info("No embedded references found, fetching domain sources...")
        # This would require additional implementation for domain-specific sources
        # For now, log a warning
        logger.warning(
            "Domain source fetching not yet implemented. "
            "DeepResearch may work better with web search tools."
        )

    logger.info(f"Loaded {len(documents)} documents for DeepResearch")
    return documents


def create_corpus_from_texts(
    texts: List[str],
    ids: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[Document]:
    """Create a corpus from raw text strings.

    Utility function for creating custom corpora.

    Args:
        texts: List of text content
        ids: Optional document IDs (auto-generated if not provided)
        metadata: Optional metadata dicts for each document

    Returns:
        List of Document objects
    """
    documents = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue

        doc_id = ids[i] if ids and i < len(ids) else f"doc_{i}"
        doc_meta = metadata[i] if metadata and i < len(metadata) else {}

        documents.append(
            Document(id=doc_id, content=text, metadata=doc_meta)
        )

    return documents
