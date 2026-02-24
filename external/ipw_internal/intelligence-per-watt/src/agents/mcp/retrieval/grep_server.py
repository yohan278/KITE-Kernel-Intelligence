"""Grep-style retrieval server for fast regex/keyword pattern matching.

Based on Recursive Language Models (https://arxiv.org/pdf/2512.24601):
- Fast regex/keyword pattern matching over documents
- No indexing required - operates directly on text
- Useful for narrowing down lines of interest before semantic retrieval
- Very low latency, zero embedding cost
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..base import MCPToolResult
from .base import BaseRetrievalServer, Document, RetrievalResult


@dataclass
class GrepMatch:
    """A single grep match with context."""

    document_id: str
    line_number: int
    line_content: str
    context_before: List[str]
    context_after: List[str]
    match_start: int
    match_end: int


class GrepRetrievalServer(BaseRetrievalServer):
    """Fast regex/keyword retrieval without indexing.

    Inspired by Recursive Language Models paper - uses grep-style
    pattern matching to narrow down relevant lines/chunks.

    This server is useful for:
    - Finding specific keywords or patterns in documents
    - Narrowing down documents before semantic search
    - Low-latency, zero-cost retrieval

    Example:
        server = GrepRetrievalServer()
        server.index_documents([
            Document(id="1", content="Python is great for ML.\\nIt has many libraries."),
            Document(id="2", content="JavaScript is for web development."),
        ])

        result = server.execute("Python", pattern="Python.*ML")
        # Returns matches with context
    """

    def __init__(
        self,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        default_context_lines: int = 2,
        max_matches: int = 50,
    ):
        """Initialize grep retrieval server.

        Args:
            telemetry_collector: Energy monitor collector for telemetry
            event_recorder: EventRecorder for per-action tracking
            default_context_lines: Default lines of context before/after match
            max_matches: Maximum number of matches to return
        """
        super().__init__(
            name="retrieval:grep",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self.default_context_lines = default_context_lines
        self.max_matches = max_matches
        self._documents: Dict[str, Document] = {}

    def index_documents(self, documents: List[Document]) -> int:
        """Index documents for grep search.

        For grep, indexing simply stores the documents - no preprocessing needed.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents indexed
        """
        count = 0
        for doc in documents:
            self._documents[doc.id] = doc
            count += 1
        self._document_count = len(self._documents)
        return count

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._document_count = 0

    def _grep_documents(
        self,
        pattern: str,
        case_sensitive: bool = False,
        context_lines: int = 2,
    ) -> List[GrepMatch]:
        """Search documents using grep-style pattern matching.

        Args:
            pattern: Regex or literal pattern to search for
            case_sensitive: Whether search is case-sensitive
            context_lines: Number of context lines before/after match

        Returns:
            List of GrepMatch objects
        """
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error:
            # Fall back to literal search if pattern is invalid regex
            compiled_pattern = re.compile(re.escape(pattern), flags)

        for doc_id, doc in self._documents.items():
            lines = doc.content.split("\n")

            for line_num, line in enumerate(lines):
                match = compiled_pattern.search(line)
                if match:
                    # Get context lines
                    start_ctx = max(0, line_num - context_lines)
                    end_ctx = min(len(lines), line_num + context_lines + 1)

                    grep_match = GrepMatch(
                        document_id=doc_id,
                        line_number=line_num + 1,  # 1-indexed
                        line_content=line,
                        context_before=lines[start_ctx:line_num],
                        context_after=lines[line_num + 1 : end_ctx],
                        match_start=match.start(),
                        match_end=match.end(),
                    )
                    matches.append(grep_match)

        return matches

    def _format_grep_matches(self, matches: List[GrepMatch]) -> str:
        """Format grep matches for output.

        Args:
            matches: List of grep matches

        Returns:
            Formatted string with matches
        """
        if not matches:
            return "No matches found."

        lines = []
        current_doc = None

        for match in matches:
            # Add document header when switching documents
            if match.document_id != current_doc:
                if current_doc is not None:
                    lines.append("")
                lines.append(f"=== {match.document_id} ===")
                current_doc = match.document_id

            # Format context and match
            for ctx_line in match.context_before:
                lines.append(f"  {ctx_line}")

            # Highlight the matching line
            highlighted = (
                match.line_content[: match.match_start]
                + ">>>"
                + match.line_content[match.match_start : match.match_end]
                + "<<<"
                + match.line_content[match.match_end :]
            )
            lines.append(f"{match.line_number}: {highlighted}")

            for ctx_line in match.context_after:
                lines.append(f"  {ctx_line}")

            lines.append("---")

        return "\n".join(lines)

    def _matches_to_results(self, matches: List[GrepMatch]) -> List[RetrievalResult]:
        """Convert grep matches to retrieval results.

        Args:
            matches: List of grep matches

        Returns:
            List of retrieval results with documents
        """
        # Group matches by document
        doc_matches: Dict[str, List[GrepMatch]] = {}
        for match in matches:
            if match.document_id not in doc_matches:
                doc_matches[match.document_id] = []
            doc_matches[match.document_id].append(match)

        results = []
        for doc_id, doc_match_list in doc_matches.items():
            doc = self._documents.get(doc_id)
            if not doc:
                continue

            # Score based on number of matches
            score = len(doc_match_list)

            # Create highlights from matches
            highlights = []
            for m in doc_match_list[:3]:  # Limit highlights
                highlight = m.line_content.strip()
                if len(highlight) > 100:
                    highlight = highlight[:100] + "..."
                highlights.append(highlight)

            results.append(
                RetrievalResult(
                    document=doc,
                    score=float(score),
                    highlights=highlights,
                )
            )

        # Sort by score (number of matches)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute grep search.

        Args:
            prompt: Search query (used as pattern if no explicit pattern given)
            **params: Additional parameters:
                - pattern: Regex or literal pattern (default: prompt)
                - case_sensitive: Case-sensitive search (default: False)
                - context_lines: Lines of context (default: 2)
                - max_matches: Maximum matches to return (default: 50)
                - return_documents: Return full documents (default: False)

        Returns:
            MCPToolResult with matched content and metadata
        """
        pattern = params.get("pattern", prompt)
        case_sensitive = params.get("case_sensitive", False)
        context_lines = params.get("context_lines", self.default_context_lines)
        max_matches = params.get("max_matches", self.max_matches)
        return_documents = params.get("return_documents", False)

        # Perform grep search
        matches = self._grep_documents(pattern, case_sensitive, context_lines)

        # Limit matches
        limited_matches = matches[:max_matches]
        total_matches = len(matches)

        if return_documents:
            # Return as RetrievalResult format
            results = self._matches_to_results(limited_matches)
            content = self._format_results(results)
        else:
            # Return raw grep output
            content = self._format_grep_matches(limited_matches)

        if total_matches > max_matches:
            content += f"\n\n[Showing {max_matches} of {total_matches} matches]"

        return MCPToolResult(
            content=content,
            cost_usd=0.0,
            metadata={
                "tool": "retrieval:grep",
                "pattern": pattern,
                "num_matches": total_matches,
                "num_returned": len(limited_matches),
                "case_sensitive": case_sensitive,
            },
        )
