# Stateful web browser simulation for BrowseComp benchmark.
# Provides browser-like tools: web_search, web_open, web_find, web_click.
# Supports live (network) or cached (replay) modes for reproducibility.

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ipw" / "browsecomp" / "web_cache"
MAX_PAGE_TEXT_LENGTH = 15000  # ~4k tokens, reduced from 50k to avoid rate limits
MAX_LINKS_PER_PAGE = 30
REQUEST_TIMEOUT = 30


class WebToolCache:
    """Cache layer for web tool results (search and pages)."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.search_dir = self.cache_dir / "search"
        self.pages_dir = self.cache_dir / "pages"
        self.search_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_key(self, *args) -> str:
        return hashlib.sha256("|".join(str(a) for a in args).encode()).hexdigest()
    
    def get_search(self, query: str, k: int, provider: str) -> Optional[dict]:
        path = self.search_dir / f"{self._hash_key(query, k, provider)}.json"
        return json.loads(path.read_text()) if path.exists() else None
    
    def put_search(self, query: str, k: int, provider: str, result: dict) -> None:
        path = self.search_dir / f"{self._hash_key(query, k, provider)}.json"
        path.write_text(json.dumps({"query": query, "k": k, "provider": provider, "timestamp": time.time(), "result": result}, indent=2))
    
    def get_page(self, url: str) -> Optional[dict]:
        path = self.pages_dir / f"{self._hash_key(url)}.json"
        return json.loads(path.read_text()) if path.exists() else None
    
    def put_page(self, url: str, result: dict) -> None:
        path = self.pages_dir / f"{self._hash_key(url)}.json"
        path.write_text(json.dumps({"url": url, "timestamp": time.time(), "result": result}, indent=2))


class CacheMissError(Exception):
    """Raised when cached mode cannot find a cached result."""


def _extract_text_trafilatura(html: str) -> tuple[str, str]:
    """Extract main content from HTML using trafilatura. Returns (title, text)."""
    try:
        import trafilatura
        text = trafilatura.extract(html, include_links=False, include_images=False, include_tables=True)
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        return title, text or ""
    except ImportError:
        logger.warning("trafilatura not installed, falling back to BeautifulSoup")
        return _extract_text_bs4(html)


def _extract_text_bs4(html: str) -> tuple[str, str]:
    """Fallback text extraction using BeautifulSoup. Returns (title, text)."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for el in soup(["script", "style", "nav", "footer", "header"]):
            el.decompose()
        title = soup.title.string if soup.title else ""
        return title or "", soup.get_text(separator="\n", strip=True)
    except ImportError:
        logger.error("Neither trafilatura nor beautifulsoup4 installed")
        return "", html[:MAX_PAGE_TEXT_LENGTH]


def _extract_links(html: str, base_url: str) -> List[Dict[str, str]]:
    """Extract links from HTML."""
    try:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            text = a.get_text(strip=True)
            links.append({"text": text[:100] if text else "", "url": urljoin(base_url, href)})
            if len(links) >= MAX_LINKS_PER_PAGE:
                break
        return links
    except ImportError:
        return []


class BrowseCompTools:
    """Web browsing tools with live and cached modes."""
    
    def __init__(
        self,
        mode: Literal["live", "cached"] = "live",
        cache_dir: Optional[Path] = None,
        search_provider: str = "tavily",
        max_search_results: int = 5,
    ):
        self.mode = mode
        self.cache = WebToolCache(cache_dir)
        self.search_provider = search_provider
        self.max_search_results = max_search_results
        if mode == "live":
            self._validate_api_keys()
    
    def _validate_api_keys(self) -> None:
        key_env = "TAVILY_API_KEY" if self.search_provider == "tavily" else "SERPER_API_KEY"
        if not os.environ.get(key_env):
            logger.warning(f"{key_env} not set. Web search will fail.")
    
    def web_search(self, query: str, k: Optional[int] = None) -> dict:
        """Search the web. Returns dict with query, results, provider, cached."""
        k = k if k is not None else self.max_search_results
        cached = self.cache.get_search(query, k, self.search_provider)
        if cached:
            cached["result"]["cached"] = True
            return cached["result"]
        if self.mode == "cached":
            raise CacheMissError(f"Search not in cache: {query[:50]}...")
        
        result = self._live_search(query, k)
        self.cache.put_search(query, k, self.search_provider, result)
        return result
    
    def _live_search(self, query: str, k: int) -> dict:
        if self.search_provider == "tavily":
            return self._tavily_search(query, k)
        elif self.search_provider == "serper":
            return self._serper_search(query, k)
        raise ValueError(f"Unknown search provider: {self.search_provider}")
    
    def _tavily_search(self, query: str, k: int) -> dict:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"query": query, "results": [], "provider": "tavily", "error": "TAVILY_API_KEY not set", "cached": False}
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=k, search_depth="basic")
            results = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:500]} for r in response.get("results", [])]
            return {"query": query, "results": results, "provider": "tavily", "cached": False}
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return {"query": query, "results": [], "provider": "tavily", "error": str(e), "cached": False}
    
    def _serper_search(self, query: str, k: int) -> dict:
        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            return {"query": query, "results": [], "provider": "serper", "error": "SERPER_API_KEY not set", "cached": False}
        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": k},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            results = [{"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")[:500]} for r in data.get("organic", [])[:k]]
            return {"query": query, "results": results, "provider": "serper", "cached": False}
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return {"query": query, "results": [], "provider": "serper", "error": str(e), "cached": False}
    
    def web_open(self, url: str) -> dict:
        """Open a URL and extract content. Returns dict with url, title, text, links, cached."""
        cached = self.cache.get_page(url)
        if cached:
            cached["result"]["cached"] = True
            return cached["result"]
        if self.mode == "cached":
            raise CacheMissError(f"Page not in cache: {url[:50]}...")
        
        result = self._live_fetch(url)
        self.cache.put_page(url, result)
        return result
    
    def _live_fetch(self, url: str) -> dict:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            html = response.text
            title, text = _extract_text_trafilatura(html)
            links = _extract_links(html, url)
            if len(text) > MAX_PAGE_TEXT_LENGTH:
                text = text[:MAX_PAGE_TEXT_LENGTH] + "\n... (truncated)"
            return {"url": url, "title": title, "text": text, "links": links, "cached": False}
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {"url": url, "title": "", "text": f"Error fetching page: {e}", "links": [], "error": str(e), "cached": False}
    
    def web_find(self, page: dict, pattern: str, max_matches: int = 20) -> dict:
        """Search within page content for a pattern. Returns dict with pattern, matches, count."""
        text = page.get("text", "")
        matches = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                start, end = max(0, match.start() - 100), min(len(text), match.end() + 100)
                context = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                matches.append({"match": match.group(), "context": context, "position": match.start()})
                if len(matches) >= max_matches:
                    break
        except re.error:
            lower_text, lower_pattern, pos = text.lower(), pattern.lower(), 0
            while len(matches) < max_matches:
                idx = lower_text.find(lower_pattern, pos)
                if idx == -1:
                    break
                start, end = max(0, idx - 100), min(len(text), idx + len(pattern) + 100)
                context = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                matches.append({"match": text[idx:idx + len(pattern)], "context": context, "position": idx})
                pos = idx + 1
        return {"pattern": pattern, "matches": matches, "count": len(matches)}
    
    def web_click(self, page: dict, link_index: int) -> dict:
        """Follow a link from a page. Returns result from web_open for the linked page."""
        links = page.get("links", [])
        if not links:
            return {"url": "", "title": "", "text": "No links available on this page.", "links": [], "error": "no_links", "cached": False}
        if link_index < 0 or link_index >= len(links):
            return {"url": "", "title": "", "text": f"Invalid link index {link_index}. Available: 0-{len(links)-1}", "links": [], "error": "invalid_index", "cached": False}
        return self.web_open(links[link_index]["url"])
    
    def get_tools(self) -> List[Callable]:
        """Get list of tool functions for use with orchestrators."""
        return [self.web_search, self.web_open, self.web_find, self.web_click]


def create_browsecomp_tools(
    mode: Literal["live", "cached"] = "live",
    cache_dir: Optional[Path] = None,
    search_provider: str = "tavily",
    max_search_results: int = 5,
) -> List[Callable]:
    """Create BrowseComp web tools. Returns [web_search, web_open, web_find, web_click]."""
    return BrowseCompTools(mode=mode, cache_dir=cache_dir, search_provider=search_provider, max_search_results=max_search_results).get_tools()
