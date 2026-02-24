"""Pre-clean reference articles for DeepResearch-Bench.

Loads raw reference articles from HuggingFace, cleans them via Gemini
(removing citations, references, footnotes), and saves the results to
a local JSONL cache. This matches the original benchmark pipeline where
reference articles are pre-cleaned before scoring.

Usage:
    python -m evals.benchmarks.deepresearch.clean_articles [--max-workers 3] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .main import _clean_article

logger = logging.getLogger(__name__)

# Default cache location (same as dataset.py)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deepresearch_bench"
CLEANED_REFERENCE_FILENAME = "cleaned_reference.jsonl"


def _load_raw_references(cache_dir: Path) -> List[Dict]:
    """Load raw reference articles from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "'datasets' package is required. Install with: pip install datasets"
        )

    dataset = load_dataset(
        "muset-ai/DeepResearch-Bench-Dataset",
        data_files="generated_reports/openai-deepresearch.jsonl",
        split="train",
        cache_dir=str(cache_dir),
    )

    items = []
    for item in dataset:
        prompt = item.get("prompt", "")
        article = item.get("article", "")
        if prompt and article:
            items.append({"prompt": prompt, "article": article})

    logger.info(f"Loaded {len(items)} raw reference articles from HuggingFace")
    return items


def _load_queries(cache_dir: Path) -> Dict[str, str]:
    """Load query.jsonl to get language info per prompt."""
    from .dataset import QUERY_URL, _download_jsonl

    query_cache = cache_dir / "query.jsonl"
    queries = _download_jsonl(QUERY_URL, query_cache)
    return {q["prompt"]: q.get("language", "en") for q in queries if "prompt" in q}


def _clean_single(
    item: Dict,
    prompt_to_language: Dict[str, str],
    lock: threading.Lock,
    output_file: Path,
    processed_prompts: set,
) -> Optional[str]:
    """Clean a single reference article and append to the output file."""
    prompt = item["prompt"]
    article = item["article"]
    language = prompt_to_language.get(prompt, "en")

    if prompt in processed_prompts:
        return None

    try:
        cleaned = _clean_article(article, language=language)
    except Exception as e:
        logger.error(f"Failed to clean article (prompt={prompt[:60]}...): {e}")
        return None

    result = {"prompt": prompt, "article": cleaned}
    with lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        processed_prompts.add(prompt)

    return prompt


def clean_reference_articles(
    cache_dir: Optional[Path] = None,
    max_workers: int = 3,
    force: bool = False,
) -> Path:
    """Pre-clean all reference articles and save to cache.

    Args:
        cache_dir: Cache directory. Defaults to ~/.cache/deepresearch_bench
        max_workers: Number of concurrent cleaning threads.
        force: If True, re-clean even if the cleaned file exists.

    Returns:
        Path to the cleaned reference JSONL file.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file = cache_dir / CLEANED_REFERENCE_FILENAME

    # Load already-processed prompts from existing output
    processed_prompts: set = set()
    if output_file.exists() and not force:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed_prompts.add(json.loads(line)["prompt"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Load raw references and language info
    raw_refs = _load_raw_references(cache_dir)
    prompt_to_language = _load_queries(cache_dir)

    if force and output_file.exists():
        output_file.unlink()
        processed_prompts.clear()

    to_process = [r for r in raw_refs if r["prompt"] not in processed_prompts]

    if not to_process:
        logger.info(
            f"All {len(processed_prompts)} reference articles already cleaned -> {output_file}"
        )
        return output_file

    logger.info(
        f"Cleaning {len(to_process)} reference articles "
        f"({len(processed_prompts)} already done) with {max_workers} workers..."
    )

    lock = threading.Lock()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _clean_single, item, prompt_to_language, lock, output_file, processed_prompts
            ): item
            for item in to_process
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                completed += 1
                if completed % 10 == 0 or completed == len(to_process):
                    logger.info(f"  Cleaned {completed}/{len(to_process)} articles")

    total = len(processed_prompts)
    logger.info(f"Done. {total} cleaned reference articles saved to {output_file}")
    return output_file


def get_cleaned_reference_path(cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the path to the cleaned reference file if it exists."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    path = cache_dir / CLEANED_REFERENCE_FILENAME
    return path if path.exists() else None


def load_cleaned_references(cache_dir: Optional[Path] = None) -> Dict[str, str]:
    """Load pre-cleaned reference articles keyed by prompt text.

    Returns empty dict if no cleaned file exists.
    """
    path = get_cleaned_reference_path(cache_dir)
    if path is None:
        return {}

    refs: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    article = item.get("article", "")
                    if prompt and article:
                        refs[prompt] = article
                except json.JSONDecodeError:
                    pass

    return refs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Pre-clean reference articles for DeepResearch-Bench"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: ~/.cache/deepresearch_bench)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of concurrent cleaning threads (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-clean all articles even if cleaned file exists",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    clean_reference_articles(
        cache_dir=cache_dir,
        max_workers=args.max_workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
