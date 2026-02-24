from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Iterator, List, Optional

from datasets import load_dataset

from .types import EvidenceItem, FinanceBenchSample

logger = logging.getLogger(__name__)

DATASET_NAME = "PatronusAI/financebench"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ipw" / "financebench"


def load_financebench_samples(
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> List[FinanceBenchSample]:
    """Load FinanceBench samples from HuggingFace dataset."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading FinanceBench dataset from HuggingFace: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=str(cache_dir))
    logger.info(f"Dataset contains {len(dataset)} samples")
    
    rows = list(dataset)
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    
    samples = []
    for idx, row in enumerate(rows):
        try:
            uid = generate_uid(row, idx)
            evidence_items = parse_evidence(row.get("evidence", []))
            
            samples.append(FinanceBenchSample(
                uid=uid,
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                company=row.get("company", ""),
                doc_name=row.get("doc_name", ""),
                question_type=row.get("question_type", ""),
                question_reasoning=row.get("question_reasoning", ""),
                justification=row.get("justification", ""),
                evidence=tuple(evidence_items),
            ))
        except Exception as e:
            logger.warning(f"Failed to process row {idx}: {e}")
    
    logger.info(f"Loaded {len(samples)} FinanceBench samples")
    return samples


def iter_financebench_samples(
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> Iterator[FinanceBenchSample]:
    """Iterate over FinanceBench samples."""
    yield from load_financebench_samples(limit=limit, shuffle=shuffle, seed=seed, cache_dir=cache_dir)


def generate_uid(row: dict, idx: int) -> str:
    """Generate a stable unique ID for a dataset row."""
    fb_id = row.get("financebench_id", "")
    if fb_id:
        return f"fb_{fb_id}"
    content = f"{row.get('question', '')}{row.get('answer', '')}"
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"fb_{idx:04d}_{hash_val}"


def parse_evidence(evidence_list: list) -> List[EvidenceItem]:
    """Parse evidence list from dataset."""
    items = []
    if not evidence_list:
        return items
    for ev in evidence_list:
        if isinstance(ev, dict):
            items.append(EvidenceItem(
                evidence_text=ev.get("evidence_text", ""),
                evidence_doc_name=ev.get("evidence_doc_name", ""),
                evidence_page_num=ev.get("evidence_page_num", 0),
                evidence_text_full_page=ev.get("evidence_text_full_page", ""),
            ))
    return items


def get_dataset_info(cache_dir: Optional[Path] = None) -> dict:
    """Get information about the FinanceBench dataset."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=str(cache_dir))
    return {
        "total_samples": len(dataset),
        "columns": dataset.column_names,
        "dataset_name": DATASET_NAME,
    }
