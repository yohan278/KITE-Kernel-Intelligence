# Dataset loading for BrowseComp benchmark.

import base64
import hashlib
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from .types import BrowseCompSample

DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ipw" / "browsecomp"


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR cipher."""
    data = base64.b64decode(ciphertext_b64)
    key = hashlib.sha256(password.encode()).digest()
    key = key * (len(data) // len(key) + 1)  # extend key to cover data
    return bytes(a ^ b for a, b in zip(data, key)).decode()


def decrypt_row(row: dict) -> tuple[str, str]:
    """Decrypt a BrowseComp dataset row. Returns (question, answer)."""
    canary, problem, answer = row["canary"], row["problem"], row["answer"]
    return decrypt(problem, canary), decrypt(answer, canary)


def load_browsecomp_samples(
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> List[BrowseCompSample]:
    """Load BrowseComp samples from the dataset."""
    csv_path = (cache_dir or DEFAULT_CACHE_DIR) / "browse_comp_test_set.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_bytes(requests.get(DATASET_URL, timeout=60).content)
    
    rows = pd.read_csv(csv_path).to_dict(orient="records")
    if shuffle:
        random.Random(seed).shuffle(rows)
    rows = rows[:limit] if limit else rows
    
    samples = []
    for idx, row in enumerate(rows):
        question, answer = decrypt_row(row)
        content = f"{row['problem']}{row['answer']}"
        uid = f"bc_{idx:04d}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        samples.append(BrowseCompSample(uid=uid, question=question, answer=answer))
    return samples
