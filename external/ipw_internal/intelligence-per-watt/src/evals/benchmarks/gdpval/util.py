"""
GDPval consolidation helpers:
- building the HuggingFace upload folder
- writing parquet tables
- optional automatic upload
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from datasets import load_dataset
from datetime import datetime
from huggingface_hub import HfApi, HfFolder

HF_DATASET_PATH = "openai/gdpval"


@dataclass
class GDPvalResult:
    task_id: str
    deliverable_text: str
    deliverable_files: Dict[str, bytes]  # relative_path -> bytes


def save_updated_table(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Updated GDPval table saved to {output_path}")


def load_hf_table() -> pd.DataFrame:
    dataset = load_dataset(HF_DATASET_PATH, split="train")
    return dataset.to_pandas()  # type: ignore[return-value]


def build_updated_table(
    results: Sequence[GDPvalResult],
) -> pd.DataFrame:
    table = load_hf_table()
    table["deliverable_text"] = ""
    table["deliverable_files"] = [[] for _ in range(len(table))]

    for result in results:
        mask = table["task_id"] == result.task_id
        if mask.sum() != 1:
            raise ValueError(
                f"Expected exactly one row for task_id={result.task_id}, got {mask.sum()}"
            )
        idx = table[mask].index[0]
        table.at[idx, "deliverable_text"] = result.deliverable_text
        table.at[idx, "deliverable_files"] = [
            f"deliverable_files/{result.task_id}/{name}"
            for name in result.deliverable_files.keys()
        ]
    return table


def write_deliverable_files(
    folder: Path,
    results: Sequence[GDPvalResult],
) -> None:
    for result in results:
        sample_dir = folder / "deliverable_files" / result.task_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for relative_name, content in result.deliverable_files.items():
            target = sample_dir / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def prepare_hf_folder(
    results: Sequence[GDPvalResult],
    *,
    base_dir: Path | None = None,
) -> Path:
    timestamp = datetime.now().isoformat().replace("+", "plus")
    base = base_dir or Path(__file__).resolve().parent / "gdpval_hf_upload"
    folder = base / f"{timestamp}_gdpval"
    (folder / "data").mkdir(parents=True, exist_ok=True)

    write_deliverable_files(folder, results)
    table = build_updated_table(results)
    save_updated_table(table, folder / "data" / "train-00000-of-00001.parquet")

    return folder


def upload_folder_to_hf(folder: Path) -> str:
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "HF token not found. Run `hf auth login` or set HF_TOKEN environment variable."
        )

    api = HfApi()
    owner = api.whoami(token)["name"]
    repo_id = f"{owner}/{folder.name}"

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder))
    url = f"{api.endpoint.rstrip('/')}/datasets/{repo_id}"
    print(f"Uploaded GDPval deliverables to {url}")
    return url