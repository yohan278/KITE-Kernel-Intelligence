"""SWE-bench dataset loader — real coding task instances from HuggingFace."""

from __future__ import annotations

from typing import List, Optional

from dataset_generator.datasets.base import BaseDatasetLoader, DatasetSample


def _require_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        )


class SWEBenchLoader(BaseDatasetLoader):
    """Load coding task instances from princeton-nlp/SWE-bench_Verified."""

    def load(self, limit: Optional[int] = None) -> List[DatasetSample]:
        ds_lib = _require_datasets()
        ds = ds_lib.load_dataset(
            "princeton-nlp/SWE-bench_Verified",
            split="test",
            streaming=True,
        )

        samples: List[DatasetSample] = []
        for row in ds:
            if limit is not None and len(samples) >= limit:
                break

            problem = row.get("problem_statement", "")
            if not problem:
                continue

            samples.append(DatasetSample(
                query=problem,
                expected_answer=row.get("patch", None),
                workload_type=self.workload_type(),
                metadata={
                    "source": "swebench",
                    "instance_id": row.get("instance_id", ""),
                    "repo": row.get("repo", ""),
                },
            ))

        return samples

    def workload_type(self) -> str:
        return "coding"

    def dataset_name(self) -> str:
        return "swebench"
