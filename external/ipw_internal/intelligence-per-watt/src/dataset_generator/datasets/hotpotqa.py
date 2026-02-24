"""HotpotQA dataset loader — multi-hop question answering."""

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


class HotpotQALoader(BaseDatasetLoader):
    """Load questions from hotpotqa/hotpot_qa (distractor config)."""

    def load(self, limit: Optional[int] = None) -> List[DatasetSample]:
        ds_lib = _require_datasets()
        ds = ds_lib.load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="train",
            streaming=True,
        )

        samples: List[DatasetSample] = []
        for row in ds:
            if limit is not None and len(samples) >= limit:
                break

            query = row.get("question", "")
            if not query:
                continue

            # Build context from the provided paragraphs
            context_titles = row.get("context", {}).get("title", [])
            context_sentences = row.get("context", {}).get("sentences", [])
            context_text = ""
            if context_titles and context_sentences:
                parts = []
                for title, sents in zip(context_titles, context_sentences):
                    parts.append(f"{title}: {''.join(sents)}")
                context_text = "\n".join(parts)

            samples.append(DatasetSample(
                query=query,
                expected_answer=row.get("answer", None),
                workload_type=self.workload_type(),
                metadata={
                    "source": "hotpotqa",
                    "supporting_facts": row.get("supporting_facts", {}),
                    "type": row.get("type", ""),
                    "level": row.get("level", ""),
                    "context": context_text,
                },
            ))

        return samples

    def workload_type(self) -> str:
        return "rag"

    def dataset_name(self) -> str:
        return "hotpotqa"
