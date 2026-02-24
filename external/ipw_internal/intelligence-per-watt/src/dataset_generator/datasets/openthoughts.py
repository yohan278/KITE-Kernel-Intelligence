"""OpenThoughts dataset loader — reasoning problems with chain-of-thought."""

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


class OpenThoughtsLoader(BaseDatasetLoader):
    """Load reasoning problems from open-thoughts/OpenThoughts-114k."""

    def load(self, limit: Optional[int] = None) -> List[DatasetSample]:
        ds_lib = _require_datasets()
        ds = ds_lib.load_dataset(
            "open-thoughts/OpenThoughts-114k",
            split="train",
            streaming=True,
        )

        samples: List[DatasetSample] = []
        for row in ds:
            if limit is not None and len(samples) >= limit:
                break

            # The dataset stores data in 'conversations' list with from/value pairs
            conversations = row.get("conversations", [])
            query = row.get("problem", "")
            solution = row.get("solution", None)
            reasoning = ""

            if not query:
                # Extract query from first user turn in conversations
                for turn in conversations:
                    if turn.get("from") == "user":
                        query = turn.get("value", "")
                        break

            if not query:
                continue

            # Extract assistant response and reasoning from conversations
            for turn in conversations:
                if turn.get("from") == "assistant":
                    value = turn.get("value", "")
                    if not solution:
                        solution = value
                    # Reasoning may be in <think>...</think> or <|begin_of_thought|>...<|end_of_thought|>
                    for open_tag, close_tag in [
                        ("<think>", "</think>"),
                        ("<|begin_of_thought|>", "<|end_of_thought|>"),
                    ]:
                        if open_tag in value:
                            start = value.find(open_tag) + len(open_tag)
                            end = value.find(close_tag)
                            if end > start:
                                reasoning = value[start:end].strip()
                                break
                    break

            metadata = {
                "source": "openthoughts",
                "domain": row.get("domain", ""),
            }
            if reasoning:
                metadata["deepseek_reasoning"] = reasoning

            samples.append(DatasetSample(
                query=query,
                expected_answer=solution,
                workload_type=self.workload_type(),
                metadata=metadata,
            ))

        return samples

    def workload_type(self) -> str:
        return "reasoning"

    def dataset_name(self) -> str:
        return "openthoughts"
