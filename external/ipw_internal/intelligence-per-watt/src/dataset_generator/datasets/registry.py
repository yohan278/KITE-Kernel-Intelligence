"""Dataset registry — maps names to loader classes."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from dataset_generator.datasets.base import BaseDatasetLoader, DatasetSample
from dataset_generator.datasets.wildchat import WildChatLoader
from dataset_generator.datasets.openthoughts import OpenThoughtsLoader
from dataset_generator.datasets.hotpotqa import HotpotQALoader
from dataset_generator.datasets.agentdata import AgentDataLoader
from dataset_generator.datasets.swebench import SWEBenchLoader

DATASET_REGISTRY: Dict[str, Type[BaseDatasetLoader]] = {
    "wildchat": WildChatLoader,
    "openthoughts": OpenThoughtsLoader,
    "hotpotqa": HotpotQALoader,
    "agentdata": AgentDataLoader,
    "swebench": SWEBenchLoader,
}


def load_dataset(name: str, limit: Optional[int] = None) -> List[DatasetSample]:
    """Load samples from a named dataset.

    Args:
        name: Dataset name (must be in DATASET_REGISTRY).
        limit: Maximum number of samples to load.

    Returns:
        List of DatasetSample instances.

    Raises:
        KeyError: If the dataset name is not registered.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list_datasets()}"
        )
    loader = DATASET_REGISTRY[name]()
    return loader.load(limit=limit)


def list_datasets() -> List[str]:
    """Return sorted list of registered dataset names."""
    return sorted(DATASET_REGISTRY.keys())
