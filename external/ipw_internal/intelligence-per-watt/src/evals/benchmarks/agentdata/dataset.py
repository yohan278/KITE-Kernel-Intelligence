"""AgentData dataset loader for eval benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class AgentDataSample:
    """A single AgentData benchmark sample."""

    original_index: int
    task: str
    expected_steps: int = 0
    expected_answer: Optional[str] = None
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_agentdata_samples(
    *,
    limit: Optional[int] = None,
    domains: Optional[List[str]] = None,
) -> Iterator[AgentDataSample]:
    """Load AgentData samples wrapping the existing AgentDataLoader."""
    from dataset_generator.datasets.agentdata import AgentDataLoader

    loader = AgentDataLoader()
    trajectories = loader.load_trajectories(limit=limit)

    for idx, traj in enumerate(trajectories):
        domain = traj.metadata.get("domain", "")
        if domains and domain not in domains:
            continue

        yield AgentDataSample(
            original_index=idx,
            task=traj.task,
            expected_steps=len(traj.steps),
            expected_answer=traj.final_answer,
            domain=domain,
            metadata=traj.metadata,
        )
