"""Agent data loader — real agentic task trajectories from HuggingFace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


@dataclass
class AgentStep:
    """A single step in an agentic trajectory."""

    action: str
    observation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectorySample:
    """A full agentic trajectory with multiple steps."""

    task: str
    steps: List[AgentStep]
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _parse_std_row(row: Dict[str, Any]) -> tuple:
    """Parse a row in the standardized (std) format.

    The std split has: content (list of messages), details, id.
    Each message has: source (user/None), content, class_, etc.

    Returns (task, steps, last_agent_content).
    """
    content = row.get("content", [])
    if not isinstance(content, list) or not content:
        return "", [], None

    task = ""
    steps: List[AgentStep] = []
    last_agent_content = None

    # Collect alternating agent actions and user observations
    pending_action = None
    for msg in content:
        source = msg.get("source")
        text = msg.get("content") or ""
        class_ = msg.get("class_", "")

        if source == "user":
            if not task:
                # First user message is the task
                task = text
            elif pending_action is not None:
                # This user message is the observation for the pending action
                steps.append(AgentStep(
                    action=pending_action,
                    observation=text,
                    metadata={"class": class_},
                ))
                pending_action = None
        else:
            # Agent message (source is None or 'assistant')
            action_text = text
            # For api_action, use function field if content is empty
            if not action_text and msg.get("function"):
                action_text = str(msg.get("function"))
            if action_text:
                if pending_action is not None:
                    # Two agent actions in a row — flush previous
                    steps.append(AgentStep(
                        action=pending_action,
                        observation="",
                    ))
                pending_action = action_text
                last_agent_content = action_text

    # Flush any remaining pending action
    if pending_action is not None:
        steps.append(AgentStep(action=pending_action, observation=""))

    return task, steps, last_agent_content


class AgentDataLoader(BaseDatasetLoader):
    """Load agentic trajectories from neulab/agent-data-collection."""

    # Configs with structured agentic trajectories, ordered by diversity
    CONFIGS = [
        "agenttuning_alfworld",   # embodied/planning (household)
        "agenttuning_db",         # knowledge/reasoning (database)
        "agenttuning_webshop",    # interactive decision-making (shopping)
        "orca_agentinstruct",     # general instruction following
        "codeactinstruct",        # code-agentic bridge
    ]

    def load(self, limit: Optional[int] = None) -> List[DatasetSample]:
        ds_lib = _require_datasets()
        trajectories = self.load_trajectories(limit=limit)

        samples: List[DatasetSample] = []
        for traj in trajectories:
            samples.append(DatasetSample(
                query=traj.task,
                expected_answer=traj.final_answer,
                workload_type=self.workload_type(),
                metadata={
                    "source": "agentdata",
                    "num_steps": len(traj.steps),
                    "domain": traj.metadata.get("domain", ""),
                },
            ))

        return samples

    def load_trajectories(
        self, limit: Optional[int] = None
    ) -> List[TrajectorySample]:
        """Load full agent trajectories with step-level detail.

        Returns TrajectorySample instances with all steps preserved,
        useful for characterizing step counts and per-step token lengths.
        """
        ds_lib = _require_datasets()

        trajectories: List[TrajectorySample] = []
        per_config_limit = (limit // len(self.CONFIGS) + 1) if limit else None

        for config_name in self.CONFIGS:
            config_count = 0
            try:
                ds = ds_lib.load_dataset(
                    "neulab/agent-data-collection",
                    name=config_name,
                    split="std",
                    streaming=True,
                )
            except Exception:
                try:
                    # Fallback to train split if std doesn't exist
                    ds = ds_lib.load_dataset(
                        "neulab/agent-data-collection",
                        name=config_name,
                        split="train",
                        streaming=True,
                    )
                except Exception:
                    continue

            for row in ds:
                if limit is not None and len(trajectories) >= limit:
                    break
                if per_config_limit is not None and config_count >= per_config_limit:
                    break

                task, steps, last_agent = _parse_std_row(row)

                if not task or not steps:
                    continue

                config_count += 1
                trajectories.append(TrajectorySample(
                    task=task,
                    steps=steps,
                    final_answer=last_agent,
                    metadata={
                        "source": "agentdata",
                        "config": config_name,
                        "domain": config_name,
                    },
                ))

            if limit is not None and len(trajectories) >= limit:
                break

        return trajectories

    def workload_type(self) -> str:
        return "agentic"

    def dataset_name(self) -> str:
        return "agentdata"
