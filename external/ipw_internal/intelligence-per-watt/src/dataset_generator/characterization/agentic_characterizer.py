"""Agentic workload characterizer — analyses agent trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from dataset_generator.characterization.base import BaseCharacterizer
from dataset_generator.characterization.registry import register_characterizer
from dataset_generator.characterization.tokenizer import FastTokenCounter
from dataset_generator.datasets.agentdata import AgentDataLoader
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


class AgenticCharacterizer(BaseCharacterizer):
    """Characterize agentic workloads from neulab/agent-data-collection."""

    def characterize(self, limit: Optional[int] = None) -> WorkloadProfile:
        loader = AgentDataLoader()
        trajectories = loader.load_trajectories(limit=limit)
        tc = FastTokenCounter()

        step_counts = []
        input_tokens = []  # task descriptions
        action_tokens = []  # agent actions (output tokens)
        observation_tokens = []  # tool results
        domain_counts: Counter[str] = Counter()

        input_by_pos: dict[int, list[float]] = {}
        output_by_pos: dict[int, list[float]] = {}

        for traj in trajectories:
            step_counts.append(float(len(traj.steps)))
            input_tokens.append(float(tc.count(traj.task)))
            domain = traj.metadata.get("domain", "unknown")
            domain_counts[domain] += 1

            for i, step in enumerate(traj.steps):
                act_tok = float(tc.count(step.action))
                obs_tok = float(tc.count(step.observation))
                action_tokens.append(act_tok)
                observation_tokens.append(obs_tok)
                output_by_pos.setdefault(i, []).append(act_tok)
                input_by_pos.setdefault(i, []).append(obs_tok)

        # Max context: task + all action/observation pairs
        max_ctx = 0
        for traj in trajectories:
            ctx = tc.count(traj.task)
            for step in traj.steps:
                ctx += tc.count(step.action) + tc.count(step.observation)
            if ctx > max_ctx:
                max_ctx = ctx

        total = sum(domain_counts.values()) or 1
        domain_mix = {k: v / total for k, v in domain_counts.most_common(10)}

        # Every step is a tool call in agentic workflows
        tool_call_prob = 1.0 if trajectories else 0.0

        return WorkloadProfile(
            workload_type=self.workload_type(),
            source_dataset=self.dataset_name(),
            n_samples=len(trajectories),
            turns_or_steps_dist=FittedDistribution.fit(step_counts) if step_counts else None,
            input_tokens_dist=FittedDistribution.fit(input_tokens) if input_tokens else None,
            answer_tokens_dist=FittedDistribution.fit(action_tokens) if action_tokens else None,
            input_tokens_by_position={
                pos: FittedDistribution.fit(vals)
                for pos, vals in input_by_pos.items()
            },
            output_tokens_by_position={
                pos: FittedDistribution.fit(vals)
                for pos, vals in output_by_pos.items()
            },
            tool_call_probability=tool_call_prob,
            tool_call_tokens_dist=FittedDistribution.fit(observation_tokens) if observation_tokens else None,
            max_context_observed=max_ctx,
            domain_mix=domain_mix,
        )

    def workload_type(self) -> str:
        return "agentic"

    def dataset_name(self) -> str:
        return "agentdata"


register_characterizer("agentdata", AgenticCharacterizer)
