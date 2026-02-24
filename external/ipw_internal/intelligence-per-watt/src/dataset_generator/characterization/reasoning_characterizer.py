"""Reasoning workload characterizer — analyses OpenThoughts problems."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from dataset_generator.characterization.base import BaseCharacterizer
from dataset_generator.characterization.registry import register_characterizer
from dataset_generator.characterization.tokenizer import FastTokenCounter
from dataset_generator.datasets.openthoughts import OpenThoughtsLoader
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


class ReasoningCharacterizer(BaseCharacterizer):
    """Characterize reasoning workloads from OpenThoughts."""

    def characterize(self, limit: Optional[int] = None) -> WorkloadProfile:
        loader = OpenThoughtsLoader()
        samples = loader.load(limit=limit)
        tc = FastTokenCounter()

        input_tokens = []
        answer_tokens = []
        thinking_tokens = []
        domain_counts: Counter[str] = Counter()

        for s in samples:
            input_tokens.append(float(tc.count(s.query)))
            if s.expected_answer:
                answer_tokens.append(float(tc.count(s.expected_answer)))

            domain = s.metadata.get("domain", "unknown")
            domain_counts[domain] += 1

            # Extract thinking tokens from deepseek_reasoning if present
            reasoning = s.metadata.get("deepseek_reasoning", "")
            if reasoning:
                thinking_tokens.append(float(tc.count(reasoning)))

        # Max context: longest problem + answer + reasoning
        max_ctx = 0
        for s in samples:
            ctx = tc.count(s.query)
            if s.expected_answer:
                ctx += tc.count(s.expected_answer)
            reasoning = s.metadata.get("deepseek_reasoning", "")
            if reasoning:
                ctx += tc.count(reasoning)
            if ctx > max_ctx:
                max_ctx = ctx

        total = sum(domain_counts.values()) or 1
        domain_mix = {k: v / total for k, v in domain_counts.most_common(10)}

        return WorkloadProfile(
            workload_type=self.workload_type(),
            source_dataset=self.dataset_name(),
            n_samples=len(samples),
            input_tokens_dist=FittedDistribution.fit(input_tokens) if input_tokens else None,
            answer_tokens_dist=FittedDistribution.fit(answer_tokens) if answer_tokens else None,
            thinking_tokens_dist=FittedDistribution.fit(thinking_tokens) if thinking_tokens else None,
            max_context_observed=max_ctx,
            domain_mix=domain_mix,
        )

    def workload_type(self) -> str:
        return "reasoning"

    def dataset_name(self) -> str:
        return "openthoughts"


register_characterizer("openthoughts", ReasoningCharacterizer)
