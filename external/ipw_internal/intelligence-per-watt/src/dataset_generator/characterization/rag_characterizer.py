"""RAG workload characterizer — analyses HotpotQA multi-hop questions."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from dataset_generator.characterization.base import BaseCharacterizer
from dataset_generator.characterization.registry import register_characterizer
from dataset_generator.characterization.tokenizer import FastTokenCounter
from dataset_generator.datasets.hotpotqa import HotpotQALoader
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


class RAGCharacterizer(BaseCharacterizer):
    """Characterize RAG workloads from HotpotQA."""

    def characterize(self, limit: Optional[int] = None) -> WorkloadProfile:
        loader = HotpotQALoader()
        samples = loader.load(limit=limit)
        tc = FastTokenCounter()

        input_tokens = []
        answer_tokens = []
        context_tokens = []
        type_counts: Counter[str] = Counter()

        for s in samples:
            input_tokens.append(float(tc.count(s.query)))
            if s.expected_answer:
                answer_tokens.append(float(tc.count(s.expected_answer)))

            context = s.metadata.get("context", "")
            if context:
                context_tokens.append(float(tc.count(context)))

            qtype = s.metadata.get("type", "unknown")
            type_counts[qtype] += 1

        # Max context: question + retrieved context + answer
        max_ctx = 0
        for s in samples:
            ctx = tc.count(s.query)
            context = s.metadata.get("context", "")
            if context:
                ctx += tc.count(context)
            if s.expected_answer:
                ctx += tc.count(s.expected_answer)
            if ctx > max_ctx:
                max_ctx = ctx

        total = sum(type_counts.values()) or 1
        domain_mix = {k: v / total for k, v in type_counts.most_common(10)}

        # Context retrieval tokens model the "tool call" aspect of RAG
        tool_call_prob = len(context_tokens) / max(len(samples), 1)

        return WorkloadProfile(
            workload_type=self.workload_type(),
            source_dataset=self.dataset_name(),
            n_samples=len(samples),
            input_tokens_dist=FittedDistribution.fit(input_tokens) if input_tokens else None,
            answer_tokens_dist=FittedDistribution.fit(answer_tokens) if answer_tokens else None,
            tool_call_probability=tool_call_prob,
            tool_call_tokens_dist=FittedDistribution.fit(context_tokens) if context_tokens else None,
            max_context_observed=max_ctx,
            domain_mix=domain_mix,
        )

    def workload_type(self) -> str:
        return "rag"

    def dataset_name(self) -> str:
        return "hotpotqa"


register_characterizer("hotpotqa", RAGCharacterizer)
