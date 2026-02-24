"""Coding workload characterizer — analyses SWE-bench task instances."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from dataset_generator.characterization.base import BaseCharacterizer
from dataset_generator.characterization.registry import register_characterizer
from dataset_generator.characterization.tokenizer import FastTokenCounter
from dataset_generator.datasets.swebench import SWEBenchLoader
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


class CodingCharacterizer(BaseCharacterizer):
    """Characterize coding workloads from SWE-bench."""

    def characterize(self, limit: Optional[int] = None) -> WorkloadProfile:
        loader = SWEBenchLoader()
        samples = loader.load(limit=limit)
        tc = FastTokenCounter()

        input_tokens = []
        answer_tokens = []
        repo_counts: Counter[str] = Counter()

        for s in samples:
            input_tokens.append(float(tc.count(s.query)))
            if s.expected_answer:
                answer_tokens.append(float(tc.count(s.expected_answer)))

            repo = s.metadata.get("repo", "unknown")
            repo_counts[repo] += 1

        # Max context: longest problem + patch
        max_ctx = 0
        for s in samples:
            ctx = tc.count(s.query)
            if s.expected_answer:
                ctx += tc.count(s.expected_answer)
            if ctx > max_ctx:
                max_ctx = ctx

        total = sum(repo_counts.values()) or 1
        domain_mix = {k: v / total for k, v in repo_counts.most_common(10)}

        return WorkloadProfile(
            workload_type=self.workload_type(),
            source_dataset=self.dataset_name(),
            n_samples=len(samples),
            input_tokens_dist=FittedDistribution.fit(input_tokens) if input_tokens else None,
            answer_tokens_dist=FittedDistribution.fit(answer_tokens) if answer_tokens else None,
            max_context_observed=max_ctx,
            domain_mix=domain_mix,
        )

    def workload_type(self) -> str:
        return "coding"

    def dataset_name(self) -> str:
        return "swebench"


register_characterizer("swebench", CodingCharacterizer)
