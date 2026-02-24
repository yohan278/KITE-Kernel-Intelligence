"""Chat workload characterizer — analyses WildChat conversations."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from dataset_generator.characterization.base import BaseCharacterizer
from dataset_generator.characterization.registry import register_characterizer
from dataset_generator.characterization.tokenizer import FastTokenCounter
from dataset_generator.datasets.wildchat import WildChatLoader
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


class ChatCharacterizer(BaseCharacterizer):
    """Characterize multi-turn chat workloads from WildChat."""

    def characterize(self, limit: Optional[int] = None) -> WorkloadProfile:
        loader = WildChatLoader()
        conversations = loader.load_conversations(limit=limit)
        tc = FastTokenCounter()

        turn_counts = []
        all_input_tokens = []
        all_output_tokens = []
        input_by_pos: dict[int, list[float]] = {}
        output_by_pos: dict[int, list[float]] = {}
        model_counts: Counter[str] = Counter()

        for conv in conversations:
            turns = conv.turns
            turn_counts.append(float(len(turns)))
            model_counts[conv.model] += 1

            pos = 0
            for turn in turns:
                tokens = tc.count(turn.content)
                if turn.role == "user":
                    all_input_tokens.append(float(tokens))
                    input_by_pos.setdefault(pos, []).append(float(tokens))
                elif turn.role == "assistant":
                    all_output_tokens.append(float(tokens))
                    output_by_pos.setdefault(pos, []).append(float(tokens))
                pos += 1

        # Compute max context: sum of all tokens across turns in the longest conversation
        max_ctx = 0
        for conv in conversations:
            ctx = sum(tc.count(t.content) for t in conv.turns)
            if ctx > max_ctx:
                max_ctx = ctx

        # Domain mix from model distribution
        total = sum(model_counts.values()) or 1
        domain_mix = {k: v / total for k, v in model_counts.most_common(10)}

        return WorkloadProfile(
            workload_type=self.workload_type(),
            source_dataset=self.dataset_name(),
            n_samples=len(conversations),
            turns_or_steps_dist=FittedDistribution.fit(turn_counts) if turn_counts else None,
            input_tokens_dist=FittedDistribution.fit(all_input_tokens) if all_input_tokens else None,
            answer_tokens_dist=FittedDistribution.fit(all_output_tokens) if all_output_tokens else None,
            input_tokens_by_position={
                pos: FittedDistribution.fit(vals)
                for pos, vals in input_by_pos.items()
            },
            output_tokens_by_position={
                pos: FittedDistribution.fit(vals)
                for pos, vals in output_by_pos.items()
            },
            max_context_observed=max_ctx,
            domain_mix=domain_mix,
        )

    def workload_type(self) -> str:
        return "chat"

    def dataset_name(self) -> str:
        return "wildchat"


register_characterizer("wildchat", ChatCharacterizer)
