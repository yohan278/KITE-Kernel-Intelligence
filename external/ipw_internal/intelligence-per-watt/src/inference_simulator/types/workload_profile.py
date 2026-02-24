"""Workload profile: empirical distributions characterizing a workload type."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from inference_simulator.types.fitted_distribution import FittedDistribution


@dataclass
class WorkloadProfile:
    """Empirically-derived workload profile from dataset characterization.

    Captures the statistical shape of a workload type (chat, reasoning,
    agentic, rag, coding) as fitted distributions over token counts,
    turn structure, tool usage, and timing.

    Attributes:
        workload_type: Workload category ("chat", "reasoning", "agentic", "rag", "coding").
        source_dataset: Dataset used for characterization (e.g., "wildchat").
        n_samples: Number of conversations/requests analyzed.
        system_prompt_tokens: Typical system prompt length in tokens.
        structured_output_fraction: Fraction of requests requiring structured output.
        turns_or_steps_dist: Distribution over number of turns/steps per conversation.
        input_tokens_dist: Distribution over input token counts.
        thinking_tokens_dist: Distribution over thinking/reasoning token counts.
        answer_tokens_dist: Distribution over answer token counts.
        input_tokens_by_position: Per-turn-position input token distributions.
        output_tokens_by_position: Per-turn-position output token distributions.
        tool_call_probability: Probability that a turn includes a tool call.
        tool_call_tokens_dist: Distribution over tokens consumed by tool results.
        tool_type_distribution: Probability mass over tool types.
        inter_turn_seconds_dist: Distribution over inter-turn delay in seconds.
        kv_cache_eviction_threshold: Context length fraction that triggers KV eviction.
        context_growth_rate_dist: Distribution over context growth rate per turn.
        max_context_observed: Largest context window observed in the dataset.
        domain_mix: Probability mass over sub-domain labels.
    """

    workload_type: str
    source_dataset: str
    n_samples: int
    system_prompt_tokens: int = 0
    structured_output_fraction: float = 0.0

    # Core distributions
    turns_or_steps_dist: Optional[FittedDistribution] = None
    input_tokens_dist: Optional[FittedDistribution] = None
    thinking_tokens_dist: Optional[FittedDistribution] = None
    answer_tokens_dist: Optional[FittedDistribution] = None

    # Position-conditioned
    input_tokens_by_position: Dict[int, FittedDistribution] = field(default_factory=dict)
    output_tokens_by_position: Dict[int, FittedDistribution] = field(default_factory=dict)

    # Tool-related
    tool_call_probability: float = 0.0
    tool_call_tokens_dist: Optional[FittedDistribution] = None
    tool_type_distribution: Dict[str, float] = field(default_factory=dict)

    # Timing
    inter_turn_seconds_dist: Optional[FittedDistribution] = None
    kv_cache_eviction_threshold: float = 0.0

    # Context
    context_growth_rate_dist: Optional[FittedDistribution] = None
    max_context_observed: int = 0

    # Domain mix
    domain_mix: Dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save the profile to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> WorkloadProfile:
        """Load a profile from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Reconstructed WorkloadProfile.
        """
        with open(Path(path), "r") as f:
            d = json.load(f)
        return cls._from_dict(d)

    def _to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""

        def _fd(fd: Optional[FittedDistribution]) -> Optional[Dict[str, Any]]:
            return fd.to_dict() if fd is not None else None

        def _fd_map(
            m: Dict[int, FittedDistribution],
        ) -> Dict[str, Dict[str, Any]]:
            return {str(k): v.to_dict() for k, v in m.items()}

        return {
            "workload_type": self.workload_type,
            "source_dataset": self.source_dataset,
            "n_samples": self.n_samples,
            "system_prompt_tokens": self.system_prompt_tokens,
            "structured_output_fraction": self.structured_output_fraction,
            "turns_or_steps_dist": _fd(self.turns_or_steps_dist),
            "input_tokens_dist": _fd(self.input_tokens_dist),
            "thinking_tokens_dist": _fd(self.thinking_tokens_dist),
            "answer_tokens_dist": _fd(self.answer_tokens_dist),
            "input_tokens_by_position": _fd_map(self.input_tokens_by_position),
            "output_tokens_by_position": _fd_map(self.output_tokens_by_position),
            "tool_call_probability": self.tool_call_probability,
            "tool_call_tokens_dist": _fd(self.tool_call_tokens_dist),
            "tool_type_distribution": dict(self.tool_type_distribution),
            "inter_turn_seconds_dist": _fd(self.inter_turn_seconds_dist),
            "kv_cache_eviction_threshold": self.kv_cache_eviction_threshold,
            "context_growth_rate_dist": _fd(self.context_growth_rate_dist),
            "max_context_observed": self.max_context_observed,
            "domain_mix": dict(self.domain_mix),
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> WorkloadProfile:
        """Reconstruct a WorkloadProfile from a dictionary."""

        def _fd(v: Optional[Dict[str, Any]]) -> Optional[FittedDistribution]:
            return FittedDistribution.from_dict(v) if v is not None else None

        def _fd_map(
            m: Optional[Dict[str, Dict[str, Any]]],
        ) -> Dict[int, FittedDistribution]:
            if m is None:
                return {}
            return {int(k): FittedDistribution.from_dict(v) for k, v in m.items()}

        return cls(
            workload_type=d["workload_type"],
            source_dataset=d["source_dataset"],
            n_samples=d["n_samples"],
            system_prompt_tokens=d.get("system_prompt_tokens", 0),
            structured_output_fraction=d.get("structured_output_fraction", 0.0),
            turns_or_steps_dist=_fd(d.get("turns_or_steps_dist")),
            input_tokens_dist=_fd(d.get("input_tokens_dist")),
            thinking_tokens_dist=_fd(d.get("thinking_tokens_dist")),
            answer_tokens_dist=_fd(d.get("answer_tokens_dist")),
            input_tokens_by_position=_fd_map(d.get("input_tokens_by_position")),
            output_tokens_by_position=_fd_map(d.get("output_tokens_by_position")),
            tool_call_probability=d.get("tool_call_probability", 0.0),
            tool_call_tokens_dist=_fd(d.get("tool_call_tokens_dist")),
            tool_type_distribution=d.get("tool_type_distribution", {}),
            inter_turn_seconds_dist=_fd(d.get("inter_turn_seconds_dist")),
            kv_cache_eviction_threshold=d.get("kv_cache_eviction_threshold", 0.0),
            context_growth_rate_dist=_fd(d.get("context_growth_rate_dist")),
            max_context_observed=d.get("max_context_observed", 0),
            domain_mix=d.get("domain_mix", {}),
        )


__all__ = ["WorkloadProfile"]
