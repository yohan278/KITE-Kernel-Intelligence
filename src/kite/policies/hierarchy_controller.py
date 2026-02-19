"""Hierarchy controller for kernel/runtime decisions."""

from __future__ import annotations

from dataclasses import dataclass

from kite.types import RuntimeState


@dataclass(slots=True)
class HierarchyDecision:
    kernel_family: str
    rationale: str


class HierarchyController:
    def choose_kernel_family(self, state: RuntimeState) -> HierarchyDecision:
        if state.phase_ratio >= 0.6:
            return HierarchyDecision(
                kernel_family="decode_optimized",
                rationale="decode-heavy phase detected",
            )
        if state.queue_depth > 32:
            return HierarchyDecision(
                kernel_family="throughput_optimized",
                rationale="queue pressure requires throughput bias",
            )
        return HierarchyDecision(
            kernel_family="balanced",
            rationale="mixed phase, balanced kernel family",
        )
