"""High-level kernel family chooser."""

from __future__ import annotations

from dataclasses import dataclass

from kite.policies.hierarchy_controller import HierarchyController
from kite.types import RuntimeState


@dataclass(slots=True)
class KernelChoice:
    family: str
    confidence: float
    rationale: str


class HighLevelKernelChooser:
    def __init__(self, controller: HierarchyController | None = None) -> None:
        self.controller = controller or HierarchyController()

    def choose(self, state: RuntimeState, explore: bool = True) -> KernelChoice:
        decision = self.controller.choose_kernel_family(state, explore=explore)
        return KernelChoice(
            family=decision.kernel_family,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )

