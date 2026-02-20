"""Hierarchical environment: kernel family selection + runtime control."""

from __future__ import annotations

from kite.envs.runtime_env import RuntimeEnv, RuntimeStepResult
from kite.policies.hierarchy_controller import HierarchyController, HierarchyDecision
from kite.policies.runtime_actor_critic import RuntimeAction
from kite.types import RuntimeState


class HRLEnv:
    """Wraps RuntimeEnv with a hierarchy controller for kernel family selection."""

    def __init__(
        self,
        runtime_env: RuntimeEnv | None = None,
        controller: HierarchyController | None = None,
    ) -> None:
        self.runtime_env = runtime_env or RuntimeEnv()
        self.controller = controller or HierarchyController()
        self._current_family: str = "balanced"

    def reset(self) -> RuntimeState:
        self._current_family = "balanced"
        return self.runtime_env.reset()

    def choose_kernel_family(self, state: RuntimeState, explore: bool = True) -> HierarchyDecision:
        decision = self.controller.choose_kernel_family(state, explore=explore)
        self._current_family = decision.kernel_family
        return decision

    @property
    def current_family(self) -> str:
        return self._current_family

    def step(self, state: RuntimeState, action: RuntimeAction) -> RuntimeStepResult:
        return self.runtime_env.step(state, action)
