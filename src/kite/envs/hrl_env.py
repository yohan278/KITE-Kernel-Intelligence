"""Hierarchical environment: kernel + runtime."""

from __future__ import annotations

from kite.envs.runtime_env import RuntimeEnv
from kite.policies.hierarchy_controller import HierarchyController
from kite.types import RuntimeState


class HRLEnv:
    def __init__(self) -> None:
        self.runtime_env = RuntimeEnv()
        self.controller = HierarchyController()

    def reset(self) -> RuntimeState:
        return self.runtime_env.reset()

    def choose_kernel_family(self, state: RuntimeState) -> str:
        return self.controller.choose_kernel_family(state).kernel_family
