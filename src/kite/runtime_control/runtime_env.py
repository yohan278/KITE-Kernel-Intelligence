"""Runtime environment for replaying phase traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kite.envs.runtime_env import RuntimeEnv
from kite.policies.runtime_actor_critic import RuntimeAction
from kite.types import RuntimeState


@dataclass(slots=True)
class PhasePoint:
    phase_id: str
    queue_depth: int
    phase_ratio: float


class PhaseTraceRuntimeEnv(RuntimeEnv):
    def __init__(self, phase_trace: Iterable[PhasePoint], **kwargs) -> None:
        super().__init__(**kwargs)
        self.phase_trace = list(phase_trace)
        self._phase_idx = 0

    def reset(self) -> RuntimeState:
        state = super().reset()
        self._phase_idx = 0
        if self.phase_trace:
            p = self.phase_trace[0]
            state.queue_depth = p.queue_depth
            state.phase_ratio = p.phase_ratio
            state.phase_id = p.phase_id
        return state

    def step(self, state: RuntimeState, action: RuntimeAction):
        out = super().step(state, action)
        self._phase_idx = min(self._phase_idx + 1, max(0, len(self.phase_trace) - 1))
        if self.phase_trace:
            p = self.phase_trace[self._phase_idx]
            out.next_state.queue_depth = p.queue_depth
            out.next_state.phase_ratio = p.phase_ratio
            out.next_state.phase_id = p.phase_id
        return out

