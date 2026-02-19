"""Runtime actor-critic style policy (table-based v0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from kite.types import RuntimeState


RuntimeAction = Tuple[int, str, int, int]


@dataclass(slots=True)
class RuntimeActorCriticConfig:
    actions: List[RuntimeAction] = field(
        default_factory=lambda: [
            (350, "efficiency", 1, 1),
            (450, "balanced", 2, 2),
            (550, "balanced", 4, 4),
            (650, "performance", 8, 8),
        ]
    )


class RuntimeActorCritic:
    def __init__(self, config: RuntimeActorCriticConfig | None = None) -> None:
        self.config = config or RuntimeActorCriticConfig()
        self.value_table: Dict[str, float] = {}

    def select_action(self, state: RuntimeState) -> RuntimeAction:
        if state.phase_ratio > 0.7:
            return self.config.actions[0]
        if state.queue_depth > 32:
            return self.config.actions[-1]
        return self.config.actions[min(len(self.config.actions) - 1, state.concurrency // 2)]

    def update_value(self, state: RuntimeState, reward: float, lr: float = 0.1) -> None:
        key = self._state_key(state)
        prev = self.value_table.get(key, 0.0)
        self.value_table[key] = (1 - lr) * prev + lr * reward

    @staticmethod
    def _state_key(state: RuntimeState) -> str:
        return (
            f"q={state.queue_depth}|p={state.phase_ratio:.2f}|b={state.batch_size}|"
            f"c={state.concurrency}|cap={state.power_cap}|clk={state.clocks}"
        )
