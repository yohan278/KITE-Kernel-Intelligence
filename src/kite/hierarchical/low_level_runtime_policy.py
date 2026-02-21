"""Low-level runtime policy wrapper."""

from __future__ import annotations

from kite.policies.runtime_actor_critic import RuntimeAction, RuntimeActorCritic
from kite.types import RuntimeState


class LowLevelRuntimePolicy:
    def __init__(self, actor: RuntimeActorCritic | None = None) -> None:
        self.actor = actor or RuntimeActorCritic()

    def act(self, state: RuntimeState, explore: bool = True) -> RuntimeAction:
        return self.actor.select_action(state, explore=explore)

    def observe_reward(self, reward: float, done: bool = False) -> None:
        self.actor.store_reward(reward, done=done)

