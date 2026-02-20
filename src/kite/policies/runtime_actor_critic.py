"""Runtime actor-critic policy with MLP backbone and PPO-style updates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from kite.types import RuntimeState
from kite.utils.logging import get_logger

logger = get_logger(__name__)

RuntimeAction = Tuple[int, str, int, int]

POWER_CAPS = [200, 300, 350, 400, 450, 550, 650, 700]
DVFS_PROFILES = ["efficiency", "balanced", "performance"]
MICROBATCH_SIZES = [1, 2, 4, 8, 16]
CONCURRENCY_LEVELS = [1, 2, 4, 8]

NUM_ACTIONS = len(POWER_CAPS) * len(DVFS_PROFILES) * len(MICROBATCH_SIZES) * len(CONCURRENCY_LEVELS)


def _action_index_to_tuple(idx: int) -> RuntimeAction:
    """Flatten multi-dimensional discrete action to a single index."""
    n_dvfs = len(DVFS_PROFILES)
    n_mb = len(MICROBATCH_SIZES)
    n_conc = len(CONCURRENCY_LEVELS)

    pc_idx = idx // (n_dvfs * n_mb * n_conc)
    remainder = idx % (n_dvfs * n_mb * n_conc)
    dvfs_idx = remainder // (n_mb * n_conc)
    remainder = remainder % (n_mb * n_conc)
    mb_idx = remainder // n_conc
    conc_idx = remainder % n_conc

    return (
        POWER_CAPS[min(pc_idx, len(POWER_CAPS) - 1)],
        DVFS_PROFILES[min(dvfs_idx, len(DVFS_PROFILES) - 1)],
        MICROBATCH_SIZES[min(mb_idx, len(MICROBATCH_SIZES) - 1)],
        CONCURRENCY_LEVELS[min(conc_idx, len(CONCURRENCY_LEVELS) - 1)],
    )


def _tuple_to_action_index(action: RuntimeAction) -> int:
    pc, dvfs, mb, conc = action
    pc_idx = POWER_CAPS.index(pc) if pc in POWER_CAPS else 0
    dvfs_idx = DVFS_PROFILES.index(dvfs) if dvfs in DVFS_PROFILES else 0
    mb_idx = MICROBATCH_SIZES.index(mb) if mb in MICROBATCH_SIZES else 0
    conc_idx = CONCURRENCY_LEVELS.index(conc) if conc in CONCURRENCY_LEVELS else 0

    n_dvfs = len(DVFS_PROFILES)
    n_mb = len(MICROBATCH_SIZES)
    n_conc = len(CONCURRENCY_LEVELS)
    return pc_idx * (n_dvfs * n_mb * n_conc) + dvfs_idx * (n_mb * n_conc) + mb_idx * n_conc + conc_idx


STATE_DIM = 7  # queue_depth, phase_ratio, batch_size, concurrency, power_cap, ttft_p95, e2e_p95


def _state_to_vector(state: RuntimeState) -> list[float]:
    return [
        float(state.queue_depth) / 64.0,
        state.phase_ratio,
        float(state.batch_size) / 16.0,
        float(state.concurrency) / 8.0,
        float(state.power_cap) / 700.0,
        state.ttft_p95 / 5.0,
        state.e2e_p95 / 60.0,
    ]


def _try_import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore

        return torch, nn, F
    except ImportError:
        return None, None, None


@dataclass(slots=True)
class RuntimeActorCriticConfig:
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    actions: List[RuntimeAction] = field(
        default_factory=lambda: [
            (350, "efficiency", 1, 1),
            (450, "balanced", 2, 2),
            (550, "balanced", 4, 4),
            (650, "performance", 8, 8),
        ]
    )


class RuntimeActorCritic:
    """Actor-critic with MLP backbone, falling back to table-based when torch unavailable."""

    def __init__(self, config: RuntimeActorCriticConfig | None = None) -> None:
        self.config = config or RuntimeActorCriticConfig()
        self._torch_module = None
        self._optimizer = None
        self.value_table: Dict[str, float] = {}
        self._trajectory: list[dict] = []

        torch_pkg, nn, F = _try_import_torch()
        if torch_pkg is not None:
            self._init_torch(torch_pkg, nn)

    def _init_torch(self, torch_pkg, nn) -> None:
        """Build the MLP actor-critic network."""

        class ActorCriticMLP(nn.Module):
            def __init__(self, state_dim, num_actions, hidden_dim):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.actor_head = nn.Linear(hidden_dim, num_actions)
                self.critic_head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                features = self.shared(x)
                logits = self.actor_head(features)
                value = self.critic_head(features).squeeze(-1)
                return logits, value

        num_actions = len(self.config.actions)
        self._torch_module = ActorCriticMLP(STATE_DIM, num_actions, self.config.hidden_dim)
        self._optimizer = torch_pkg.optim.Adam(
            self._torch_module.parameters(), lr=self.config.learning_rate
        )
        self._torch = torch_pkg

    @property
    def has_torch(self) -> bool:
        return self._torch_module is not None

    def select_action(self, state: RuntimeState, explore: bool = True) -> RuntimeAction:
        if self.has_torch:
            return self._select_action_torch(state, explore)
        return self._select_action_table(state)

    def _select_action_torch(self, state: RuntimeState, explore: bool) -> RuntimeAction:
        torch = self._torch
        sv = torch.tensor([_state_to_vector(state)], dtype=torch.float32)
        with torch.no_grad():
            logits, value = self._torch_module(sv)
            probs = torch.softmax(logits, dim=-1)

        if explore:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()
        else:
            action_idx = probs.argmax(dim=-1).item()
            log_prob = math.log(probs[0, action_idx].item() + 1e-8)

        action = self.config.actions[min(action_idx, len(self.config.actions) - 1)]
        self._trajectory.append({
            "state": _state_to_vector(state),
            "action_idx": action_idx,
            "log_prob": log_prob,
            "value": value.item(),
        })
        return action

    def _select_action_table(self, state: RuntimeState) -> RuntimeAction:
        if state.phase_ratio > 0.7:
            return self.config.actions[0]
        if state.queue_depth > 32:
            return self.config.actions[-1]
        return self.config.actions[min(len(self.config.actions) - 1, state.concurrency // 2)]

    def store_reward(self, reward: float, done: bool = False) -> None:
        if self._trajectory:
            self._trajectory[-1]["reward"] = reward
            self._trajectory[-1]["done"] = done

    def update_value(self, state: RuntimeState, reward: float, lr: float = 0.1) -> None:
        """Legacy table-based value update (used by stub path)."""
        key = self._state_key(state)
        prev = self.value_table.get(key, 0.0)
        self.value_table[key] = (1 - lr) * prev + lr * reward

    def ppo_update(self) -> dict[str, float]:
        """Run PPO update on accumulated trajectory."""
        if not self.has_torch or not self._trajectory:
            self._trajectory.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        torch = self._torch
        cfg = self.config

        states = torch.tensor(
            [t["state"] for t in self._trajectory], dtype=torch.float32
        )
        actions = torch.tensor(
            [t["action_idx"] for t in self._trajectory], dtype=torch.long
        )
        old_log_probs = torch.tensor(
            [t["log_prob"] for t in self._trajectory], dtype=torch.float32
        )
        old_values = torch.tensor(
            [t["value"] for t in self._trajectory], dtype=torch.float32
        )
        rewards_raw = [t.get("reward", 0.0) for t in self._trajectory]
        dones = [t.get("done", False) for t in self._trajectory]

        # GAE computation
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(rewards_raw))):
            if dones[i]:
                next_value = 0.0
                gae = 0.0
            delta = rewards_raw[i] + cfg.gamma * next_value - old_values[i].item()
            gae = delta + cfg.gamma * cfg.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[i].item())
            next_value = old_values[i].item()

        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        if advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(cfg.ppo_epochs):
            logits, values = self._torch_module(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages_t
            surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (values - returns_t).pow(2).mean()

            loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._torch_module.parameters(), cfg.max_grad_norm
            )
            self._optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        n = max(1, cfg.ppo_epochs)
        self._trajectory.clear()

        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    def save(self, path) -> None:
        if self.has_torch:
            self._torch.save(self._torch_module.state_dict(), str(path))

    def load(self, path) -> None:
        if self.has_torch:
            self._torch_module.load_state_dict(
                self._torch.load(str(path), weights_only=True)
            )

    @staticmethod
    def _state_key(state: RuntimeState) -> str:
        return (
            f"q={state.queue_depth}|p={state.phase_ratio:.2f}|b={state.batch_size}|"
            f"c={state.concurrency}|cap={state.power_cap}|clk={state.clocks}"
        )
