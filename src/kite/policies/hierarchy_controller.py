"""Hierarchy controller for kernel family selection.

Implements a learned policy (MLP when torch is available) that selects which
kernel family/variant to use based on the current workload regime.  The high-level
choice is then passed to the low-level runtime controller as conditioning context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kite.types import RuntimeState
from kite.utils.logging import get_logger

logger = get_logger(__name__)

KERNEL_FAMILIES = [
    "balanced",
    "prefill_optimized",
    "decode_optimized",
    "throughput_optimized",
    "energy_optimized",
]

STATE_DIM = 7


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


@dataclass(slots=True)
class HierarchyDecision:
    kernel_family: str
    rationale: str
    confidence: float = 0.0


@dataclass(slots=True)
class HierarchyControllerConfig:
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    families: List[str] = field(default_factory=lambda: list(KERNEL_FAMILIES))


def _try_import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        return torch, nn
    except ImportError:
        return None, None


class HierarchyController:
    """Selects kernel family conditioned on workload state.

    Uses an MLP classifier when torch is available, otherwise falls back
    to rule-based heuristics.
    """

    def __init__(self, config: HierarchyControllerConfig | None = None) -> None:
        self.config = config or HierarchyControllerConfig()
        self._model = None
        self._optimizer = None
        self._torch = None
        self.selection_history: list[dict] = []

        torch_pkg, nn = _try_import_torch()
        if torch_pkg is not None:
            self._init_torch(torch_pkg, nn)

    def _init_torch(self, torch_pkg, nn) -> None:
        num_families = len(self.config.families)

        class FamilySelector(nn.Module):
            def __init__(self, state_dim, num_classes, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        self._model = FamilySelector(STATE_DIM, num_families, self.config.hidden_dim)
        self._optimizer = torch_pkg.optim.Adam(
            self._model.parameters(), lr=self.config.learning_rate
        )
        self._torch = torch_pkg

    @property
    def has_torch(self) -> bool:
        return self._model is not None

    def choose_kernel_family(
        self,
        state: RuntimeState,
        explore: bool = True,
    ) -> HierarchyDecision:
        if self.has_torch:
            return self._choose_torch(state, explore)
        return self._choose_heuristic(state)

    def _choose_torch(self, state: RuntimeState, explore: bool) -> HierarchyDecision:
        torch = self._torch
        sv = torch.tensor([_state_to_vector(state)], dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(sv)
            probs = torch.softmax(logits, dim=-1)

        if explore:
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample().item()
        else:
            idx = probs.argmax(dim=-1).item()

        family = self.config.families[min(idx, len(self.config.families) - 1)]
        confidence = probs[0, idx].item()

        self.selection_history.append({
            "state": _state_to_vector(state),
            "family_idx": idx,
            "confidence": confidence,
        })

        return HierarchyDecision(
            kernel_family=family,
            rationale=f"MLP selected {family} (p={confidence:.3f})",
            confidence=confidence,
        )

    def _choose_heuristic(self, state: RuntimeState) -> HierarchyDecision:
        if state.phase_ratio >= 0.7:
            family = "decode_optimized"
            rationale = "decode-heavy phase detected"
        elif state.phase_ratio <= 0.3:
            family = "prefill_optimized"
            rationale = "prefill-heavy phase detected"
        elif state.queue_depth > 32:
            family = "throughput_optimized"
            rationale = "queue pressure requires throughput bias"
        elif state.power_cap <= 350:
            family = "energy_optimized"
            rationale = "low power cap favors energy efficiency"
        else:
            family = "balanced"
            rationale = "mixed phase, balanced kernel family"

        return HierarchyDecision(
            kernel_family=family,
            rationale=rationale,
            confidence=1.0,
        )

    def update_from_reward(self, reward: float) -> dict[str, float]:
        """REINFORCE-style update using the most recent selection."""
        if not self.has_torch or not self.selection_history:
            return {"loss": 0.0}

        torch = self._torch
        entry = self.selection_history[-1]
        sv = torch.tensor([entry["state"]], dtype=torch.float32)
        target_idx = torch.tensor([entry["family_idx"]], dtype=torch.long)

        logits = self._model(sv)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_prob = log_probs[0, target_idx]

        loss = -selected_log_prob * reward

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {"loss": loss.item()}

    def save(self, path) -> None:
        if self.has_torch:
            self._torch.save(self._model.state_dict(), str(path))

    def load(self, path) -> None:
        if self.has_torch:
            self._model.load_state_dict(
                self._torch.load(str(path), weights_only=True)
            )
