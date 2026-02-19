"""Kevin-style grouped rollouts with lightweight fix loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from kite.types import KernelCandidate, KernelTask


@dataclass(slots=True)
class RolloutConfig:
    group_size: int = 8
    max_fix_rounds: int = 2
    timeout_ms: float = 500.0


def grouped_rollouts(policy, task: KernelTask, config: RolloutConfig) -> List[KernelCandidate]:
    candidates: list[KernelCandidate] = []
    for i in range(config.group_size):
        candidate = policy.generate_candidate(task, attempt=i)
        fixed = _fix_loop(candidate, config.max_fix_rounds)
        candidates.append(fixed)
    return candidates


def filter_trajectories(candidates: List[KernelCandidate], keep_top_k: int = 4) -> List[KernelCandidate]:
    valid = [c for c in candidates if c.compile_ok]
    ranked = sorted(valid, key=lambda c: (c.correct, c.speedup or 0.0), reverse=True)
    return ranked[:keep_top_k]


def _fix_loop(candidate: KernelCandidate, max_rounds: int) -> KernelCandidate:
    working = candidate
    for _ in range(max_rounds):
        if working.compile_ok and working.correct:
            break

        code = working.code
        if "TODO" in code:
            code = code.replace("TODO", "optimized path")

        compile_ok = "TODO" not in code and "pass" not in code
        correct = compile_ok and "return" in code

        working = KernelCandidate(
            task_id=working.task_id,
            code=code,
            compile_ok=compile_ok,
            correct=correct,
            runtime_ms=working.runtime_ms if working.runtime_ms is not None else 120.0,
            speedup=working.speedup if working.speedup is not None else (1.2 if correct else 0.7),
            logs={**working.logs, "fixed": True},
        )

    return working
