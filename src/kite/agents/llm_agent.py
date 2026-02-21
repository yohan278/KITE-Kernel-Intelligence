"""LLM agent for Kevin-style multi-turn kernel optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from kite.policies.qwen_policy import QwenPolicy
from kite.types import KernelTask


@dataclass(slots=True)
class MultiTurnStep:
    turn: int
    code: str
    compile_ok: bool
    correct: bool
    reward: float
    runtime_ms: float
    joules: float


@dataclass(slots=True)
class MultiTurnResult:
    task_id: str
    steps: List[MultiTurnStep]

    @property
    def pass_at_k(self) -> bool:
        return any(step.correct for step in self.steps)

    @property
    def turns_to_success(self) -> int:
        for step in self.steps:
            if step.correct:
                return step.turn
        return -1


class LLMKernelAgent:
    def __init__(self, policy: QwenPolicy) -> None:
        self.policy = policy

    def optimize_task(
        self,
        task: KernelTask,
        evaluate_fn: Callable[[str], dict],
        max_turns: int = 5,
    ) -> MultiTurnResult:
        steps: list[MultiTurnStep] = []

        candidate = self.policy.generate_candidate(task, attempt=0)
        code = candidate.code
        feedback = ""

        for turn in range(1, max_turns + 1):
            metrics = evaluate_fn(code)
            step = MultiTurnStep(
                turn=turn,
                code=code,
                compile_ok=bool(metrics.get("compile_ok", False)),
                correct=bool(metrics.get("correct", False)),
                reward=float(metrics.get("reward", 0.0)),
                runtime_ms=float(metrics.get("runtime_ms", 0.0)),
                joules=float(metrics.get("joules", 0.0)),
            )
            steps.append(step)
            if step.correct:
                break

            if not step.compile_ok:
                feedback = "Compilation failed. Fix syntax/import/runtime errors and return corrected kernel only."
            else:
                feedback = (
                    "Kernel is incorrect or suboptimal. Keep correctness and improve speedup while reducing joules."
                )
            repair_prompt = (
                f"{task.prompt}\n\n"
                f"Previous kernel:\n```python\n{code}\n```\n\n"
                f"Feedback: {feedback}\n"
                "Return a corrected improved kernel."
            )
            code = self.policy.generate_text(repair_prompt).strip()

        return MultiTurnResult(task_id=task.task_id, steps=steps)

