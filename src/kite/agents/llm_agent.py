"""LLM agent for Kevin-style multi-turn kernel optimization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import signal
import threading
import time
from typing import Any, Callable, List

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
    timeout: bool = False
    error: str | None = None


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
        turn_timeout_seconds: float = 0.0,
        on_step: Callable[[MultiTurnStep], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> MultiTurnResult:
        steps: list[MultiTurnStep] = []

        def _emit(payload: dict[str, Any]) -> None:
            if on_event is not None:
                on_event(payload)

        @contextmanager
        def _turn_timeout_context() -> Any:
            timeout_s = float(turn_timeout_seconds)
            if timeout_s <= 0.0:
                yield
                return
            if not hasattr(signal, "SIGALRM"):
                yield
                return
            if threading.current_thread() is not threading.main_thread():
                yield
                return

            def _handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"Turn timed out after {timeout_s:.1f}s")

            prev_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
            try:
                yield
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, prev_handler)

        try:
            t0 = time.monotonic()
            with _turn_timeout_context():
                candidate = self.policy.generate_candidate(task, attempt=0)
            code = candidate.code
            _emit(
                {
                    "event": "initial_candidate",
                    "turn": 0,
                    "duration_s": time.monotonic() - t0,
                    "code": code,
                }
            )
        except TimeoutError as exc:
            code = (
                "import torch\n"
                "import torch.nn as nn\n\n"
                "class ModelNew(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n\n"
                "    def forward(self, *args):\n"
                "        raise RuntimeError('initial generation timeout')\n"
            )
            _emit(
                {
                    "event": "initial_generation_timeout",
                    "turn": 0,
                    "error": str(exc),
                    "code": code,
                }
            )

        feedback = ""

        for turn in range(1, max_turns + 1):
            timed_out = False
            error_msg = None
            metrics: dict[str, Any]
            eval_t0 = time.monotonic()
            try:
                with _turn_timeout_context():
                    metrics = evaluate_fn(code)
                _emit(
                    {
                        "event": "evaluate_ok",
                        "turn": turn,
                        "duration_s": time.monotonic() - eval_t0,
                        "compile_ok": bool(metrics.get("compile_ok", False)),
                        "correct": bool(metrics.get("correct", False)),
                        "reward": float(metrics.get("reward", 0.0)),
                    }
                )
            except TimeoutError as exc:
                timed_out = True
                error_msg = f"evaluation_timeout: {exc}"
                metrics = {
                    "compile_ok": False,
                    "correct": False,
                    "reward": -1.0,
                    "runtime_ms": 0.0,
                    "joules": 0.0,
                }
                _emit(
                    {
                        "event": "evaluation_timeout",
                        "turn": turn,
                        "duration_s": time.monotonic() - eval_t0,
                        "error": error_msg,
                    }
                )
            step = MultiTurnStep(
                turn=turn,
                code=code,
                compile_ok=bool(metrics.get("compile_ok", False)),
                correct=bool(metrics.get("correct", False)),
                reward=float(metrics.get("reward", 0.0)),
                runtime_ms=float(metrics.get("runtime_ms", 0.0)),
                joules=float(metrics.get("joules", 0.0)),
                timeout=timed_out,
                error=error_msg,
            )
            steps.append(step)
            if on_step is not None:
                on_step(step)
            if step.correct:
                break

            if step.timeout:
                feedback = (
                    "Previous evaluation timed out. Produce a simpler kernel and avoid heavy custom classes."
                )
            elif not step.compile_ok:
                feedback = "Compilation failed. Fix syntax/import/runtime errors and return corrected kernel only."
            else:
                feedback = (
                    "Kernel is incorrect or suboptimal. Keep correctness and improve speedup while reducing joules."
                )
            repair_prompt = (
                f"{task.prompt}\n\n"
                f"Previous kernel:\n```python\n{code}\n```\n\n"
                f"Feedback: {feedback}\n"
                "Return only valid Python code that defines class ModelNew(nn.Module).\n"
                "Do not return markdown fences or prose.\n"
                "Do not use Triton.\n"
            )
            _emit(
                {
                    "event": "repair_prompt",
                    "turn": turn,
                    "feedback": feedback,
                    "prompt": repair_prompt,
                }
            )

            gen_t0 = time.monotonic()
            try:
                with _turn_timeout_context():
                    repaired_raw = self.policy.generate_text(repair_prompt).strip()
                extracted = self.policy.extract_code(repaired_raw).strip()
                code = extracted if extracted else repaired_raw
                _emit(
                    {
                        "event": "repair_generation",
                        "turn": turn,
                        "duration_s": time.monotonic() - gen_t0,
                        "raw_output": repaired_raw,
                        "extracted_code": extracted,
                    }
                )
            except TimeoutError as exc:
                _emit(
                    {
                        "event": "generation_timeout",
                        "turn": turn,
                        "duration_s": time.monotonic() - gen_t0,
                        "error": str(exc),
                    }
                )
                # Keep previous code and continue to next turn.
                continue

        return MultiTurnResult(task_id=task.task_id, steps=steps)
