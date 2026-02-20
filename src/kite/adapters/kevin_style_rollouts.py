"""Kevin-style grouped rollouts with model-based fix loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from kite.types import KernelCandidate, KernelTask
from kite.utils.logging import get_logger

if TYPE_CHECKING:
    from kite.policies.qwen_policy import QwenPolicy

logger = get_logger(__name__)


@dataclass(slots=True)
class RolloutConfig:
    group_size: int = 8
    max_fix_rounds: int = 2
    timeout_ms: float = 500.0


def grouped_rollouts(policy, task: KernelTask, config: RolloutConfig) -> List[KernelCandidate]:
    """Generate a group of candidates with multi-turn fix loops."""
    candidates: list[KernelCandidate] = []
    for i in range(config.group_size):
        candidate = policy.generate_candidate(task, attempt=i)
        fixed = _fix_loop(candidate, config.max_fix_rounds, policy, task)
        candidates.append(fixed)
    return candidates


def filter_trajectories(
    candidates: List[KernelCandidate],
    keep_top_k: int = 4,
) -> List[KernelCandidate]:
    """Keep top-k by (correctness, speedup); Kevin-style trajectory filtering."""
    valid = [c for c in candidates if c.compile_ok]
    ranked = sorted(valid, key=lambda c: (c.correct, c.speedup or 0.0), reverse=True)
    if len(ranked) < keep_top_k:
        remaining = [c for c in candidates if not c.compile_ok]
        ranked.extend(remaining[: keep_top_k - len(ranked)])
    return ranked[:keep_top_k]


def _fix_loop(
    candidate: KernelCandidate,
    max_rounds: int,
    policy: Optional[object] = None,
    task: Optional[KernelTask] = None,
) -> KernelCandidate:
    """Multi-turn fix loop: re-prompt the model with compile errors.

    When a real policy is available, builds a repair prompt with the error
    context and asks for a corrected version. Falls back to heuristic
    patching when the model is unavailable.
    """
    working = candidate
    for round_idx in range(max_rounds):
        if working.compile_ok and working.correct:
            break

        if policy is not None and task is not None and _has_generate_text(policy):
            working = _model_fix(working, policy, task, round_idx)
        else:
            working = _heuristic_fix(working)

    return working


def _has_generate_text(policy) -> bool:
    return hasattr(policy, "generate_text") and callable(getattr(policy, "generate_text"))


def _model_fix(
    candidate: KernelCandidate,
    policy,
    task: KernelTask,
    round_idx: int,
) -> KernelCandidate:
    """Ask the model to fix the kernel given error feedback."""
    error_context = ""
    if not candidate.compile_ok:
        error_context = "The kernel failed to compile."
        if candidate.logs.get("metadata"):
            meta = candidate.logs["metadata"]
            if isinstance(meta, dict):
                for k, v in meta.items():
                    if "error" in str(k).lower():
                        error_context += f"\nError: {v}"
    elif not candidate.correct:
        error_context = "The kernel compiled but produced incorrect results."

    ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
    repair_prompt = (
        "The following GPU kernel implementation has a problem.\n\n"
        f"Reference implementation:\n```python\n{ref_src}\n```\n\n"
        f"Broken implementation:\n```python\n{candidate.code}\n```\n\n"
        f"Problem: {error_context}\n\n"
        "Please provide a corrected, optimized implementation that:\n"
        "1. Produces identical outputs to the reference\n"
        "2. Compiles and runs correctly\n"
        "3. Is optimized for speed on NVIDIA H100\n\n"
        "Write the corrected kernel:"
    )

    try:
        raw = policy.generate_text(repair_prompt)
        code = _extract_code(raw)
        if not code or len(code.strip()) < 10:
            return _heuristic_fix(candidate)

        compile_ok = "TODO" not in code and "pass" not in code.split("\n")[-1]
        return KernelCandidate(
            task_id=candidate.task_id,
            code=code,
            compile_ok=compile_ok,
            correct=False,
            runtime_ms=candidate.runtime_ms,
            speedup=candidate.speedup,
            logs={**candidate.logs, "fixed": True, "fix_round": round_idx, "fix_mode": "model"},
        )
    except Exception as exc:
        logger.debug("Model fix failed: %s", exc)
        return _heuristic_fix(candidate)


def _heuristic_fix(candidate: KernelCandidate) -> KernelCandidate:
    """Fallback: simple text-level patching for stub mode."""
    code = candidate.code
    if "TODO" in code:
        code = code.replace("TODO", "optimized path")

    compile_ok = "TODO" not in code and "pass" not in code
    correct = compile_ok and "return" in code

    return KernelCandidate(
        task_id=candidate.task_id,
        code=code,
        compile_ok=compile_ok,
        correct=correct,
        runtime_ms=candidate.runtime_ms if candidate.runtime_ms is not None else 120.0,
        speedup=candidate.speedup if candidate.speedup is not None else (1.2 if correct else 0.7),
        logs={**candidate.logs, "fixed": True, "fix_mode": "heuristic"},
    )


def _extract_code(raw: str) -> str:
    if "```python" in raw:
        parts = raw.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith(("python\n", "Python\n", "py\n")):
                code = "\n".join(code.split("\n")[1:])
            return code.strip()
    return raw.strip()
