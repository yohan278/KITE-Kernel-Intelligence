"""AgentData scoring: LLM-as-judge task completion."""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

TASK_COMPLETION_TEMPLATE = """You are evaluating whether an AI agent successfully completed a task.

Task: {task}
Agent's response/output: {response}

Did the agent complete the task? Consider:
1. Did the agent understand the task?
2. Did the agent make progress toward completing it?
3. Is the final output reasonable for the given task?

Respond with one of:
- COMPLETED: The agent successfully completed the task
- PARTIAL: The agent made meaningful progress but didn't fully complete it
- FAILED: The agent failed to make meaningful progress

Grade: """


def score_task_completion(task: str, response: str) -> str:
    """Score task completion (synchronous, no LLM judge - uses heuristics)."""
    if not response or not response.strip():
        return "FAILED"

    # Basic heuristic: if response is substantial and doesn't contain error indicators
    error_indicators = ["error", "failed", "cannot", "unable to", "i don't know"]
    response_lower = response.lower()

    if len(response.strip()) < 10:
        return "FAILED"

    if any(indicator in response_lower for indicator in error_indicators):
        return "PARTIAL"

    return "COMPLETED"


def score_agentdata(results: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate task completion scores."""
    total = len(results)
    if total == 0:
        return {"total_samples": 0.0, "completion_rate": 0.0}

    completed = 0
    partial = 0
    failed = 0
    errors = 0

    for r in results.values():
        if r.get("error"):
            errors += 1
            failed += 1
            continue

        grade = r.get("grade", "FAILED")
        if grade == "COMPLETED":
            completed += 1
        elif grade == "PARTIAL":
            partial += 1
        else:
            failed += 1

    return {
        "total_samples": float(total),
        "completed_count": float(completed),
        "partial_count": float(partial),
        "failed_count": float(failed),
        "error_count": float(errors),
        "completion_rate": round((completed / total) * 100, 2),
        "success_rate": round(((completed + partial) / total) * 100, 2),
    }
