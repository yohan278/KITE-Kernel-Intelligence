"""WildChat scoring: completion rate (all turns responded)."""
from __future__ import annotations

from typing import Any, Dict


def score_completion(results: Dict[str, Any]) -> Dict[str, float]:
    """Score based on conversation completion rate."""
    total = len(results)
    if total == 0:
        return {"completion_rate": 0.0, "total_samples": 0.0}

    completed = sum(1 for r in results.values() if r.get("completed", False))
    error_count = sum(1 for r in results.values() if r.get("error"))

    return {
        "total_samples": float(total),
        "completed_count": float(completed),
        "completion_rate": round((completed / total) * 100, 2),
        "error_count": float(error_count),
        "avg_turns": round(
            sum(r.get("num_turns_completed", 0) for r in results.values()) / total, 2
        ),
    }
