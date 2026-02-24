"""SLA constraint checker for inference search."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from inference_search.types import SLAConstraint


def check(
    metrics: Dict[str, float],
    constraints: Sequence[SLAConstraint],
) -> Tuple[bool, List[str]]:
    """Check whether metrics satisfy all SLA constraints.

    Args:
        metrics: Dictionary of metric_name to metric_value.
        constraints: SLA constraints to check against.

    Returns:
        Tuple of (all_pass, violations) where violations is a list of
        human-readable descriptions of which constraints were violated.
    """
    violations: List[str] = []

    for constraint in constraints:
        value = metrics.get(constraint.metric_name)
        if value is None:
            violations.append(
                f"{constraint.metric_name}: metric not found in results"
            )
            continue

        if constraint.direction == "max" and value > constraint.threshold:
            violations.append(
                f"{constraint.metric_name}={value:.4f} exceeds max threshold {constraint.threshold:.4f}"
            )
        elif constraint.direction == "min" and value < constraint.threshold:
            violations.append(
                f"{constraint.metric_name}={value:.4f} below min threshold {constraint.threshold:.4f}"
            )

    return (len(violations) == 0, violations)


__all__ = ["check"]
