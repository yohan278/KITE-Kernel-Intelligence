"""Confidence-aware SLA checking using PPI-rectified simulation metrics."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from inference_search.types import SLAConstraint
from inference_simulator.metrics.ppi_validation import RectifiedSimulationMetrics


def check_with_confidence(
    rectified_metrics: RectifiedSimulationMetrics,
    constraints: Sequence[SLAConstraint],
    confidence_mode: str = "conservative",
) -> Tuple[bool, List[str]]:
    """Check SLA constraints against PPI-rectified metrics.

    Args:
        rectified_metrics: Rectified simulation metrics with CIs.
        constraints: SLA constraints to check.
        confidence_mode: ``"conservative"`` uses CI upper bound for max-constraints
            and CI lower bound for min-constraints. ``"optimistic"`` uses
            the rectified point estimate.

    Returns:
        Tuple of (all_pass, violations).
    """
    if confidence_mode not in ("conservative", "optimistic"):
        raise ValueError(
            f"confidence_mode must be 'conservative' or 'optimistic', got '{confidence_mode}'"
        )

    violations: List[str] = []

    for constraint in constraints:
        # Get point estimate
        point_value = getattr(rectified_metrics, constraint.metric_name, None)
        if point_value is None:
            violations.append(
                f"{constraint.metric_name}: metric not found in rectified metrics"
            )
            continue

        # Get CI if available
        ci_attr = f"{constraint.metric_name}_ci"
        ci = getattr(rectified_metrics, ci_attr, None)

        if confidence_mode == "conservative" and ci is not None:
            ci_lower, ci_upper = ci
            if constraint.direction == "max":
                # Use upper bound — must be within threshold for conservative check
                check_value = ci_upper
            else:
                # Use lower bound — must meet minimum threshold
                check_value = ci_lower
        else:
            check_value = point_value

        if constraint.direction == "max" and check_value > constraint.threshold:
            violations.append(
                f"{constraint.metric_name}={check_value:.4f} exceeds max threshold "
                f"{constraint.threshold:.4f} (mode={confidence_mode})"
            )
        elif constraint.direction == "min" and check_value < constraint.threshold:
            violations.append(
                f"{constraint.metric_name}={check_value:.4f} below min threshold "
                f"{constraint.threshold:.4f} (mode={confidence_mode})"
            )

    return (len(violations) == 0, violations)


__all__ = ["check_with_confidence"]
