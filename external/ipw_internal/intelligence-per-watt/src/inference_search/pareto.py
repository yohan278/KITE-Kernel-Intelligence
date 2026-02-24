"""Pareto frontier computation for multi-objective optimization."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from inference_search.types import ConfigurationResult


def compute(
    results: Sequence[ConfigurationResult],
    objectives: Sequence[Tuple[str, str]],
) -> List[ConfigurationResult]:
    """Compute the Pareto frontier for multiple objectives.

    A configuration is Pareto-dominated if another configuration is
    strictly better on ALL objectives simultaneously. The frontier
    contains all non-dominated configurations.

    Args:
        results: List of configuration results to analyze.
        objectives: List of (metric_name, direction) tuples where
                    direction is "minimize" or "maximize".

    Returns:
        List of non-dominated ConfigurationResult instances,
        sorted by the first objective.
    """
    if not results or not objectives:
        return list(results)

    non_dominated: List[ConfigurationResult] = []

    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            if _dominates(other, candidate, objectives):
                dominated = True
                break
        if not dominated:
            non_dominated.append(candidate)

    # Sort by first objective
    if non_dominated and objectives:
        metric_name, direction = objectives[0]
        reverse = direction == "maximize"
        non_dominated.sort(
            key=lambda r: r.metrics.get(metric_name, float("inf")),
            reverse=reverse,
        )

    return non_dominated


def compute_2d(
    results: Sequence[ConfigurationResult],
    x_metric: str,
    y_metric: str,
    x_dir: str = "minimize",
    y_dir: str = "minimize",
) -> List[ConfigurationResult]:
    """Compute the 2D Pareto frontier.

    Convenience wrapper for the common case of two objectives.

    Args:
        results: List of configuration results.
        x_metric: First metric name.
        y_metric: Second metric name.
        x_dir: Direction for x_metric ("minimize" or "maximize").
        y_dir: Direction for y_metric ("minimize" or "maximize").

    Returns:
        Non-dominated configurations sorted by x_metric.
    """
    return compute(results, [(x_metric, x_dir), (y_metric, y_dir)])


def _dominates(
    a: ConfigurationResult,
    b: ConfigurationResult,
    objectives: Sequence[Tuple[str, str]],
) -> bool:
    """Check if configuration `a` dominates configuration `b`.

    `a` dominates `b` if `a` is at least as good as `b` on all objectives,
    and strictly better on at least one.

    Args:
        a: Potentially dominating configuration.
        b: Potentially dominated configuration.
        objectives: List of (metric_name, direction) tuples.

    Returns:
        True if a dominates b.
    """
    at_least_as_good_all = True
    strictly_better_any = False

    for metric_name, direction in objectives:
        a_val = a.metrics.get(metric_name, float("inf"))
        b_val = b.metrics.get(metric_name, float("inf"))

        if direction == "minimize":
            if a_val > b_val:
                at_least_as_good_all = False
                break
            if a_val < b_val:
                strictly_better_any = True
        else:  # maximize
            if a_val < b_val:
                at_least_as_good_all = False
                break
            if a_val > b_val:
                strictly_better_any = True

    return at_least_as_good_all and strictly_better_any


__all__ = ["compute", "compute_2d"]
