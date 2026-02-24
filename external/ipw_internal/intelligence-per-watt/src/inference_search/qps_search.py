"""Binary search over QPS to find maximum sustainable throughput."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Sequence

from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec, WorkloadSpec

from inference_search.oracle import SimulatorOracle
from inference_search.sla_checker import check
from inference_search.types import ConfigurationResult, SLAConstraint

logger = logging.getLogger(__name__)


def search(
    model_spec: ModelSpec,
    hardware_spec: HardwareSpec,
    inference_spec: InferenceSpec,
    workload_spec: WorkloadSpec,
    sla_constraints: Sequence[SLAConstraint],
    simulator: SimulatorOracle,
    min_qps: float = 0.1,
    max_qps: float = 1000.0,
    tolerance: float = 0.5,
) -> ConfigurationResult:
    """Binary search for the maximum QPS that satisfies all SLA constraints.

    At each iteration, the midpoint QPS is tested by running the simulator.
    If all SLA constraints pass, the search moves to higher QPS.
    If any constraint fails, it moves to lower QPS.
    The search terminates when the QPS range is smaller than tolerance.

    Args:
        model_spec: Model architecture to evaluate.
        hardware_spec: Hardware target.
        inference_spec: Serving configuration.
        workload_spec: Base workload pattern (qps field will be overridden).
        sla_constraints: SLA constraints that must hold.
        simulator: Oracle to evaluate configurations.
        min_qps: Lower bound of QPS search range.
        max_qps: Upper bound of QPS search range.
        tolerance: Stop when range < tolerance.

    Returns:
        ConfigurationResult with the maximum sustainable QPS and metrics at that point.
    """
    best_qps = 0.0
    best_metrics = {}
    best_violations: list[str] = []
    sim_count = 0

    lo = min_qps
    hi = max_qps

    # First check if even the minimum QPS passes
    wl = _with_qps(workload_spec, lo)
    metrics = simulator.simulate(model_spec, hardware_spec, inference_spec, wl)
    sim_count += 1
    passed, violations = check(metrics, sla_constraints)

    if not passed:
        # Cannot sustain even minimum QPS
        return ConfigurationResult(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            max_qps=0.0,
            metrics=metrics,
            sla_violations=violations,
        )

    best_qps = lo
    best_metrics = metrics

    while (hi - lo) >= tolerance:
        mid = (lo + hi) / 2.0
        wl = _with_qps(workload_spec, mid)
        metrics = simulator.simulate(model_spec, hardware_spec, inference_spec, wl)
        sim_count += 1

        passed, violations = check(metrics, sla_constraints)

        if passed:
            best_qps = mid
            best_metrics = metrics
            best_violations = []
            lo = mid
        else:
            best_violations = violations
            hi = mid

    logger.debug(
        "QPS search converged: max_qps=%.2f after %d simulations",
        best_qps,
        sim_count,
    )

    return ConfigurationResult(
        model_spec=model_spec,
        hardware_spec=hardware_spec,
        inference_spec=inference_spec,
        max_qps=best_qps,
        metrics=best_metrics,
        sla_violations=[],  # Best result always passes SLA
    )


def _with_qps(workload_spec: WorkloadSpec, qps: float) -> WorkloadSpec:
    """Create a new WorkloadSpec with the given QPS."""
    # WorkloadSpec is frozen, so use dataclasses.replace
    return replace(workload_spec, qps=qps)


__all__ = ["search"]
