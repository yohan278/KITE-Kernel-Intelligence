"""Type definitions for the inference search package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec, WorkloadSpec


@dataclass(frozen=True)
class SLAConstraint:
    """A single service-level agreement constraint.

    Attributes:
        metric_name: Metric to constrain (e.g., "ttft_s", "throughput_tps", "avg_power_w").
        threshold: The boundary value.
        direction: "max" means metric must be <= threshold;
                   "min" means metric must be >= threshold.
    """

    metric_name: str
    threshold: float
    direction: str  # "max" or "min"

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got '{self.direction}'")


@dataclass
class SearchConfig:
    """Input configuration for a search run.

    Attributes:
        model_specs: List of model architectures to search over.
        hardware_specs: List of hardware targets.
        inference_specs: List of serving configurations.
        workload_spec: Workload pattern to simulate.
        sla_constraints: SLA constraints that must be satisfied.
        optimization_targets: Metric names to optimize on the Pareto frontier.
        duration_s: Simulation duration per configuration in seconds.
    """

    model_specs: List[ModelSpec]
    hardware_specs: List[HardwareSpec]
    inference_specs: List[InferenceSpec]
    workload_spec: WorkloadSpec
    sla_constraints: List[SLAConstraint] = field(default_factory=list)
    optimization_targets: List[str] = field(
        default_factory=lambda: ["ipj", "ipw", "cost_per_query_usd"]
    )
    duration_s: float = 60.0
    search_method: str = "exhaustive"
    accuracy_score: float = 1.0
    price_per_gpu_hour_usd: float = 0.0


@dataclass
class ConfigurationResult:
    """Result for a single (model, hardware, inference_spec) configuration.

    Attributes:
        model_spec: The model architecture evaluated.
        hardware_spec: The hardware target evaluated.
        inference_spec: The serving configuration evaluated.
        max_qps: Maximum sustainable QPS meeting all SLA constraints.
        metrics: All simulator output metrics at max_qps.
        sla_violations: Descriptions of any violated SLA constraints (empty if all pass).
    """

    model_spec: ModelSpec
    hardware_spec: HardwareSpec
    inference_spec: InferenceSpec
    max_qps: float
    metrics: Dict[str, float] = field(default_factory=dict)
    sla_violations: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Aggregate result from a complete search run.

    Attributes:
        all_results: Results for every feasible configuration.
        pareto_frontier: Non-dominated subset of all_results.
        search_config: The search configuration that produced these results.
        total_simulations: Total number of simulator calls made.
        elapsed_seconds: Wall-clock time for the entire search.
    """

    all_results: List[ConfigurationResult]
    pareto_frontier: List[ConfigurationResult]
    search_config: SearchConfig
    total_simulations: int
    elapsed_seconds: float


__all__ = [
    "ConfigurationResult",
    "SLAConstraint",
    "SearchConfig",
    "SearchResult",
]
