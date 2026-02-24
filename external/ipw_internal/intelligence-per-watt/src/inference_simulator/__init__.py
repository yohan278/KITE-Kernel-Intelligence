"""Inference Simulator — analytical performance and energy models for LLM inference."""

from inference_simulator.types import (
    ArchitectureType,
    AttentionType,
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    OperatorCategory,
    OperatorMeasurement,
    ProfilingResult,
    SLASpec,
    WorkloadSpec,
)

from inference_simulator.estimator import (
    BaseRuntimeEstimator,
    EstimatorResult,
    LookupTableEstimator,
    RooflineEstimator,
)

__all__ = [
    "ArchitectureType",
    "AttentionType",
    "BaseRuntimeEstimator",
    "EstimatorResult",
    "HardwareSpec",
    "InferenceSpec",
    "LookupTableEstimator",
    "ModelSpec",
    "OperatorCategory",
    "OperatorMeasurement",
    "ProfilingResult",
    "RooflineEstimator",
    "SLASpec",
    "WorkloadSpec",
]

# Conditional import for RandomForestEstimator (requires scikit-learn)
try:
    from inference_simulator.estimator import RandomForestEstimator
    __all__.append("RandomForestEstimator")
except ImportError:
    pass
