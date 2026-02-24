"""LLM Inference Simulator.

Predicts energy usage and completion time for arbitrary
(hardware, model, workload) combinations using a hybrid
analytical + calibrated model.

Usage:
    from ipw.simulator import InferenceSimulator, SimulatorConfig, WorkloadProfile

    config = SimulatorConfig(
        gpu_type="h100_80gb",
        model_type="qwen3-8b",
        workload=WorkloadProfile(avg_input_tokens=500, avg_output_tokens=200),
    )
    result = InferenceSimulator().simulate(config)
    print(f"Predicted energy: {result.total_energy_joules:.2f} J")
    print(f"Predicted time: {result.total_time_seconds:.3f} s")
"""

from ipw.simulator.calibration import CalibrationDB, fit_from_grid_eval, fit_from_phase_regression
from ipw.simulator.hardware_specs import (
    HARDWARE_SPECS_REGISTRY,
    HardwareSpecs,
    get_hardware_specs,
    get_model_specs,
)
from ipw.simulator.inference_model import (
    BottomUpEnergyBreakdown,
    DEFAULT_ALPHA,
    DEFAULT_ETA_DECODE,
    DEFAULT_ETA_PREFILL,
    OVERHEAD_BASE,
    OVERHEAD_BETA,
    OVERHEAD_DECAY,
    SYSTEM_OVERHEAD_MULTIPLIER,
    batch_overhead_multiplier,
    estimate_alpha_for_model_size,
    estimate_decode,
    estimate_eta_for_model_size,
    estimate_power,
    estimate_prefill,
    estimate_prefill_batch_aware,
    estimate_prefill_bottomup,
    predict,
)
from ipw.simulator.simulator import InferenceSimulator, format_result
from ipw.simulator.types import (
    CalibrationFactors,
    ConfidenceLevel,
    PhaseResult,
    SimulationResult,
    SimulatorConfig,
    SingleInferenceResult,
    WorkloadProfile,
    WorkloadType,
)
from ipw.simulator.workload_model import (
    WorkloadDistribution,
    fit_workload_distribution,
    project,
)

__all__ = [
    # Main orchestrator
    "InferenceSimulator",
    "format_result",
    # Config and result types
    "CalibrationFactors",
    "ConfidenceLevel",
    "PhaseResult",
    "SimulationResult",
    "SimulatorConfig",
    "SingleInferenceResult",
    "WorkloadProfile",
    "WorkloadType",
    # Hardware
    "HARDWARE_SPECS_REGISTRY",
    "HardwareSpecs",
    "get_hardware_specs",
    "get_model_specs",
    # Inference model
    "BottomUpEnergyBreakdown",
    "DEFAULT_ALPHA",
    "DEFAULT_ETA_DECODE",
    "DEFAULT_ETA_PREFILL",
    "OVERHEAD_BASE",
    "OVERHEAD_BETA",
    "OVERHEAD_DECAY",
    "SYSTEM_OVERHEAD_MULTIPLIER",
    "batch_overhead_multiplier",
    "estimate_alpha_for_model_size",
    "estimate_decode",
    "estimate_eta_for_model_size",
    "estimate_power",
    "estimate_prefill",
    "estimate_prefill_batch_aware",
    "estimate_prefill_bottomup",
    "predict",
    # Calibration
    "CalibrationDB",
    "fit_from_grid_eval",
    "fit_from_phase_regression",
    # Workload model
    "WorkloadDistribution",
    "fit_workload_distribution",
    "project",
]
