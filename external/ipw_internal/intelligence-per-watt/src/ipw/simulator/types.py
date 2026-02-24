"""Type definitions for the LLM inference simulator.

Defines input/output contracts for simulating energy and latency
of LLM inference on arbitrary (hardware, model, workload) combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ConfidenceLevel(str, Enum):
    """Confidence in simulation prediction accuracy.

    HIGH: calibration data exists for this exact (hardware, model) pair.
    MEDIUM: interpolated from similar configs or partial calibration.
    LOW: pure roofline with conservative efficiency assumptions.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WorkloadType(str, Enum):
    """Type of inference workload to simulate."""

    SINGLE_QUERY = "single_query"
    AGENTIC_REASONING = "agentic_reasoning"
    RAG = "rag"
    MULTI_TURN = "multi_turn"


@dataclass(slots=True)
class WorkloadProfile:
    """Description of a workload to simulate.

    For SINGLE_QUERY: just avg_input_tokens and avg_output_tokens.
    For agentic workloads: includes multi-turn and tool-call parameters.
    """

    workload_type: WorkloadType = WorkloadType.SINGLE_QUERY
    avg_input_tokens: int = 500
    avg_output_tokens: int = 200
    avg_turns: int = 1
    avg_tool_calls: int = 0
    avg_tool_latency_seconds: float = 0.0
    context_growth_per_turn: int = 0


@dataclass(slots=True)
class SimulatorConfig:
    """Input configuration for a simulation run.

    Attributes:
        gpu_type: Hardware identifier (must be a key in GPU_TYPE_REGISTRY).
        model_type: Model identifier (must be a key in MODEL_REGISTRY).
        resource_config: Resource allocation (GPU count + CPU count).
        workload: Workload description.
        calibration_path: Optional path to calibration JSON for fitted factors.
    """

    gpu_type: str
    model_type: str
    resource_config: str = "1gpu_8cpu"
    workload: WorkloadProfile = field(default_factory=WorkloadProfile)
    calibration_path: Optional[str] = None


@dataclass(slots=True)
class PhaseResult:
    """Prediction for a single inference phase (prefill or decode)."""

    time_seconds: float = 0.0
    energy_joules: float = 0.0
    flops: float = 0.0
    bytes_transferred: float = 0.0


@dataclass(slots=True)
class SingleInferenceResult:
    """Prediction for a single LLM inference call."""

    prefill: PhaseResult = field(default_factory=PhaseResult)
    decode: PhaseResult = field(default_factory=PhaseResult)
    total_time_seconds: float = 0.0
    total_energy_joules: float = 0.0
    avg_power_watts: float = 0.0
    tokens_per_second: float = 0.0


@dataclass(slots=True)
class SimulationResult:
    """Output of a simulation run.

    Contains the predicted energy and latency with breakdowns,
    plus metadata about which model specs and calibration were used.
    """

    # Top-level predictions
    total_energy_joules: float = 0.0
    total_time_seconds: float = 0.0
    avg_power_watts: float = 0.0

    # Phase breakdown (single-inference or summed across turns)
    prefill_time_seconds: float = 0.0
    prefill_energy_joules: float = 0.0
    decode_time_seconds: float = 0.0
    decode_energy_joules: float = 0.0
    idle_time_seconds: float = 0.0
    idle_energy_joules: float = 0.0

    # Agentic workload details
    num_turns: int = 1
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Confidence and calibration
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    calibration_used: bool = False

    # Hardware/model metadata used for prediction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CalibrationFactors:
    """Learned correction factors bridging roofline theory and real performance.

    Attributes:
        eta_prefill: Actual MFU during prefill (fraction of peak TFLOPS).
        eta_decode: Actual MBU during decode (fraction of peak mem bandwidth).
        alpha: Average power as fraction of TDP.
        energy_per_input_token_j: Fitted energy per input token (from regression).
        energy_per_output_token_j: Fitted energy per output token (from regression).
        intercept_j: Fixed energy overhead per inference (from regression).
        gpu_type: Hardware this calibration was fitted for.
        model_type: Model this calibration was fitted for.
        sample_count: Number of E2E traces used to fit these factors.
        r_squared: Goodness of fit (R^2) from the regression.
    """

    eta_prefill: float = 0.5
    eta_decode: float = 0.6
    alpha: float = 0.65
    energy_per_input_token_j: Optional[float] = None
    energy_per_output_token_j: Optional[float] = None
    intercept_j: Optional[float] = None
    gpu_type: str = ""
    model_type: str = ""
    sample_count: int = 0
    r_squared: Optional[float] = None


__all__ = [
    "CalibrationFactors",
    "ConfidenceLevel",
    "PhaseResult",
    "SimulationResult",
    "SimulatorConfig",
    "SingleInferenceResult",
    "WorkloadProfile",
    "WorkloadType",
]
