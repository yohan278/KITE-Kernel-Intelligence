"""Main simulator orchestrator: ties hardware specs, inference model,
calibration, and workload model together.

Usage:
    from ipw.simulator import InferenceSimulator, SimulatorConfig

    config = SimulatorConfig(
        gpu_type="h100_80gb",
        model_type="qwen3-8b",
        workload=WorkloadProfile(avg_input_tokens=500, avg_output_tokens=200),
    )
    result = InferenceSimulator().simulate(config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ipw.simulator.calibration import CalibrationDB
from ipw.simulator.hardware_specs import (
    HardwareSpecs,
    get_hardware_specs,
    get_model_specs,
)
from ipw.simulator.types import (
    CalibrationFactors,
    ConfidenceLevel,
    SimulationResult,
    SimulatorConfig,
)
from ipw.simulator.workload_model import project

logger = logging.getLogger(__name__)


class InferenceSimulator:
    """Orchestrates inference simulation for arbitrary (hardware, model, workload) combos.

    Workflow:
    1. Resolve hardware specs from HARDWARE_SPECS_REGISTRY.
    2. Resolve model specs from MODEL_REGISTRY.
    3. Load calibration factors from CalibrationDB (if available).
    4. Run workload projection (which calls the analytical inference model).
    5. Attach confidence level and metadata to the result.
    """

    def __init__(self, calibration_db: Optional[CalibrationDB] = None) -> None:
        self._calibration_db = calibration_db or CalibrationDB()

    @property
    def calibration_db(self) -> CalibrationDB:
        return self._calibration_db

    def load_calibration(self, path: Path) -> None:
        """Load a calibration database from JSON."""
        self._calibration_db.load(path)
        logger.info("Loaded %d calibration entries from %s", len(self._calibration_db), path)

    def simulate(self, config: SimulatorConfig) -> SimulationResult:
        """Run a simulation for the given configuration.

        Args:
            config: Simulation input parameters.

        Returns:
            SimulationResult with predicted energy and latency.

        Raises:
            KeyError: If gpu_type or model_type is not recognized.
        """
        # 1. Resolve hardware
        hw = get_hardware_specs(config.gpu_type)

        # 2. Resolve model
        model_specs = get_model_specs(config.model_type)
        active_params_b = model_specs.get("active_params_b", model_specs.get("total_params_b", 1.0))
        total_params_b = model_specs.get("total_params_b", active_params_b)

        # Determine bytes per parameter from quantization
        quantization = model_specs.get("quantization")
        if quantization == "fp8" and hw.peak_fp8_tflops > 0:
            bytes_per_param = 1.0
        elif quantization == "fp8" and hw.peak_fp8_tflops == 0:
            # Hardware doesn't support FP8, fall back to FP16
            bytes_per_param = 2.0
        else:
            bytes_per_param = 2.0

        # Determine number of GPUs from resource config
        num_gpus = _parse_gpu_count(config.resource_config)

        # 3. Load calibration (from config path or DB)
        calibration = None
        confidence = ConfidenceLevel.LOW

        if config.calibration_path:
            try:
                db = CalibrationDB()
                db.load(Path(config.calibration_path))
                calibration = db.get(config.gpu_type, config.model_type)
                if calibration is None:
                    calibration = db.get_or_interpolate(config.gpu_type, config.model_type)
                    if calibration is not None:
                        confidence = ConfidenceLevel.MEDIUM
                else:
                    confidence = ConfidenceLevel.HIGH
            except Exception as e:
                logger.warning("Failed to load calibration from %s: %s", config.calibration_path, e)

        if calibration is None and self._calibration_db:
            calibration = self._calibration_db.get(config.gpu_type, config.model_type)
            if calibration is not None:
                confidence = ConfidenceLevel.HIGH
            else:
                calibration = self._calibration_db.get_or_interpolate(
                    config.gpu_type, config.model_type,
                )
                if calibration is not None:
                    confidence = ConfidenceLevel.MEDIUM

        # 4. Run workload projection
        result = project(
            hw=hw,
            active_params_b=active_params_b,
            workload=config.workload,
            bytes_per_param=bytes_per_param,
            calibration=calibration,
            num_gpus=num_gpus,
        )

        # 5. Attach metadata
        result.confidence = confidence
        result.calibration_used = calibration is not None

        result.metadata = {
            "gpu_type": config.gpu_type,
            "model_type": config.model_type,
            "resource_config": config.resource_config,
            "hardware_name": hw.name,
            "active_params_b": active_params_b,
            "total_params_b": total_params_b,
            "bytes_per_param": bytes_per_param,
            "num_gpus": num_gpus,
            "peak_tflops": hw.peak_tflops * num_gpus,
            "hbm_bandwidth_gb_s": hw.hbm_bandwidth_gb_s * num_gpus,
            "tdp_watts": hw.tdp_watts * num_gpus,
            "quantization": quantization,
            "workload_type": config.workload.workload_type.value,
        }

        return result


def _parse_gpu_count(resource_config: str) -> int:
    """Extract GPU count from resource config string like '1gpu_8cpu'."""
    # Import lazily to avoid hard dependency
    try:
        from grid_eval.config import RESOURCE_CONFIG_REGISTRY, ResourceConfig
        try:
            rc = ResourceConfig(resource_config)
            return RESOURCE_CONFIG_REGISTRY[rc]["gpu_count"]
        except (ValueError, KeyError):
            pass
    except ImportError:
        pass

    # Fallback: parse from string
    rc = resource_config.lower()
    for part in rc.split("_"):
        if part.endswith("gpu"):
            try:
                return int(part.replace("gpu", ""))
            except ValueError:
                pass
    return 1


def format_result(result: SimulationResult) -> str:
    """Format a SimulationResult as a human-readable string."""
    lines = [
        "Simulation Results",
        "=" * 50,
        "",
        f"Total Energy:    {result.total_energy_joules:>10.2f} J",
        f"Total Time:      {result.total_time_seconds:>10.3f} s",
        f"Avg Power:       {result.avg_power_watts:>10.1f} W",
        "",
        "Phase Breakdown:",
        f"  Prefill Time:  {result.prefill_time_seconds:>10.4f} s",
        f"  Prefill Energy:{result.prefill_energy_joules:>10.2f} J",
        f"  Decode Time:   {result.decode_time_seconds:>10.4f} s",
        f"  Decode Energy: {result.decode_energy_joules:>10.2f} J",
    ]

    if result.idle_time_seconds > 0:
        lines.extend([
            f"  Idle Time:     {result.idle_time_seconds:>10.4f} s",
            f"  Idle Energy:   {result.idle_energy_joules:>10.2f} J",
        ])

    if result.num_turns > 1:
        lines.extend([
            "",
            "Workload:",
            f"  Turns:         {result.num_turns:>10d}",
            f"  Input Tokens:  {result.total_input_tokens:>10d}",
            f"  Output Tokens: {result.total_output_tokens:>10d}",
        ])

    lines.extend([
        "",
        f"Confidence:      {result.confidence.value}",
        f"Calibration:     {'yes' if result.calibration_used else 'no (roofline only)'}",
    ])

    if result.metadata:
        lines.extend([
            "",
            "Configuration:",
            f"  Hardware:      {result.metadata.get('hardware_name', 'unknown')}",
            f"  GPUs:          {result.metadata.get('num_gpus', 1)}",
            f"  Model Params:  {result.metadata.get('active_params_b', 0):.1f}B active"
            f" / {result.metadata.get('total_params_b', 0):.1f}B total",
            f"  Quantization:  {result.metadata.get('quantization', 'fp16')}",
            f"  Peak TFLOPS:   {result.metadata.get('peak_tflops', 0):.1f}",
            f"  HBM BW:        {result.metadata.get('hbm_bandwidth_gb_s', 0):.0f} GB/s",
        ])

    return "\n".join(lines)


__all__ = [
    "InferenceSimulator",
    "format_result",
]
