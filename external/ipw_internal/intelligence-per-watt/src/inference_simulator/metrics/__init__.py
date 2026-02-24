"""Metrics collection and computation for inference simulation."""

from inference_simulator.metrics.collector import MetricsCollector, SimulationMetrics

__all__ = [
    "MetricsCollector",
    "SimulationMetrics",
]

# PPI-rectified validation metrics (requires ppi-python)
try:
    from inference_simulator.metrics.ppi_validation import (
        RealServingMeasurements,
        RectifiedSimulationMetrics,
        SimulatedLatencies,
        rectify_simulation_metrics,
    )
    __all__.extend([
        "RealServingMeasurements",
        "RectifiedSimulationMetrics",
        "SimulatedLatencies",
        "rectify_simulation_metrics",
    ])
except ImportError:
    pass
