"""Inference Search — configuration search for optimal (model, hardware, inference_spec) triples."""

from inference_search.bayesian_search import BayesianSearcher
from inference_search.ml_oracle import MLBackedOracle
from inference_search.oracle import RooflineOracle, SimulatorOracle
from inference_search.types import (
    ConfigurationResult,
    SLAConstraint,
    SearchConfig,
    SearchResult,
)

__all__ = [
    "BayesianSearcher",
    "ConfigurationResult",
    "MLBackedOracle",
    "RooflineOracle",
    "SLAConstraint",
    "SearchConfig",
    "SearchResult",
    "SimulatorOracle",
]

# PPI confidence-aware SLA checking (requires ppi-python)
try:
    from inference_search.ppi_sla_checker import check_with_confidence
    __all__.append("check_with_confidence")
except ImportError:
    pass
