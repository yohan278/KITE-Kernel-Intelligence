"""Runtime estimators for predicting operator execution time, energy, and power."""

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.estimator.lookup_table import LookupTableEstimator
from inference_simulator.estimator.roofline import RooflineEstimator

__all__ = [
    "BaseRuntimeEstimator",
    "EstimatorResult",
    "LookupTableEstimator",
    "RooflineEstimator",
]

# Conditional import for SklearnEstimatorBase and subclasses (requires scikit-learn)
try:
    from inference_simulator.estimator.sklearn_base import SklearnEstimatorBase
    __all__.append("SklearnEstimatorBase")
except ImportError:
    pass

try:
    from inference_simulator.estimator.random_forest import RandomForestEstimator
    __all__.append("RandomForestEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.linear_regression import LinearRegressionEstimator
    __all__.append("LinearRegressionEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.ridge import RidgeRegressionEstimator
    __all__.append("RidgeRegressionEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.lasso import LassoRegressionEstimator
    __all__.append("LassoRegressionEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.svr import SVREstimator
    __all__.append("SVREstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.gaussian_process import GaussianProcessEstimator
    __all__.append("GaussianProcessEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.knn import KNNEstimator
    __all__.append("KNNEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.bayesian_linear import BayesianLinearEstimator
    __all__.append("BayesianLinearEstimator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.mlp import MLPEstimator
    __all__.append("MLPEstimator")
except ImportError:
    pass

# XGBoost (requires xgboost)
try:
    from inference_simulator.estimator.xgboost_estimator import XGBoostEstimator
    __all__.append("XGBoostEstimator")
except ImportError:
    pass

# LightGBM (requires lightgbm)
try:
    from inference_simulator.estimator.lightgbm_estimator import LightGBMEstimator
    __all__.append("LightGBMEstimator")
except ImportError:
    pass

# Multi-output wrapper
try:
    from inference_simulator.estimator.multi_output import MultiOutputEstimatorWrapper
    __all__.append("MultiOutputEstimatorWrapper")
except ImportError:
    pass

# Model comparison utilities
try:
    from inference_simulator.estimator.model_comparison import (
        compare_estimators,
        cross_model_evaluation,
        pick_best_estimator,
    )
    __all__.extend(["compare_estimators", "cross_model_evaluation", "pick_best_estimator"])
except ImportError:
    pass

# LUT generation and lookup
try:
    from inference_simulator.estimator.lut_generator import LUTGenerator
    __all__.append("LUTGenerator")
except ImportError:
    pass

try:
    from inference_simulator.estimator.lut_lookup import LUTLookup
    __all__.append("LUTLookup")
except ImportError:
    pass

# Tool distribution fitting (requires scipy)
try:
    from inference_simulator.estimator.tool_distribution import ToolDistributionFitter
    __all__.append("ToolDistributionFitter")
except ImportError:
    pass

# Per-operator estimator (Vidur-inspired)
try:
    from inference_simulator.estimator.per_operator_estimator import PerOperatorEstimator
    __all__.append("PerOperatorEstimator")
except ImportError:
    pass

# Prediction cache
try:
    from inference_simulator.estimator.prediction_cache import PredictionCache
    __all__.append("PredictionCache")
except ImportError:
    pass

# PPI-rectified estimator (requires ppi-python)
try:
    from inference_simulator.estimator.ppi_rectifier import PPIRectifiedEstimator, RectifiedResult
    __all__.extend(["PPIRectifiedEstimator", "RectifiedResult"])
except ImportError:
    pass
