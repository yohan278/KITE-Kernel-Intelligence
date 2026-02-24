"""Abstract base class for runtime estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from inference_simulator.types.operators import OperatorCategory


@dataclass(frozen=True)
class EstimatorResult:
    """Prediction result from a runtime estimator.

    Attributes:
        time_s: Predicted execution time in seconds.
        energy_j: Predicted energy consumption in joules (None if unavailable).
        power_w: Predicted power draw in watts (None if unavailable).
    """

    time_s: float
    energy_j: Optional[float] = None
    power_w: Optional[float] = None


class BaseRuntimeEstimator(ABC):
    """Abstract base for operator runtime estimation.

    Estimators predict (time, energy, power) for a given operator and input
    dimensions. Implementations range from simple lookup tables to ML models
    trained on profiling data.
    """

    @abstractmethod
    def estimate(
        self,
        operator_category: OperatorCategory,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Predict runtime, energy, and power for an operator invocation.

        Args:
            operator_category: The category of the operator to estimate.
            batch_size: Batch size for the invocation.
            seq_len: Sequence length for the invocation.
            **kwargs: Additional model/hardware dimensions (hidden_dim, num_heads, etc.).

        Returns:
            EstimatorResult with predicted time, energy, and power.
        """
        raise NotImplementedError

    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether this estimator has been trained/loaded with data."""
        raise NotImplementedError

    def estimate_prefill(
        self,
        batch_size: int,
        seq_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Convenience: estimate prefill phase (attention + linear ops).

        Default implementation sums estimates across prefill-relevant categories.
        Subclasses may override for more accurate aggregate predictions.
        """
        categories = [
            OperatorCategory.LINEAR,
            OperatorCategory.ATTENTION_PREFILL,
            OperatorCategory.NORMALIZATION,
            OperatorCategory.ACTIVATION,
        ]
        total_time = 0.0
        total_energy = 0.0
        total_power_samples = []

        for cat in categories:
            result = self.estimate(cat, batch_size, seq_len, **kwargs)
            total_time += result.time_s
            if result.energy_j is not None:
                total_energy += result.energy_j
            if result.power_w is not None:
                total_power_samples.append(result.power_w)

        avg_power = (
            sum(total_power_samples) / len(total_power_samples)
            if total_power_samples
            else None
        )

        return EstimatorResult(
            time_s=total_time,
            energy_j=total_energy if total_energy > 0 else None,
            power_w=avg_power,
        )

    def estimate_decode_step(
        self,
        batch_size: int,
        kv_cache_len: int,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Convenience: estimate a single decode step.

        Default implementation estimates attention_decode + linear ops for seq_len=1.
        """
        categories = [
            OperatorCategory.LINEAR,
            OperatorCategory.ATTENTION_DECODE,
            OperatorCategory.NORMALIZATION,
            OperatorCategory.ACTIVATION,
        ]
        total_time = 0.0
        total_energy = 0.0
        total_power_samples = []

        for cat in categories:
            result = self.estimate(cat, batch_size, seq_len=1, kv_cache_len=kv_cache_len, **kwargs)
            total_time += result.time_s
            if result.energy_j is not None:
                total_energy += result.energy_j
            if result.power_w is not None:
                total_power_samples.append(result.power_w)

        avg_power = (
            sum(total_power_samples) / len(total_power_samples)
            if total_power_samples
            else None
        )

        return EstimatorResult(
            time_s=total_time,
            energy_j=total_energy if total_energy > 0 else None,
            power_w=avg_power,
        )
