"""Learned operator composition weights for layer-level timing aggregation.

Replaces hardcoded per-layer operator counts (ops_linear_per_layer=5, etc.)
with learned multiplicative correction factors conditioned on model architecture.

Inspired by IrEne (ACL'21, arxiv:2106.01199) which learns aggregation weights
alpha = 1 + tanh(Wx)/tau near 1.0 for hierarchical energy composition.
PIE-P (arxiv:2512.12801) adds communication nodes for TP>1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompositionWeights:
    """Learned per-layer operator composition weights.

    Each weight represents the effective multiplier for that operator category
    within a single transformer layer. Defaults match the hardcoded values in
    SimulatorConfig for backward compatibility.

    Attributes:
        linear_weight: Effective count of linear ops per layer (default: 5).
        norm_weight: Effective count of normalization ops per layer (default: 2).
        activation_weight: Effective count of activation ops per layer (default: 4).
        embedding_weight: Effective count of embedding ops per layer (default: 0).
        communication_weight: AllReduce ops per layer for TP>1 (default: 2, from PIE-P).
        overlap_correction: Kernel overlap correction factor (default: 1.0).
        metadata: Training provenance (model, hardware, training loss, etc.).
    """

    linear_weight: float = 5.0
    norm_weight: float = 2.0
    activation_weight: float = 4.0
    embedding_weight: float = 0.0
    communication_weight: float = 2.0
    overlap_correction: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: object) -> CompositionWeights:
        """Create CompositionWeights from a SimulatorConfig, using its integer defaults."""
        return cls(
            linear_weight=float(getattr(config, "ops_linear_per_layer", 5)),
            norm_weight=float(getattr(config, "ops_norm_per_layer", 2)),
            activation_weight=float(getattr(config, "ops_activation_per_layer", 4)),
            embedding_weight=float(getattr(config, "ops_embedding_per_layer", 0)),
            overlap_correction=float(getattr(config, "overlap_correction", 1.0)),
            metadata={"source": "SimulatorConfig"},
        )

    def to_json(self, path: str | Path) -> None:
        """Serialize to JSON file."""
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> CompositionWeights:
        """Deserialize from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


def load_fused_measurements(
    profiling_dir: Path,
) -> List:
    """Load FUSED_PREFILL measurements from a profiling directory.

    Returns:
        List of OperatorMeasurement with category FUSED_PREFILL.
    """
    from inference_simulator.estimator.sklearn_base import load_csv_measurements
    from inference_simulator.types.operators import OperatorCategory

    fused_csv = Path(profiling_dir) / "fused_prefill.csv"
    if not fused_csv.exists():
        return []
    return load_csv_measurements(fused_csv, OperatorCategory.FUSED_PREFILL)


def load_gt_ttft_samples(
    gt_dir: Path, model_key: str
) -> List[Dict[str, Any]]:
    """Load ground-truth TTFT samples from benchmark results.

    Returns:
        List of dicts with keys: batch_size, seq_len, ttft_s.
    """
    import csv

    gt_dir = Path(gt_dir)
    results: List[Dict[str, Any]] = []

    # Try model-specific CSV first, then generic
    for name in [f"{model_key}_ttft.csv", "ttft_samples.csv"]:
        csv_path = gt_dir / name
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        results.append({
                            "batch_size": int(row["batch_size"]),
                            "seq_len": int(row["seq_len"]),
                            "ttft_s": float(row["ttft_s"]),
                        })
                    except (KeyError, ValueError):
                        continue
            break
    return results


class CompositionModelTrainer:
    """Trains composition weights from fused kernel profiling data.

    Uses scipy L-BFGS-B (5 parameters, trivially fast) to find optimal
    weights that minimize log-scale MSE between predicted per-layer time
    and measured fused kernel time / num_layers.
    """

    def __init__(
        self,
        model_spec: object,
        hardware_spec: object,
        config: Optional[object] = None,
    ) -> None:
        self._model_spec = model_spec
        self._hardware_spec = hardware_spec
        self._config = config

    def fit_from_fused_data(
        self,
        fused_measurements: Sequence,
        token_ops_estimator: object,
        tp_size: int = 1,
    ) -> CompositionWeights:
        """Train composition weights from FUSED_PREFILL measurements.

        For each (batch_size, seq_len) in fused measurements:
          target = measurement.time_s / num_layers
          Predict individual op times from token_ops_estimator
          Optimize weights via L-BFGS-B on log-scale MSE

        Args:
            fused_measurements: FUSED_PREFILL measurements with time_s.
            token_ops_estimator: Fitted estimator for predicting per-op times.
            tp_size: Tensor parallel degree.

        Returns:
            Fitted CompositionWeights.
        """
        from scipy.optimize import minimize

        from inference_simulator.types.operators import OperatorCategory

        num_layers = getattr(self._model_spec, "num_layers", 32)

        # Default weights for bounds computation
        defaults = np.array([5.0, 2.0, 4.0, 0.0, 1.0])  # linear, norm, act, embed, overlap
        lower = defaults * 0.5
        upper = defaults * 2.0
        # Clamp lower bounds to non-negative
        lower = np.maximum(lower, 0.0)
        # Overlap correction in [0.5, 1.0]
        lower[4] = 0.5
        upper[4] = 1.0

        bounds = list(zip(lower, upper))

        # Pre-compute per-op times for each measurement
        targets = []
        op_times_per_sample = []

        for m in fused_measurements:
            target = m.time_s / num_layers
            if target <= 0:
                continue

            # Predict individual operator times
            bs, sl = m.batch_size, m.seq_len
            try:
                t_linear = token_ops_estimator.estimate(
                    OperatorCategory.LINEAR, batch_size=bs * sl, seq_len=1
                ).time_s
                t_norm = token_ops_estimator.estimate(
                    OperatorCategory.NORMALIZATION, batch_size=bs * sl, seq_len=1
                ).time_s
                t_act = token_ops_estimator.estimate(
                    OperatorCategory.ACTIVATION, batch_size=bs * sl, seq_len=1
                ).time_s
                t_embed = token_ops_estimator.estimate(
                    OperatorCategory.EMBEDDING, batch_size=bs * sl, seq_len=1
                ).time_s
            except Exception:
                continue

            targets.append(target)
            op_times_per_sample.append([t_linear, t_norm, t_act, t_embed])

        if len(targets) < 3:
            logger.warning(
                "Too few valid fused measurements (%d < 3); using defaults",
                len(targets),
            )
            return CompositionWeights.from_config(self._config) if self._config else CompositionWeights()

        targets_arr = np.array(targets)
        op_times_arr = np.array(op_times_per_sample)  # shape: (N, 4)

        # Log-scale MSE objective
        log_targets = np.log(np.maximum(targets_arr, 1e-15))

        def objective(params: np.ndarray) -> float:
            w_linear, w_norm, w_act, w_embed, overlap = params
            predicted = (
                op_times_arr[:, 0] * w_linear
                + op_times_arr[:, 1] * w_norm
                + op_times_arr[:, 2] * w_act
                + op_times_arr[:, 3] * w_embed
            ) * overlap
            log_predicted = np.log(np.maximum(predicted, 1e-15))
            return float(np.mean((log_predicted - log_targets) ** 2))

        result = minimize(
            objective,
            x0=defaults,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )

        w = result.x
        comm_weight = 2.0 if tp_size > 1 else 0.0

        weights = CompositionWeights(
            linear_weight=float(w[0]),
            norm_weight=float(w[1]),
            activation_weight=float(w[2]),
            embedding_weight=float(w[3]),
            communication_weight=comm_weight,
            overlap_correction=float(w[4]),
            metadata={
                "source": "fused_data",
                "num_samples": len(targets),
                "final_loss": float(result.fun),
                "converged": result.success,
                "tp_size": tp_size,
            },
        )
        logger.info(
            "Composition weights fitted: linear=%.2f norm=%.2f act=%.2f "
            "embed=%.2f overlap=%.3f (loss=%.4e, %d samples)",
            w[0], w[1], w[2], w[3], w[4], result.fun, len(targets),
        )
        return weights

    def fit_from_gt_benchmark(
        self,
        gt_results: Sequence[Dict[str, Any]],
        estimator: object,
        tp_size: int = 1,
    ) -> CompositionWeights:
        """Fallback: train composition weights from ground-truth TTFT benchmarks.

        Uses TTFT as the target for the full prefill forward pass.

        Args:
            gt_results: List of dicts with batch_size, seq_len, ttft_s.
            estimator: Fitted per-operator estimator.
            tp_size: Tensor parallel degree.

        Returns:
            Fitted CompositionWeights.
        """
        from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement

        # Convert GT results to synthetic fused measurements
        fused = []
        for r in gt_results:
            fused.append(
                OperatorMeasurement(
                    operator_name="fused_prefill_gt",
                    category=OperatorCategory.FUSED_PREFILL,
                    batch_size=r["batch_size"],
                    seq_len=r["seq_len"],
                    time_s=r["ttft_s"],
                )
            )

        if not fused:
            return CompositionWeights()

        return self.fit_from_fused_data(fused, estimator, tp_size)


__all__ = [
    "CompositionWeights",
    "CompositionModelTrainer",
    "load_fused_measurements",
    "load_gt_ttft_samples",
]
