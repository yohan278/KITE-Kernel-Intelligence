"""Bayesian optimization search over inference configurations."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec

from inference_search.oracle import SimulatorOracle
from inference_search.sla_checker import check
from inference_search.types import ConfigurationResult, SLAConstraint, SearchConfig

logger = logging.getLogger(__name__)


class BayesianSearcher:
    """Search for optimal configs using Bayesian optimization.

    Encodes the discrete configuration space (model x hardware x inference_spec)
    as bounded continuous variables and decodes to the nearest valid discrete
    configuration for each evaluation. Uses the ``bayesian-optimization`` library.
    """

    def __init__(
        self,
        oracle: SimulatorOracle,
        search_config: SearchConfig,
        configs: Sequence[Tuple[ModelSpec, HardwareSpec, InferenceSpec]] | None = None,
    ) -> None:
        self._oracle = oracle
        self._config = search_config
        if configs is not None:
            self._configs = list(configs)
        else:
            from inference_search.enumerator import enumerate_configurations
            self._configs = enumerate_configurations(search_config)

    def search(self) -> List[ConfigurationResult]:
        """Run Bayesian optimization over the config space.

        Returns:
            List of ConfigurationResult for each evaluated configuration.
        """
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            logger.warning(
                "bayesian-optimization not installed; falling back to exhaustive search"
            )
            return self._exhaustive_fallback()

        if not self._configs:
            return []

        n_configs = len(self._configs)
        targets = self._config.optimization_targets

        def objective(config_idx: float) -> float:
            idx = self._clamp_index(config_idx, n_configs)
            model, hw, inf = self._configs[idx]

            wl = replace(self._config.workload_spec, qps=1.0)
            metrics = self._oracle.simulate(model, hw, inf, wl)
            passed, _ = check(metrics, self._config.sla_constraints)
            if not passed:
                return -1e6

            # Weighted sum of optimization targets
            score = 0.0
            for target in targets:
                val = metrics.get(target, 0.0)
                # Maximize throughput/ipw/ipj, minimize latency/cost/power
                if "throughput" in target or target in ("ipw", "ipj"):
                    score += val
                else:
                    score -= val
            return score

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={"config_idx": (0.0, float(n_configs - 1))},
            random_state=42,
            allow_duplicate_points=True,
        )

        # Number of iterations: explore then exploit
        n_init = min(5, n_configs)
        n_iter = min(25, n_configs * 2)

        optimizer.maximize(init_points=n_init, n_iter=n_iter)

        # Collect unique configs that were evaluated and return results
        seen_indices: set[int] = set()
        results: List[ConfigurationResult] = []

        for params, _ in zip(optimizer.res, optimizer.res):
            idx = self._clamp_index(params["params"]["config_idx"], n_configs)
            if idx in seen_indices:
                continue
            seen_indices.add(idx)

            model, hw, inf = self._configs[idx]
            result = self._evaluate_config(model, hw, inf)
            results.append(result)

        return results

    def _evaluate_config(
        self,
        model: ModelSpec,
        hw: HardwareSpec,
        inf: InferenceSpec,
    ) -> ConfigurationResult:
        """Evaluate a single config with QPS binary search."""
        from inference_search.qps_search import search as qps_search

        return qps_search(
            model_spec=model,
            hardware_spec=hw,
            inference_spec=inf,
            workload_spec=self._config.workload_spec,
            sla_constraints=self._config.sla_constraints,
            simulator=self._oracle,
        )

    def _exhaustive_fallback(self) -> List[ConfigurationResult]:
        """Fall back to exhaustive evaluation of all configs."""
        results = []
        for model, hw, inf in self._configs:
            results.append(self._evaluate_config(model, hw, inf))
        return results

    @staticmethod
    def _clamp_index(value: float, n: int) -> int:
        """Clamp a continuous value to the nearest valid discrete index."""
        return max(0, min(n - 1, round(value)))


__all__ = ["BayesianSearcher"]
