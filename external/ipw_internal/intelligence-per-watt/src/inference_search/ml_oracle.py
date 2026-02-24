"""ML-backed oracle using Pipeline #1b estimators + EventDrivenSimulator."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec, WorkloadSpec

from inference_search.oracle import RooflineOracle, SimulatorOracle

if TYPE_CHECKING:
    from inference_simulator.metrics.ppi_validation import RealServingMeasurements

logger = logging.getLogger(__name__)


class MLBackedOracle:
    """Oracle using trained Pipeline #1b estimators + EventDrivenSimulator.

    Loads a LUT bundle and runs the Pipeline #2 simulator to evaluate configs.
    Falls back to RooflineOracle if no bundle is loaded or if the simulator
    fails.

    Attributes:
        accuracy_score: Model accuracy score for IPW/IPJ computation.
        price_per_hour_usd: Cost per GPU-hour for cost_per_query_usd.
    """

    def __init__(
        self,
        lut_bundle_dir: Optional[Path] = None,
        accuracy_score: float = 1.0,
        price_per_hour_usd: float = 0.0,
        real_measurements: Optional[RealServingMeasurements] = None,
    ) -> None:
        self.accuracy_score = accuracy_score
        self.price_per_hour_usd = price_per_hour_usd
        self._bundle = None
        self._real_measurements = real_measurements
        self._fallback = RooflineOracle(
            accuracy_score=accuracy_score,
            price_per_hour_usd=price_per_hour_usd,
        )
        if lut_bundle_dir is not None:
            self._load_bundle(lut_bundle_dir)

    def _load_bundle(self, bundle_dir: Path) -> None:
        """Load LUT bundle from a directory."""
        try:
            from inference_simulator.types.lut_bundle import LUTBundle

            base = Path(bundle_dir)
            # Convention: required LUT files are in the bundle directory
            # Support both naming conventions (gpu_ prefix and without)
            def _find_lut(name: str) -> Optional[Path]:
                p = base / name
                if p.exists():
                    return p
                return None

            self._bundle = LUTBundle(
                base_dir=base,
                model_id="",
                hardware_id="",
                quantization="",
                gpu_token_ops_lut=_find_lut("gpu_token_ops.npz") or _find_lut("token_ops.npz"),
                gpu_attention_prefill_lut=_find_lut("gpu_attention_prefill.npz") or _find_lut("attention_prefill.npz"),
                gpu_attention_decode_lut=_find_lut("gpu_attention_decode.npz") or _find_lut("attention_decode.npz"),
                gpu_moe_lut=_find_lut("gpu_moe.npz") or _find_lut("moe.npz"),
                network_lut=_find_lut("network.npz"),
                energy_lut=_find_lut("energy.npz"),
                tool_distributions=_find_lut("tool_distributions.pkl"),
            )
            if not self._bundle.exists():
                logger.warning("LUT bundle at %s missing required files; will use fallback", bundle_dir)
                self._bundle = None
        except (ImportError, Exception) as exc:
            logger.warning("Failed to load LUT bundle from %s: %s", bundle_dir, exc)
            self._bundle = None

    def simulate(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        workload_spec: WorkloadSpec,
    ) -> Dict[str, float]:
        """Run simulator with LUT bundle and return metrics dict.

        If no bundle is loaded, delegates to RooflineOracle. If the simulator
        fails for any reason, also falls back to RooflineOracle.

        Returns the same metric dict format as RooflineOracle (including
        ipw, ipj, cost, and percentile variants).
        """
        if self._bundle is None:
            return self._fallback.simulate(model_spec, hardware_spec, inference_spec, workload_spec)

        try:
            return self._run_simulator(model_spec, hardware_spec, inference_spec, workload_spec)
        except Exception as exc:
            logger.warning("Simulator failed, falling back to roofline: %s", exc)
            return self._fallback.simulate(model_spec, hardware_spec, inference_spec, workload_spec)

    def _run_simulator(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        workload_spec: WorkloadSpec,
    ) -> Dict[str, float]:
        """Run the EventDrivenSimulator with the loaded LUT bundle."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.scheduler.vllm import VLLMScheduler

        scheduler = VLLMScheduler(
            max_num_seqs=inference_spec.max_batch_size,
            max_num_batched_tokens=inference_spec.max_batch_size * 2048,
        )

        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hardware_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
            lut_bundle=self._bundle,
        )

        sim_metrics = sim.run(workload_spec, duration_s=10.0, seed=42)

        # Extract base metrics from SimulationMetrics
        avg_power = sim_metrics.avg_power_w if sim_metrics.avg_power_w > 0 else 1.0
        total_energy = sim_metrics.total_energy_j
        throughput_rps = sim_metrics.throughput_rps

        energy_per_query = total_energy / throughput_rps if throughput_rps > 0 else float("inf")

        ipw = self.accuracy_score / avg_power if avg_power > 0 else 0.0
        ipj = self.accuracy_score / energy_per_query if energy_per_query > 0 and energy_per_query != float("inf") else 0.0

        e2e_base = sim_metrics.e2e_p50 if sim_metrics.e2e_p50 > 0 else 0.0
        num_gpus = inference_spec.num_gpus
        cost_per_query = self.price_per_hour_usd * num_gpus * e2e_base / 3600.0

        result = {
            "ttft_s": sim_metrics.ttft_p50,
            "tbt_s": sim_metrics.tbt_p50,
            "e2e_latency_s": sim_metrics.e2e_p50,
            "throughput_tps": sim_metrics.throughput_tps,
            "throughput_rps": sim_metrics.throughput_rps,
            "total_energy_j": sim_metrics.total_energy_j,
            "avg_power_w": sim_metrics.avg_power_w,
            "ipw": ipw,
            "ipj": ipj,
            "cost_per_query_usd": cost_per_query,
            "energy_per_query_j": energy_per_query,
            "ttft_p50": sim_metrics.ttft_p50,
            "ttft_p90": sim_metrics.ttft_p90,
            "ttft_p95": sim_metrics.ttft_p95,
            "ttft_p99": sim_metrics.ttft_p99,
            "tbt_p50": sim_metrics.tbt_p50,
            "tbt_p90": sim_metrics.tbt_p90,
            "tbt_p95": sim_metrics.tbt_p95,
            "tbt_p99": sim_metrics.tbt_p99,
            "e2e_p50": sim_metrics.e2e_p50,
            "e2e_p90": sim_metrics.e2e_p90,
            "e2e_p95": sim_metrics.e2e_p95,
            "e2e_p99": sim_metrics.e2e_p99,
        }

        # PPI rectification when real measurements are available
        if self._real_measurements is not None:
            try:
                from inference_simulator.metrics.ppi_validation import (
                    SimulatedLatencies,
                    rectify_simulation_metrics,
                )

                import numpy as _np

                # Build simulated latency arrays from the sim_metrics percentiles
                # as proxy for per-request arrays (the simulator aggregates internally)
                n = len(self._real_measurements.ttft_s)
                sim_labeled = SimulatedLatencies(
                    ttft_s=_np.full(n, sim_metrics.ttft_p50),
                    tbt_s=_np.full(n, sim_metrics.tbt_p50),
                    e2e_s=_np.full(n, sim_metrics.e2e_p50),
                )
                # Use full percentile spread as unlabeled predictions
                ttft_unlabeled = _np.array([
                    sim_metrics.ttft_p50, sim_metrics.ttft_p90,
                    sim_metrics.ttft_p95, sim_metrics.ttft_p99,
                ] * max(n, 10))
                tbt_unlabeled = _np.array([
                    sim_metrics.tbt_p50, sim_metrics.tbt_p90,
                    sim_metrics.tbt_p95, sim_metrics.tbt_p99,
                ] * max(n, 10))
                e2e_unlabeled = _np.array([
                    sim_metrics.e2e_p50, sim_metrics.e2e_p90,
                    sim_metrics.e2e_p95, sim_metrics.e2e_p99,
                ] * max(n, 10))

                rectified = rectify_simulation_metrics(
                    self._real_measurements,
                    sim_labeled,
                    ttft_unlabeled,
                    tbt_unlabeled,
                    e2e_unlabeled,
                )

                # Add CI-suffixed keys for each percentile metric
                for metric_prefix in ("ttft", "tbt", "e2e"):
                    for pct in ("p50", "p90", "p95", "p99"):
                        key = f"{metric_prefix}_{pct}"
                        ci_key = f"{key}_ci"
                        ci = getattr(rectified, ci_key, None)
                        if ci is not None:
                            result[f"{key}_ci_lower"] = ci[0]
                            result[f"{key}_ci_upper"] = ci[1]
                        # Override with rectified point estimate
                        rect_val = getattr(rectified, key, None)
                        if rect_val is not None:
                            result[key] = rect_val

            except (ImportError, Exception) as exc:
                logger.debug("PPI rectification skipped: %s", exc)

        return result


__all__ = ["MLBackedOracle"]
