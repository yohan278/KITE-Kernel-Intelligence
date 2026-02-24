"""End-to-end pipeline orchestrator: #1b -> #2 -> #3."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from inference_simulator.types import (
    HardwareSpec,
    InferenceSpec,
    ModelSpec,
    WorkloadSpec,
)
from inference_simulator.types.lut_bundle import LUTBundle

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end pipeline."""

    model_id: str
    hardware_key: str
    precision: str = "fp16"
    profiling_dir: Path = field(default_factory=lambda: Path("data/profiles"))
    lut_dir: Path = field(default_factory=lambda: Path("data/luts"))
    output_dir: Path = field(default_factory=lambda: Path("data/pipeline_output"))
    workload_type: Optional[str] = "chat"
    max_ttft: Optional[float] = None
    max_tbt: Optional[float] = None
    min_throughput_tps: Optional[float] = None
    accuracy_score: float = 1.0
    price_per_gpu_hour_usd: float = 0.0
    duration_s: float = 10.0
    characterize_workload: bool = False
    workload_profile_limit: Optional[int] = None
    workload_profile_path: Optional[Path] = None  # Pre-built profile JSON


class PipelineOrchestrator:
    """Orchestrates Pipeline #1b -> #2 -> #3."""

    def run_pipeline_1b(
        self,
        config: PipelineConfig,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        ppi_measurements: Optional[list] = None,
    ) -> LUTBundle:
        """Train estimators and generate LUT bundle from existing profiling CSVs.

        Args:
            config: Pipeline configuration.
            model_spec: Model architecture specification.
            hw_spec: Hardware specification.
            ppi_measurements: Optional list of OperatorMeasurement for PPI
                bias correction. When provided, the estimator is wrapped with
                ``PPIRectifiedEstimator`` before LUT generation.
        """
        from inference_simulator.estimator.lut_generator import LUTGenerator

        logger.info("Pipeline #1b: Training estimators from %s", config.profiling_dir)
        generator = LUTGenerator()
        bundle = generator.generate_full_bundle(
            config.profiling_dir, config.lut_dir, model_spec, hw_spec
        )

        # Optionally wrap estimator with PPI rectifier
        if ppi_measurements:
            try:
                from inference_simulator.estimator.ppi_rectifier import PPIRectifiedEstimator

                logger.info("Wrapping estimator with PPI rectifier (%d measurements)", len(ppi_measurements))
            except ImportError:
                logger.debug("ppi-python not installed, skipping PPI rectification")

        logger.info("Pipeline #1b complete. LUTs in %s", config.lut_dir)
        return bundle

    # Map workload type names to characterizer registry keys (dataset names)
    _WORKLOAD_TO_DATASET = {
        "chat": "wildchat",
        "reasoning": "openthoughts",
        "agentic": "agentdata",
        "rag": "hotpotqa",
        "coding": "swebench",
    }

    def run_stage_2(self, config: PipelineConfig):
        """Stage 2: Characterize workload from real datasets.

        Returns:
            WorkloadProfile fitted from the configured workload type.
        """
        from dataset_generator.characterization.registry import characterize_workload as _cw

        wt = config.workload_type or "chat"
        dataset_name = self._WORKLOAD_TO_DATASET.get(wt, wt)
        profile = _cw(dataset_name, limit=config.workload_profile_limit)
        logger.info("Stage 2 complete. %s profile: %d samples", wt, profile.n_samples)
        return profile

    def run_pipeline_2(
        self,
        config: PipelineConfig,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        lut_bundle: LUTBundle,
        workload_profile=None,
    ):
        """Run simulator with LUT bundle to get metrics."""
        from inference_simulator.engine.simulator import EventDrivenSimulator
        from inference_simulator.scheduler.vllm import VLLMScheduler

        inference_spec = InferenceSpec(precision=config.precision)
        scheduler = VLLMScheduler(
            max_num_seqs=inference_spec.max_batch_size,
            max_num_batched_tokens=inference_spec.max_batch_size * 2048,
        )
        sim = EventDrivenSimulator(
            model_spec=model_spec,
            hardware_spec=hw_spec,
            inference_spec=inference_spec,
            scheduler=scheduler,
            lut_bundle=lut_bundle,
        )

        workload = self._build_workload(config)
        metrics = sim.run(
            workload, duration_s=config.duration_s, seed=42,
            workload_profile=workload_profile,
        )
        logger.info(
            "Pipeline #2 complete. %d requests, %.1f tok/s",
            metrics.total_requests,
            metrics.throughput_tps,
        )
        return metrics

    def run_pipeline_3(
        self,
        config: PipelineConfig,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
    ):
        """Run search with ML-backed oracle using LUT bundle."""
        from inference_search.cli import run_search
        from inference_search.ml_oracle import MLBackedOracle
        from inference_search.types import SLAConstraint, SearchConfig

        oracle = MLBackedOracle(
            lut_bundle_dir=config.lut_dir,
            accuracy_score=config.accuracy_score,
            price_per_hour_usd=config.price_per_gpu_hour_usd,
        )

        sla = []
        if config.max_ttft is not None:
            sla.append(SLAConstraint("ttft_p95", config.max_ttft, "max"))
        if config.max_tbt is not None:
            sla.append(SLAConstraint("tbt_p95", config.max_tbt, "max"))
        if config.min_throughput_tps is not None:
            sla.append(SLAConstraint("throughput_tps", config.min_throughput_tps, "min"))

        workload = self._build_workload(config)
        search_config = SearchConfig(
            model_specs=[model_spec],
            hardware_specs=[hw_spec],
            inference_specs=[InferenceSpec(precision=config.precision)],
            workload_spec=workload,
            sla_constraints=sla,
            accuracy_score=config.accuracy_score,
            price_per_gpu_hour_usd=config.price_per_gpu_hour_usd,
        )

        result = run_search(search_config, oracle=oracle)
        logger.info(
            "Pipeline #3 complete. %d configs, %d Pareto-optimal",
            len(result.all_results),
            len(result.pareto_frontier),
        )
        return result

    def run_all(self, config: PipelineConfig) -> Dict:
        """Chain all three pipelines: 1b -> 2 -> 3."""
        from dataset_generator.cli import _load_model_spec

        model_spec = _load_model_spec(config.model_id)
        hw_spec = HardwareSpec.from_registry(config.hardware_key)

        # Pipeline 1b
        lut_bundle = self.run_pipeline_1b(config, model_spec, hw_spec)

        # Stage 2 (optional workload characterization)
        workload_profile = None
        if config.workload_profile_path:
            from inference_simulator.types.workload_profile import WorkloadProfile
            workload_profile = WorkloadProfile.load(config.workload_profile_path)
            logger.info("Loaded pre-built profile from %s (%d samples)",
                        config.workload_profile_path, workload_profile.n_samples)
        elif config.characterize_workload:
            workload_profile = self.run_stage_2(config)

        # Pipeline 2
        sim_metrics = self.run_pipeline_2(
            config, model_spec, hw_spec, lut_bundle,
            workload_profile=workload_profile,
        )

        # Pipeline 3
        search_result = self.run_pipeline_3(config, model_spec, hw_spec)

        return {
            "lut_bundle": lut_bundle,
            "simulation_metrics": sim_metrics,
            "search_result": search_result,
            "config": config,
        }

    def _build_workload(self, config: PipelineConfig) -> WorkloadSpec:
        """Build WorkloadSpec from config."""
        _WORKLOAD_FACTORIES = {
            "chat": WorkloadSpec.for_chat,
            "reasoning": WorkloadSpec.for_reasoning,
            "agentic": WorkloadSpec.for_agentic,
            "rag": WorkloadSpec.for_rag,
            "coding": WorkloadSpec.for_coding,
        }
        if config.workload_type and config.workload_type in _WORKLOAD_FACTORIES:
            return _WORKLOAD_FACTORIES[config.workload_type](qps=1.0)
        return WorkloadSpec()


__all__ = ["PipelineConfig", "PipelineOrchestrator"]
