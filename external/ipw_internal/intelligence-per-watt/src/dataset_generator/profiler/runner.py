"""Profiling runner — orchestrates a full profiling run."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional

from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.results import ProfilingResult
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.output import ProfilingOutputWriter


class ProfilingRunner:
    """Orchestrates a full operator-profiling run.

    Runs selected profilers (token_ops, attention, agentic) for a given
    model × hardware × precision configuration and writes CSV outputs.
    """

    # Available profiler names mapped to their classes
    PROFILER_REGISTRY: Dict[str, str] = {
        "token_ops": "dataset_generator.profiler.token_ops.TokenOpProfiler",
        "attention": "dataset_generator.profiler.attention.AttentionProfiler",
        "agentic": "dataset_generator.profiler.agentic.AgenticProfiler",
        "communication": "dataset_generator.profiler.communication.CommunicationProfiler",
        "moe": "dataset_generator.profiler.moe.MoEProfiler",
        "ssm": "dataset_generator.profiler.ssm.SSMProfiler",
        "sampling": "dataset_generator.profiler.sampling.SamplingProfiler",
        "mtp": "dataset_generator.profiler.mtp.MTPProfiler",
        "cpu_host": "dataset_generator.profiler.cpu_host.CPUHostProfiler",
        "vllm_engine": "dataset_generator.profiler.vllm_engine.VLLMEngineProfiler",
    }

    def __init__(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        sweep_config: Optional[SweepConfig] = None,
        output_dir: Optional[Path] = None,
        precision: str = "fp16",
    ) -> None:
        self.model_spec = model_spec
        self.hardware_spec = hardware_spec
        self.sweep_config = sweep_config or SweepConfig()
        self.precision = precision

        if output_dir is None:
            output_dir = Path("data/profiles")
        # Organize by model / hardware / precision
        model_slug = model_spec.model_id.replace("/", "_")
        hw_slug = hardware_spec.name.replace(" ", "_").lower()
        self.output_dir = output_dir / model_slug / hw_slug / precision
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._writer = ProfilingOutputWriter()

    def run(
        self, profilers: Optional[List[str]] = None
    ) -> ProfilingResult:
        """Run all selected profilers and write CSV outputs.

        Args:
            profilers: List of profiler names to run. If None, runs all.

        Returns:
            ProfilingResult aggregating all measurements.
        """
        if profilers is None:
            profilers = list(self.PROFILER_REGISTRY.keys())

        all_measurements: List[OperatorMeasurement] = []
        errors: Dict[str, str] = {}
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for profiler_name in profilers:
            if profiler_name not in self.PROFILER_REGISTRY:
                print(f"Unknown profiler: {profiler_name}, skipping")
                continue

            print(f"Running {profiler_name} profiler...")
            try:
                profiler = self._load_profiler(profiler_name)
                measurements = profiler.profile(
                    self.model_spec, self.hardware_spec, self.sweep_config,
                    precision=self.precision,
                )
                all_measurements.extend(measurements)
                self._write_output(profiler_name, measurements)
                print(f"  {profiler_name}: {len(measurements)} measurements")
            except NotImplementedError:
                print(f"  {profiler_name}: not implemented, skipping")
            except Exception as e:
                errors[profiler_name] = str(e)
                print(f"  {profiler_name}: ERROR — {e}")

        result = ProfilingResult(
            model_spec=self.model_spec,
            hardware_spec=self.hardware_spec,
            precision=self.precision,
            timestamp=timestamp,
            measurements=all_measurements,
            metadata={
                "output_dir": str(self.output_dir),
                "errors": errors,
            },
        )

        print(
            f"Profiling complete: {len(all_measurements)} total measurements "
            f"in {self.output_dir}"
        )
        return result

    def _load_profiler(self, name: str):
        """Dynamically import and instantiate a profiler."""
        module_path = self.PROFILER_REGISTRY[name]
        module_name, class_name = module_path.rsplit(".", 1)

        import importlib
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls()

    def _write_output(
        self, profiler_name: str, measurements: List[OperatorMeasurement]
    ) -> None:
        """Write measurements to the appropriate CSV file."""
        if not measurements:
            return

        if profiler_name == "token_ops":
            self._writer.write_token_ops(
                measurements, self.output_dir / "token_ops.csv"
            )
        elif profiler_name == "attention":
            self._writer.write_attention(
                measurements, self.output_dir / "attention.csv"
            )
        elif profiler_name == "agentic":
            self._writer.write_agentic(
                measurements, self.output_dir / "agentic.csv"
            )
        elif profiler_name == "communication":
            self._writer.write_communication(
                measurements, self.output_dir / "communication.csv"
            )
        elif profiler_name == "moe":
            self._writer.write_moe(
                measurements, self.output_dir / "moe.csv"
            )
        elif profiler_name == "ssm":
            self._writer.write_ssm(
                measurements, self.output_dir / "ssm.csv"
            )
        elif profiler_name == "sampling":
            self._writer.write_sampling(
                measurements, self.output_dir / "sampling.csv"
            )
        elif profiler_name == "mtp":
            self._writer.write_mtp(
                measurements, self.output_dir / "mtp.csv"
            )
        elif profiler_name == "cpu_host":
            self._writer.write_cpu_host(
                measurements, self.output_dir / "cpu_host.csv"
            )
        elif profiler_name == "vllm_engine":
            self._writer.write_token_ops(
                measurements, self.output_dir / "vllm_engine.csv"
            )
