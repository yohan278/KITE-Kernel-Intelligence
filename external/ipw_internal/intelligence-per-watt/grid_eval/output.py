"""JSONL output writer for grid evaluation results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """Result for a single query evaluation.

    Attributes:
        query_id: Unique identifier for the query
        benchmark: Benchmark name (hle or gaia)
        model: Model name
        agent: Agent type
        gpu_type: GPU hardware type (new API)
        resource_config: Resource allocation config (new API)
        hardware: Hardware configuration (deprecated - for backwards compatibility)
        avg_joules: Average energy consumption in joules (GPU-only, kept for backwards compat)
        gpu_joules: GPU energy consumption in joules
        cpu_joules: CPU energy consumption in joules (from RAPL)
        max_power_watts: Maximum power draw in watts
        latency_seconds: Total latency in seconds
        tools_used: Dictionary of tool names to usage counts
        turns: Number of agent turns
        models_called: Dictionary of model IDs to call counts
        is_correct: Whether the response was correct
        response: Agent response text
        ground_truth: Expected answer
        error: Error message if evaluation failed
    """

    query_id: str
    benchmark: str
    model: str
    agent: str
    gpu_type: str
    resource_config: str
    hardware: str  # Deprecated - kept for backwards compatibility
    avg_joules: float  # GPU-only, kept for backwards compatibility
    gpu_joules: float  # GPU energy
    cpu_joules: float  # CPU energy (RAPL)
    max_power_watts: float
    latency_seconds: float
    tools_used: Dict[str, int]
    turns: int
    models_called: Dict[str, int]
    is_correct: bool
    response: str
    ground_truth: str
    error: Optional[str] = None
    grade: Optional[str] = None
    total_params_b: Optional[float] = None
    active_params_b: Optional[float] = None
    # Per-action energy breakdown
    action_breakdowns: Optional[List[Dict[str, Any]]] = None
    energy_by_action: Optional[Dict[str, float]] = None
    # Batch mode metadata
    batch_size: Optional[int] = None
    concurrency: Optional[int] = None
    energy_amortized: bool = False


@dataclass
class ConfigSummary:
    """Aggregated summary for a single configuration."""

    benchmark: str
    model: str
    agent: str
    gpu_type: str
    resource_config: str
    hardware: str  # Deprecated - kept for backwards compatibility
    num_queries: int
    accuracy: float
    avg_joules: float
    avg_gpu_joules: float
    avg_cpu_joules: float
    avg_latency_seconds: float
    max_power_watts: float
    total_joules: float
    total_gpu_joules: float
    total_cpu_joules: float
    total_latency_seconds: float


@dataclass
class GridMetadata:
    """Metadata for the entire grid evaluation run."""

    gpu_types: List[str]
    resource_configs: List[str]
    benchmarks: List[str]
    models: List[str]
    agents: List[str]
    hardware_configs: List[str]  # Deprecated - kept for backwards compatibility
    queries_per_benchmark: int
    seed: int
    timestamp: str
    total_combinations: int
    total_queries: int


class JSONLWriter:
    """Writer for grid evaluation results in JSONL format.

    Writes results incrementally to JSONL file, then generates
    summary and metadata JSON files on completion.

    Example:
        >>> writer = JSONLWriter(Path("results/run_001"))
        >>> writer.write_query_result(result)
        >>> writer.finalize()  # Writes summary and metadata
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize writer.

        Args:
            output_dir: Directory to write results to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.output_dir / f"results_{timestamp}.jsonl"
        self.summary_path = self.output_dir / f"summary_{timestamp}.json"
        self.metadata_path = self.output_dir / f"metadata_{timestamp}.json"

        self._results: List[QueryResult] = []
        self._file_handle = None

    def __enter__(self) -> "JSONLWriter":
        """Open results file for writing."""
        self._file_handle = open(self.results_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close results file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def write_query_result(self, result: QueryResult) -> None:
        """Write a single query result to JSONL.

        Args:
            result: Query result to write
        """
        self._results.append(result)

        # Convert to dict and write
        result_dict = asdict(result)
        line = json.dumps(result_dict, default=str) + "\n"

        if self._file_handle:
            self._file_handle.write(line)
            self._file_handle.flush()
        else:
            # If not used as context manager, append to file
            with open(self.results_path, "a") as f:
                f.write(line)

    def write_summary(self) -> None:
        """Generate and write aggregated summary."""
        summaries = self._aggregate_results()

        summary_data = {
            "configs": [asdict(s) for s in summaries],
            "total_queries": len(self._results),
            "overall_accuracy": (
                sum(1 for r in self._results if r.is_correct) / len(self._results)
                if self._results
                else 0.0
            ),
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

    def write_metadata(self, metadata: GridMetadata) -> None:
        """Write grid configuration metadata.

        Args:
            metadata: Grid metadata to write
        """
        with open(self.metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

    def finalize(self, metadata: GridMetadata) -> None:
        """Finalize output by writing summary and metadata.

        Args:
            metadata: Grid metadata
        """
        self.write_summary()
        self.write_metadata(metadata)

    def _aggregate_results(self) -> List[ConfigSummary]:
        """Aggregate results by configuration.

        Returns:
            List of ConfigSummary objects, one per unique configuration
        """
        # Group results by config (using new 5-tuple key)
        config_groups: Dict[tuple, List[QueryResult]] = {}
        for result in self._results:
            key = (
                result.benchmark,
                result.model,
                result.agent,
                result.gpu_type,
                result.resource_config,
            )
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(result)

        # Compute summaries
        summaries = []
        for (benchmark, model, agent, gpu_type, resource_config), results in config_groups.items():
            num_queries = len(results)
            num_correct = sum(1 for r in results if r.is_correct)
            total_joules = sum(r.avg_joules for r in results)
            total_gpu_joules = sum(r.gpu_joules for r in results)
            total_cpu_joules = sum(r.cpu_joules for r in results)
            total_latency = sum(r.latency_seconds for r in results)
            max_power = max((r.max_power_watts for r in results), default=0.0)

            # Compute combined hardware string for backwards compatibility
            hardware = f"{gpu_type}/{resource_config}"

            summaries.append(
                ConfigSummary(
                    benchmark=benchmark,
                    model=model,
                    agent=agent,
                    gpu_type=gpu_type,
                    resource_config=resource_config,
                    hardware=hardware,
                    num_queries=num_queries,
                    accuracy=num_correct / num_queries if num_queries > 0 else 0.0,
                    avg_joules=total_joules / num_queries if num_queries > 0 else 0.0,
                    avg_gpu_joules=total_gpu_joules / num_queries if num_queries > 0 else 0.0,
                    avg_cpu_joules=total_cpu_joules / num_queries if num_queries > 0 else 0.0,
                    avg_latency_seconds=(
                        total_latency / num_queries if num_queries > 0 else 0.0
                    ),
                    max_power_watts=max_power,
                    total_joules=total_joules,
                    total_gpu_joules=total_gpu_joules,
                    total_cpu_joules=total_cpu_joules,
                    total_latency_seconds=total_latency,
                )
            )

        return summaries


__all__ = [
    "ConfigSummary",
    "GridMetadata",
    "JSONLWriter",
    "QueryResult",
]
