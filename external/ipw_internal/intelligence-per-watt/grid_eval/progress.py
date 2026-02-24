"""Progress tracking for grid evaluation runs.

Provides CSV-based tracking of completed, skipped, and failed configurations
to support resumable evaluation runs.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ProgressTracker:
    """Track completed configurations in CSV format.

    This class manages a CSV file that records the status of each
    configuration combination (gpu_type, resource_config, agent, model, benchmark).
    It supports resuming interrupted runs by skipping already-completed
    configurations.

    Example:
        >>> tracker = ProgressTracker(Path("results/progress.csv"))
        >>> if not tracker.is_completed("a100_80gb", "1gpu_8cpu", "react", "qwen3-8b", "gaia"):
        ...     # Run the evaluation
        ...     tracker.mark_completed("a100_80gb", "1gpu_8cpu", "react", "qwen3-8b", "gaia", 100)
    """

    csv_path: Path

    def __post_init__(self) -> None:
        """Initialize CSV file if it doesn't exist."""
        if not self.csv_path.exists():
            self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV with headers."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "config_hash",
            "status",
            "timestamp",
            "gpu_type",
            "resource_config",
            "agent",
            "model",
            "benchmark",
            "num_queries",
            "error_message",
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def get_config_hash(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
    ) -> str:
        """Generate unique hash for configuration.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name

        Returns:
            12-character MD5 hash of the configuration
        """
        key = f"{gpu_type}|{resource_config}|{agent}|{model}|{benchmark}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _read_entries(self) -> dict[str, dict]:
        """Read all entries from CSV as dict keyed by config_hash.

        Returns:
            Dictionary mapping config_hash to row data
        """
        entries = {}
        if not self.csv_path.exists():
            return entries

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries[row["config_hash"]] = row
        return entries

    def is_completed(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
    ) -> bool:
        """Check if configuration already completed.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name

        Returns:
            True if configuration has "completed" status
        """
        config_hash = self.get_config_hash(
            gpu_type, resource_config, agent, model, benchmark
        )
        entries = self._read_entries()
        if config_hash in entries:
            return entries[config_hash]["status"] == "completed"
        return False

    def is_skipped(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
    ) -> bool:
        """Check if configuration was skipped.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name

        Returns:
            True if configuration has "skipped" status
        """
        config_hash = self.get_config_hash(
            gpu_type, resource_config, agent, model, benchmark
        )
        entries = self._read_entries()
        if config_hash in entries:
            return entries[config_hash]["status"] == "skipped"
        return False

    def get_status(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
    ) -> Optional[str]:
        """Get status of a configuration.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name

        Returns:
            Status string ("completed", "skipped", "failed") or None if not found
        """
        config_hash = self.get_config_hash(
            gpu_type, resource_config, agent, model, benchmark
        )
        entries = self._read_entries()
        if config_hash in entries:
            return entries[config_hash]["status"]
        return None

    def _write_entry(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
        status: str,
        num_queries: int = 0,
        error_message: str = "",
    ) -> None:
        """Write an entry to the CSV file.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name
            status: Status string
            num_queries: Number of queries completed
            error_message: Error message if failed
        """
        config_hash = self.get_config_hash(
            gpu_type, resource_config, agent, model, benchmark
        )
        timestamp = datetime.now().isoformat()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    config_hash,
                    status,
                    timestamp,
                    gpu_type,
                    resource_config,
                    agent,
                    model,
                    benchmark,
                    num_queries,
                    error_message,
                ]
            )

    def mark_completed(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
        num_queries: int,
    ) -> None:
        """Mark configuration as completed.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name
            num_queries: Number of queries completed
        """
        self._write_entry(
            gpu_type=gpu_type,
            resource_config=resource_config,
            agent=agent,
            model=model,
            benchmark=benchmark,
            status="completed",
            num_queries=num_queries,
        )

    def mark_skipped(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
        reason: str,
    ) -> None:
        """Mark configuration as skipped (e.g., N/A for GPU constraints).

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name
            reason: Reason for skipping
        """
        self._write_entry(
            gpu_type=gpu_type,
            resource_config=resource_config,
            agent=agent,
            model=model,
            benchmark=benchmark,
            status="skipped",
            error_message=reason,
        )

    def mark_failed(
        self,
        gpu_type: str,
        resource_config: str,
        agent: str,
        model: str,
        benchmark: str,
        error: str,
    ) -> None:
        """Mark configuration as failed with error message.

        Args:
            gpu_type: GPU type name
            resource_config: Resource configuration name
            agent: Agent name
            model: Model name
            benchmark: Benchmark name
            error: Error message
        """
        self._write_entry(
            gpu_type=gpu_type,
            resource_config=resource_config,
            agent=agent,
            model=model,
            benchmark=benchmark,
            status="failed",
            error_message=error,
        )

    def get_summary(self) -> dict[str, int]:
        """Get summary counts of each status.

        Returns:
            Dictionary with counts for each status type
        """
        entries = self._read_entries()
        summary = {"completed": 0, "skipped": 0, "failed": 0, "total": len(entries)}
        for entry in entries.values():
            status = entry["status"]
            if status in summary:
                summary[status] += 1
        return summary


__all__ = ["ProgressTracker"]
