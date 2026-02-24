"""Tracking matrix — monitors dataset x operator profiling completeness."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set

from dataset_generator.pipeline.checklist import OPERATOR_CHECKLIST, _scan_csv_for_operators


class TrackingMatrix:
    """Tracks which (dataset, operator) pairs have profiling data.

    Scans a profiling directory for dataset subdirectories and CSV files,
    building a matrix of dataset x operator completeness.
    """

    def __init__(self, datasets: List[str], profiling_dir: Path):
        self.datasets = datasets
        self.profiling_dir = Path(profiling_dir)
        # dataset -> set of profiled operator names
        self._matrix: Dict[str, Set[str]] = {ds: set() for ds in datasets}
        # All operators flattened
        self._all_operators: List[str] = []
        for ops in OPERATOR_CHECKLIST.values():
            self._all_operators.extend(ops)

    def scan(self) -> None:
        """Scan profiling_dir for dataset subdirectories and populate the matrix."""
        if not self.profiling_dir.is_dir():
            return

        for ds_name in self.datasets:
            ds_dir = self.profiling_dir / ds_name
            if not ds_dir.is_dir():
                # Also check for CSV files at top level containing dataset name
                for csv_file in self.profiling_dir.glob("*.csv"):
                    if ds_name in csv_file.stem:
                        self._matrix[ds_name].update(
                            _scan_csv_for_operators(csv_file)
                        )
                continue

            for csv_file in ds_dir.rglob("*.csv"):
                self._matrix[ds_name].update(_scan_csv_for_operators(csv_file))

    def to_csv(self, path: Path) -> None:
        """Write the tracking matrix to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["operator"] + self.datasets

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for op in self._all_operators:
                row: Dict[str, str] = {"operator": op}
                for ds in self.datasets:
                    row[ds] = "1" if op in self._matrix[ds] else "0"
                writer.writerow(row)

    def to_markdown(self) -> str:
        """Render the tracking matrix as a markdown table."""
        header = "| Operator | " + " | ".join(self.datasets) + " |"
        separator = "| --- | " + " | ".join("---" for _ in self.datasets) + " |"

        lines = [header, separator]
        for op in self._all_operators:
            cells = []
            for ds in self.datasets:
                cells.append("x" if op in self._matrix[ds] else " ")
            lines.append(f"| {op} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def completion_pct(self) -> float:
        """Return overall completion percentage (0.0 to 100.0)."""
        total_cells = len(self._all_operators) * len(self.datasets)
        if total_cells == 0:
            return 0.0
        filled = sum(
            1
            for ds in self.datasets
            for op in self._all_operators
            if op in self._matrix[ds]
        )
        return (filled / total_cells) * 100.0
