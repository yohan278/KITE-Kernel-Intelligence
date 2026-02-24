"""CLI for grid_eval result analysis.

Usage:
    python -m grid_eval.analyze_cli results/grid_eval_20260209/ --format all
    python -m grid_eval.analyze_cli results/ --models gpt-oss-120b --by-benchmark
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click


@click.command()
@click.argument(
    "results_dir",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for CSV/markdown output. Default: <results_dir>/analysis",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["terminal", "csv", "markdown", "all"]),
    default="terminal",
    help="Output format. Default: terminal",
)
@click.option(
    "--by-benchmark",
    is_flag=True,
    help="Sub-group statistics by benchmark",
)
@click.option(
    "--models",
    type=str,
    default=None,
    help="Comma-separated model filter (e.g. gpt-oss-120b,glm-4.7-flash)",
)
@click.option(
    "--agents",
    type=str,
    default=None,
    help="Comma-separated agent filter (e.g. react,openhands)",
)
@click.option(
    "--benchmarks",
    type=str,
    default=None,
    help="Comma-separated benchmark filter (e.g. gaia,hle)",
)
def analyze(
    results_dir: Path,
    output_dir: Optional[Path],
    fmt: str,
    by_benchmark: bool,
    models: Optional[str],
    agents: Optional[str],
    benchmarks: Optional[str],
) -> None:
    """Analyze grid evaluation results and produce comparison tables.

    RESULTS_DIR is the directory containing grid_eval JSONL output files.

    Examples:

        # Terminal tables for all data
        python -m grid_eval.analyze_cli results/grid_eval_20260209/

        # All formats with filters
        python -m grid_eval.analyze_cli results/ -f all --models gpt-oss-120b

        # Sub-grouped by benchmark
        python -m grid_eval.analyze_cli results/ --by-benchmark
    """
    from grid_eval.analyze import analyze_grid_results

    analyze_grid_results(
        results_dir=results_dir,
        output_dir=output_dir,
        group_by_benchmark=by_benchmark,
        fmt=fmt,
        models=models.split(",") if models else None,
        agents=agents.split(",") if agents else None,
        benchmarks=benchmarks.split(",") if benchmarks else None,
    )


if __name__ == "__main__":
    analyze()
