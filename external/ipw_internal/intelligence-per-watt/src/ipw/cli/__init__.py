"""Command-line interface for the Intelligence Per Watt platform (Click-based)."""

from __future__ import annotations

import click

from .analyze import analyze
from .bench import bench
from .diagnostic import benchmark
from .list import list_cmd
from .plot import plot
from .profile import profile
from .simulate import simulate


@click.group(help="Intelligence Per Watt development CLI tool")
def cli() -> None:
    """Top-level CLI group."""


cli.add_command(profile, "profile")
cli.add_command(analyze, "analyze")
cli.add_command(plot, "plot")
cli.add_command(list_cmd, "list")
cli.add_command(bench, "bench")
cli.add_command(benchmark, "benchmark")
cli.add_command(simulate, "simulate")


# Lazy import to avoid circular dependency (grid_eval -> ipw.cli -> grid_eval)
try:
    from grid_eval.analyze_cli import analyze as grid_analyze
    cli.add_command(grid_analyze, "grid-analyze")
except ImportError:
    pass


def main() -> None:
    """CLI entry point for console scripts."""
    cli()


__all__ = ["cli", "main"]
