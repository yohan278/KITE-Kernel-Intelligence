"""Run profiling against an inference client."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import click
from ipw.core.types import ProfilerConfig

from ipw.cli._console import success


def _collect_params(ctx, param, values):
    collected: Dict[str, str] = {}
    for item in values:
        for piece in item.split(","):
            if not piece:
                continue
            key, _, raw = piece.partition("=")
            key = key.strip()
            if not key:
                continue
            collected[key] = raw.strip()
    return collected


@click.command(help="Run profiling against an inference client.")
@click.option("--client", "client_id", required=True, help="Client identifier")
@click.option("--model", required=True, help="Model name to invoke")
@click.option("--dataset", "dataset_id", default="ipw", help="Dataset identifier")
@click.option("--client-base-url", help="Client base URL")
@click.option(
    "--phased",
    is_flag=True,
    default=False,
    help="Enable phase-aware energy attribution (prefill vs decode).",
)
@click.option(
    "--dataset-param",
    multiple=True,
    callback=_collect_params,
    help="Dataset params key=value",
)
@click.option(
    "--client-param",
    multiple=True,
    callback=_collect_params,
    help="Client params key=value",
)
@click.option("--output-dir", type=click.Path())
@click.option("--max-queries", type=int)
@click.option(
    "--max-concurrency",
    "max_concurrency",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Maximum number of concurrent inference requests",
)
def profile(
    dataset_id: str,
    client_id: str,
    client_base_url: str | None,
    model: str,
    dataset_param,
    client_param,
    output_dir: str | None,
    max_queries: int | None,
    max_concurrency: int,
    phased: bool,
) -> None:
    """Execute profiling run with the execution pipeline."""
    import ipw.clients
    import ipw.data_loaders

    ipw.clients.ensure_registered()
    missing_reason = getattr(ipw.clients, "MISSING_CLIENTS", {}).get(client_id)
    if missing_reason:
        raise click.ClickException(
            f"Inference client '{client_id}' is unavailable: {missing_reason}"
        )

    ipw.data_loaders.ensure_registered()
    from ipw.execution import ProfilerRunner  # Deferred import for heavy dependencies

    config = ProfilerConfig(
        dataset_id=dataset_id,
        client_id=client_id,
        client_base_url=client_base_url,
        dataset_params=dataset_param,
        client_params=client_param,
        model=model,
        max_queries=max_queries,
        max_concurrency=max_concurrency,
        phased_profiling=phased,
        output_dir=Path(output_dir) if output_dir else None,
    )

    runner = ProfilerRunner(config)
    runner.run()
    success("Profiling run completed")


__all__ = ["profile"]
