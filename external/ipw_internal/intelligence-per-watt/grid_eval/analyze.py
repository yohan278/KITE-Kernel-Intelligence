"""Analysis tooling for grid_eval JSONL results.

Reads JSONL output from grid evaluation runs and produces
comparison tables with descriptive statistics at per-step
and per-trace aggregation levels.

Usage:
    from grid_eval.analyze import analyze_grid_results
    analyze_grid_results(Path("results/grid_eval_20260209/"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)

# Metrics extracted from each ActionEnergyBreakdown in action_breakdowns
STEP_METRIC_COLS = [
    "gpu_energy_j",
    "cpu_energy_j",
    "gpu_avg_power_w",
    "cpu_avg_power_w",
    "duration_ms",
    "cost_usd",
]

# Display names and units for step-level metrics
STEP_METRIC_DISPLAY: Dict[str, str] = {
    "gpu_energy_j": "GPU Energy (J)",
    "cpu_energy_j": "CPU Energy (J)",
    "gpu_avg_power_w": "GPU Avg Power (W)",
    "cpu_avg_power_w": "CPU Avg Power (W)",
    "duration_ms": "Latency (ms)",
    "cost_usd": "Cost ($)",
}

# Metrics computed at the trace (query) level
TRACE_METRIC_COLS = [
    "total_gpu_energy_j",
    "total_cpu_energy_j",
    "avg_gpu_power_w",
    "avg_cpu_power_w",
    "total_latency_s",
    "total_cost_usd",
    "is_correct",
    "num_steps",
]

# Display names for trace-level metrics
TRACE_METRIC_DISPLAY: Dict[str, str] = {
    "total_gpu_energy_j": "Total GPU Energy (J)",
    "total_cpu_energy_j": "Total CPU Energy (J)",
    "avg_gpu_power_w": "Avg GPU Power (W)",
    "avg_cpu_power_w": "Avg CPU Power (W)",
    "total_latency_s": "Total Latency (s)",
    "total_cost_usd": "Total Cost ($)",
    "is_correct": "Accuracy (%)",
    "num_steps": "Num Steps",
}

# Metrics that should show mean +/- std in cross-config comparison
COMPARISON_METRICS = [
    "is_correct",
    "total_gpu_energy_j",
    "total_cpu_energy_j",
    "avg_gpu_power_w",
    "avg_cpu_power_w",
    "total_latency_s",
    "total_cost_usd",
    "num_steps",
]

COMPARISON_DISPLAY: Dict[str, str] = {
    "is_correct": "Accuracy (%)",
    "total_gpu_energy_j": "GPU Energy (J)",
    "total_cpu_energy_j": "CPU Energy (J)",
    "avg_gpu_power_w": "GPU Power (W)",
    "avg_cpu_power_w": "CPU Power (W)",
    "total_latency_s": "Latency (s)",
    "total_cost_usd": "Cost ($)",
    "num_steps": "Steps",
}

# Groupings for visual separators in tables
STEP_METRIC_GROUPS = [
    ["gpu_energy_j", "cpu_energy_j"],
    ["gpu_avg_power_w", "cpu_avg_power_w"],
    ["duration_ms", "cost_usd"],
]

TRACE_METRIC_GROUPS = [
    ["total_gpu_energy_j", "total_cpu_energy_j"],
    ["avg_gpu_power_w", "avg_cpu_power_w"],
    ["total_latency_s", "total_cost_usd"],
    ["is_correct", "num_steps"],
]

COMPARISON_METRIC_GROUPS = [
    ["is_correct"],
    ["total_gpu_energy_j", "total_cpu_energy_j"],
    ["avg_gpu_power_w", "avg_cpu_power_w"],
    ["total_latency_s", "total_cost_usd", "num_steps"],
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSONL result files from a directory.

    Globs for ``results_*.jsonl``, parses each line.

    Args:
        results_dir: Directory containing grid_eval output files.

    Returns:
        List of parsed query result dicts.

    Raises:
        FileNotFoundError: If no JSONL files found in directory.
    """
    results_dir = Path(results_dir)
    jsonl_files = sorted(results_dir.glob("results_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No results_*.jsonl files found in {results_dir}"
        )

    records: List[Dict[str, Any]] = []
    for path in jsonl_files:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d", path.name, line_num
                    )
    logger.info("Loaded %d query results from %d files", len(records), len(jsonl_files))
    return records


# ---------------------------------------------------------------------------
# DataFrame Construction
# ---------------------------------------------------------------------------


def build_step_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build per-step DataFrame by flattening action_breakdowns.

    Each row is one agent action (LM call, tool call, etc.) from one query.
    Queries with missing or empty ``action_breakdowns`` are skipped.

    Returns:
        DataFrame with columns: model, agent, benchmark, query_id,
        step_number, action_type, and all STEP_METRIC_COLS.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        breakdowns = rec.get("action_breakdowns")
        if not breakdowns:
            continue
        for bd in breakdowns:
            rows.append({
                "model": rec.get("model", ""),
                "agent": rec.get("agent", ""),
                "benchmark": rec.get("benchmark", ""),
                "query_id": rec.get("query_id", ""),
                "step_number": bd.get("step_number", 0),
                "action_type": bd.get("action_type", ""),
                "gpu_energy_j": bd.get("gpu_energy_joules"),
                "cpu_energy_j": bd.get("cpu_energy_joules"),
                "gpu_avg_power_w": bd.get("gpu_avg_power_watts"),
                "cpu_avg_power_w": bd.get("cpu_avg_power_watts"),
                "duration_ms": bd.get("duration_ms"),
                "cost_usd": bd.get("cost_usd"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Convert None → NaN for numeric columns
    for col in STEP_METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_trace_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build per-trace DataFrame by aggregating steps per query.

    For energy/latency/cost: sums across steps.
    For power: duration-weighted average.
    Falls back to top-level ``avg_joules`` / ``latency_seconds`` when
    ``action_breakdowns`` is missing.

    Returns:
        DataFrame with columns: model, agent, benchmark, query_id,
        and all TRACE_METRIC_COLS.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        breakdowns = rec.get("action_breakdowns")

        if breakdowns:
            total_gpu_e = 0.0
            total_cpu_e = 0.0
            total_dur_ms = 0.0
            total_cost = 0.0
            weighted_gpu_power = 0.0
            weighted_cpu_power = 0.0
            total_power_dur = 0.0  # duration with valid power readings

            for bd in breakdowns:
                gpu_e = bd.get("gpu_energy_joules") or 0.0
                cpu_e = bd.get("cpu_energy_joules") or 0.0
                dur = bd.get("duration_ms") or 0.0
                cost = bd.get("cost_usd") or 0.0
                gpu_pw = bd.get("gpu_avg_power_watts")
                cpu_pw = bd.get("cpu_avg_power_watts")

                total_gpu_e += gpu_e
                total_cpu_e += cpu_e
                total_dur_ms += dur
                total_cost += cost

                if gpu_pw is not None and dur > 0:
                    weighted_gpu_power += gpu_pw * dur
                    weighted_cpu_power += (cpu_pw or 0.0) * dur
                    total_power_dur += dur

            avg_gpu_pw = (
                weighted_gpu_power / total_power_dur
                if total_power_dur > 0
                else np.nan
            )
            avg_cpu_pw = (
                weighted_cpu_power / total_power_dur
                if total_power_dur > 0
                else np.nan
            )

            rows.append({
                "model": rec.get("model", ""),
                "agent": rec.get("agent", ""),
                "benchmark": rec.get("benchmark", ""),
                "query_id": rec.get("query_id", ""),
                "total_gpu_energy_j": total_gpu_e,
                "total_cpu_energy_j": total_cpu_e,
                "avg_gpu_power_w": avg_gpu_pw,
                "avg_cpu_power_w": avg_cpu_pw,
                "total_latency_s": total_dur_ms / 1000.0,
                "total_cost_usd": total_cost,
                "is_correct": 1.0 if rec.get("is_correct") else 0.0,
                "num_steps": len(breakdowns),
            })
        else:
            # Fallback: use top-level fields
            rows.append({
                "model": rec.get("model", ""),
                "agent": rec.get("agent", ""),
                "benchmark": rec.get("benchmark", ""),
                "query_id": rec.get("query_id", ""),
                "total_gpu_energy_j": rec.get("avg_joules", np.nan),
                "total_cpu_energy_j": np.nan,
                "avg_gpu_power_w": rec.get("max_power_watts", np.nan),
                "avg_cpu_power_w": np.nan,
                "total_latency_s": rec.get("latency_seconds", np.nan),
                "total_cost_usd": np.nan,
                "is_correct": 1.0 if rec.get("is_correct") else 0.0,
                "num_steps": rec.get("turns", np.nan),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in TRACE_METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


STAT_FUNCS = {
    "min": "min",
    "max": "max",
    "median": "median",
    "mean": "mean",
    "std": "std",
    "count": "count",
}


def aggregate_stats(
    df: pd.DataFrame,
    group_by: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute descriptive statistics over metrics, grouped by config.

    Args:
        df: Input DataFrame (step-level or trace-level).
        group_by: Columns to group by (e.g. ["model", "agent"]).
        metric_cols: Numeric columns to compute stats over.

    Returns:
        DataFrame with MultiIndex columns: (metric, stat).
    """
    if df.empty:
        return df

    available = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(list(group_by))[available]
    result = grouped.agg(list(STAT_FUNCS.values()))
    return result


def build_comparison_table(
    trace_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build cross-configuration comparison (model x agent pivot).

    Computes mean and std per (model, agent) for each metric.

    Returns:
        DataFrame with rows=metrics, columns=(model, agent) pairs,
        values as "mean +/- std" strings.
    """
    if trace_df.empty:
        return trace_df

    available = [c for c in COMPARISON_METRICS if c in trace_df.columns]
    grouped = trace_df.groupby(["model", "agent"])[available]
    means = grouped.mean()
    stds = grouped.std()

    # Build formatted strings
    configs = means.index.tolist()  # list of (model, agent) tuples
    result_rows: List[Dict[str, Any]] = []

    for metric in available:
        row: Dict[str, Any] = {"metric": COMPARISON_DISPLAY.get(metric, metric)}
        for model, agent in configs:
            col_key = f"{model} / {agent}"
            m = means.loc[(model, agent), metric]
            s = stds.loc[(model, agent), metric]

            if metric == "is_correct":
                # Show as percentage
                row[col_key] = _fmt_pct(m)
            elif pd.isna(m):
                row[col_key] = "-"
            elif pd.isna(s) or s == 0:
                row[col_key] = _fmt_val(m, metric)
            else:
                row[col_key] = f"{_fmt_val(m, metric)}\u00b1{_fmt_val(s, metric)}"
        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Number Formatting
# ---------------------------------------------------------------------------


def _fmt_val(v: float, metric: str) -> str:
    """Format a numeric value based on its metric type."""
    if pd.isna(v):
        return "-"
    if "cost" in metric:
        return f"{v:.3f}"
    if "correct" in metric:
        return f"{v * 100:.1f}"
    if v >= 1000:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _fmt_pct(v: float) -> str:
    """Format a 0-1 fraction as a percentage string."""
    if pd.isna(v):
        return "-"
    return f"{v * 100:.1f}"


def _fmt_stat(v: float, metric: str) -> str:
    """Format a statistic value for table display."""
    if pd.isna(v):
        return "-"
    return _fmt_val(v, metric)


# ---------------------------------------------------------------------------
# Rendering — Rich Tables
# ---------------------------------------------------------------------------


def _get_group_boundaries(
    metrics: Sequence[str],
    groups: Sequence[Sequence[str]],
) -> List[int]:
    """Return indices after which to insert a separator row.

    Based on metric group definitions — insert a separator
    after the last metric in each group (except the final group).
    """
    boundaries: List[int] = []
    idx = 0
    for group in groups[:-1]:  # No separator after the last group
        for m in group:
            if m in metrics:
                idx += 1
        boundaries.append(idx)
    return boundaries


def render_step_table(
    stats_df: pd.DataFrame,
    model: str,
    agent: str,
    console: Console,
) -> None:
    """Render per-step statistics as a rich table for one (model, agent)."""
    title = f"{model} + {agent}  Per-Step"

    table = Table(title=title, box=box.HEAVY_HEAD, show_lines=False)
    table.add_column("Metric", style="bold", min_width=20)
    for stat_name in STAT_FUNCS:
        justify = "right"
        table.add_column(stat_name.capitalize(), justify=justify, min_width=8)

    available = [c for c in STEP_METRIC_COLS if c in [col[0] for col in stats_df.columns]]
    boundaries = _get_group_boundaries(available, STEP_METRIC_GROUPS)
    row_idx = 0

    for metric in available:
        display_name = STEP_METRIC_DISPLAY.get(metric, metric)
        row_vals = [display_name]
        for stat in STAT_FUNCS:
            try:
                val = stats_df.loc[(model, agent), (metric, stat)]
                if stat == "count":
                    row_vals.append(str(int(val)) if not pd.isna(val) else "-")
                else:
                    row_vals.append(_fmt_stat(val, metric))
            except KeyError:
                row_vals.append("-")
        table.add_row(*row_vals)
        row_idx += 1

        if row_idx in boundaries:
            table.add_row(*[""] * (1 + len(STAT_FUNCS)))

    console.print(table)
    console.print()


def render_trace_table(
    stats_df: pd.DataFrame,
    model: str,
    agent: str,
    console: Console,
) -> None:
    """Render per-trace statistics as a rich table for one (model, agent)."""
    title = f"{model} + {agent}  Per-Trace"

    table = Table(title=title, box=box.HEAVY_HEAD, show_lines=False)
    table.add_column("Metric", style="bold", min_width=22)
    for stat_name in STAT_FUNCS:
        justify = "right"
        table.add_column(stat_name.capitalize(), justify=justify, min_width=8)

    available = [c for c in TRACE_METRIC_COLS if c in [col[0] for col in stats_df.columns]]
    boundaries = _get_group_boundaries(available, TRACE_METRIC_GROUPS)
    row_idx = 0

    for metric in available:
        display_name = TRACE_METRIC_DISPLAY.get(metric, metric)
        row_vals = [display_name]
        for stat in STAT_FUNCS:
            try:
                val = stats_df.loc[(model, agent), (metric, stat)]
                if metric == "is_correct":
                    # Accuracy: show as %, only mean is meaningful
                    if stat == "mean":
                        row_vals.append(_fmt_pct(val))
                    elif stat == "count":
                        row_vals.append(str(int(val)) if not pd.isna(val) else "-")
                    else:
                        row_vals.append("-")
                elif stat == "count":
                    row_vals.append(str(int(val)) if not pd.isna(val) else "-")
                else:
                    row_vals.append(_fmt_stat(val, metric))
            except KeyError:
                row_vals.append("-")
        table.add_row(*row_vals)
        row_idx += 1

        if row_idx in boundaries:
            table.add_row(*[""] * (1 + len(STAT_FUNCS)))

    console.print(table)
    console.print()


def render_comparison_table(
    comparison_df: pd.DataFrame,
    console: Console,
) -> None:
    """Render the cross-config comparison as a rich table."""
    if comparison_df.empty:
        console.print("[dim]No data for comparison table.[/dim]")
        return

    title = "Cross-Config Comparison"
    table = Table(title=title, box=box.HEAVY_HEAD, show_lines=False)
    table.add_column("Metric", style="bold", min_width=18)

    # Add a column per (model, agent) config
    config_cols = [c for c in comparison_df.columns if c != "metric"]
    for col in config_cols:
        table.add_column(col, justify="right", min_width=14)

    # Determine group boundaries for separator rows
    metric_names = comparison_df["metric"].tolist()
    display_to_key = {v: k for k, v in COMPARISON_DISPLAY.items()}
    metric_keys = [display_to_key.get(m, m) for m in metric_names]
    boundaries = _get_group_boundaries(metric_keys, COMPARISON_METRIC_GROUPS)

    for row_idx, (_, row) in enumerate(comparison_df.iterrows()):
        vals = [row["metric"]] + [str(row[c]) for c in config_cols]
        table.add_row(*vals)

        if (row_idx + 1) in boundaries:
            table.add_row(*[""] * (1 + len(config_cols)))

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Rendering — CSV / Markdown
# ---------------------------------------------------------------------------


def render_csv(
    df: pd.DataFrame,
    output_path: Path,
    label: str = "",
) -> None:
    """Export a DataFrame to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info("Wrote %s CSV to %s", label, output_path)


def render_markdown(df: pd.DataFrame, title: str) -> str:
    """Render a DataFrame as a markdown table string."""
    header = f"## {title}\n\n"
    try:
        return header + df.to_markdown(index=True) + "\n"
    except ImportError:
        # tabulate not installed — fall back to CSV-like format
        return header + df.to_string() + "\n"


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def analyze_grid_results(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    group_by_benchmark: bool = False,
    fmt: str = "terminal",
    models: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
) -> None:
    """Analyze grid evaluation results and produce comparison tables.

    Args:
        results_dir: Directory containing grid_eval JSONL output.
        output_dir: Directory for CSV/markdown output (default: results_dir/analysis).
        group_by_benchmark: If True, sub-group statistics by benchmark.
        fmt: Output format — "terminal", "csv", "markdown", or "all".
        models: Optional filter for model names.
        agents: Optional filter for agent names.
        benchmarks: Optional filter for benchmark names.
    """
    console = Console()

    # 1. Load
    records = load_results(results_dir)
    if not records:
        console.print("[red]No results found.[/red]")
        return

    # 2. Apply filters
    if models:
        records = [r for r in records if r.get("model") in models]
    if agents:
        records = [r for r in records if r.get("agent") in agents]
    if benchmarks:
        records = [r for r in records if r.get("benchmark") in benchmarks]

    if not records:
        console.print("[red]No results after filtering.[/red]")
        return

    console.print(f"[bold]Analyzing {len(records)} query results[/bold]")
    console.print()

    # 3. Build DataFrames
    step_df = build_step_dataframe(records)
    trace_df = build_trace_dataframe(records)

    # 4. Define grouping
    group_cols = ["model", "agent"]
    if group_by_benchmark:
        group_cols.append("benchmark")

    # 5. Aggregate
    step_stats = aggregate_stats(step_df, group_cols, STEP_METRIC_COLS) if not step_df.empty else pd.DataFrame()
    trace_stats = aggregate_stats(trace_df, group_cols, TRACE_METRIC_COLS) if not trace_df.empty else pd.DataFrame()
    comparison = build_comparison_table(trace_df) if not trace_df.empty else pd.DataFrame()

    # 6. Output
    if output_dir is None:
        output_dir = Path(results_dir) / "analysis"

    show_terminal = fmt in ("terminal", "all")
    write_csv = fmt in ("csv", "all")
    write_md = fmt in ("markdown", "all")

    # Get unique (model, agent) pairs
    configs = []
    if not trace_df.empty:
        configs = trace_df.groupby(group_cols).size().index.tolist()

    # Per-step tables
    if not step_stats.empty and show_terminal:
        console.rule("[bold]Per-Step Statistics[/bold]")
        console.print()
        for config_key in configs:
            if isinstance(config_key, tuple):
                model, agent = config_key[0], config_key[1]
            else:
                model, agent = config_key, ""
            try:
                render_step_table(step_stats, model, agent, console)
            except KeyError:
                pass

    # Per-trace tables
    if not trace_stats.empty and show_terminal:
        console.rule("[bold]Per-Trace Statistics[/bold]")
        console.print()
        for config_key in configs:
            if isinstance(config_key, tuple):
                model, agent = config_key[0], config_key[1]
            else:
                model, agent = config_key, ""
            try:
                render_trace_table(trace_stats, model, agent, console)
            except KeyError:
                pass

    # Comparison table
    if not comparison.empty and show_terminal:
        console.rule("[bold]Cross-Config Comparison[/bold]")
        console.print()
        render_comparison_table(comparison, console)

    # CSV output
    if write_csv:
        if not step_stats.empty:
            render_csv(step_stats, output_dir / "step_statistics.csv", "per-step")
        if not trace_stats.empty:
            render_csv(trace_stats, output_dir / "trace_statistics.csv", "per-trace")
        if not comparison.empty:
            render_csv(comparison, output_dir / "comparison.csv", "comparison")

    # Markdown output
    if write_md:
        md_parts: List[str] = []
        if not step_stats.empty:
            md_parts.append(render_markdown(step_stats, "Per-Step Statistics"))
        if not trace_stats.empty:
            md_parts.append(render_markdown(trace_stats, "Per-Trace Statistics"))
        if not comparison.empty:
            md_parts.append(render_markdown(comparison, "Cross-Config Comparison"))

        if md_parts:
            md_path = output_dir / "analysis.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text("\n".join(md_parts))
            logger.info("Wrote markdown to %s", md_path)
            if fmt == "markdown":
                console.print("\n".join(md_parts))

    console.print("[green]Analysis complete.[/green]")


__all__ = [
    "analyze_grid_results",
    "build_comparison_table",
    "build_step_dataframe",
    "build_trace_dataframe",
    "load_results",
]
