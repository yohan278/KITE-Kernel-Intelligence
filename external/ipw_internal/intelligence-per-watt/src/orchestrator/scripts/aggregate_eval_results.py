#!/usr/bin/env python3
"""Aggregate evaluation results for 3-way comparison.

Generates comparison table across GAIA/HLE/FRAMES benchmarks for:
- Config A: Qwen3-8B (base) without tools
- Config B: Qwen3-8B (base) with full orchestrator toolkit
- Config C: Qwen3-8B (SFT'd) with full orchestrator toolkit

Usage:
    python src/orchestrator/scripts/aggregate_eval_results.py \\
        --results-dir evals/results \\
        --output evals/results/comparison_table.md

    # With specific result directories:
    python src/orchestrator/scripts/aggregate_eval_results.py \\
        --baseline-dir evals/results/baseline_notool \\
        --orchestrator-dir evals/results/orchestrator_notrain \\
        --sft-dir evals/results/orchestrator_sft \\
        --output evals/results/comparison_table.md
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load evaluation results from a directory.

    Looks for JSON files with benchmark results in the format:
    - gaia_results.json
    - hle_results.json
    - frames_results.json

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dictionary mapping benchmark name to results
    """
    results = {}

    for benchmark in ["gaia", "hle", "frames"]:
        # Try multiple possible file patterns
        patterns = [
            f"{benchmark}_results.json",
            f"{benchmark}.json",
            f"results_{benchmark}.json",
        ]

        for pattern in patterns:
            result_file = results_dir / pattern
            if result_file.exists():
                with open(result_file) as f:
                    results[benchmark] = json.load(f)
                break

    return results


def extract_accuracy(results: Dict[str, Any], benchmark: str) -> Optional[float]:
    """Extract accuracy from benchmark results.

    Args:
        results: Benchmark results dictionary
        benchmark: Benchmark name

    Returns:
        Accuracy as float (0-1) or None if not found
    """
    if not results:
        return None

    # Common keys to check
    accuracy_keys = ["accuracy", "overall_accuracy", "score", "correct_rate"]

    for key in accuracy_keys:
        if key in results:
            value = results[key]
            # Handle percentage vs decimal
            if isinstance(value, (int, float)):
                return value if value <= 1 else value / 100
            if isinstance(value, str):
                try:
                    value = float(value.rstrip("%"))
                    return value if value <= 1 else value / 100
                except ValueError:
                    pass

    # Try nested metrics
    if "metrics" in results:
        return extract_accuracy(results["metrics"], benchmark)

    return None


def extract_detailed_metrics(results: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    """Extract detailed metrics from benchmark results.

    Args:
        results: Benchmark results dictionary
        benchmark: Benchmark name

    Returns:
        Dictionary of detailed metrics
    """
    metrics = {}

    if not results:
        return metrics

    if benchmark == "gaia":
        # GAIA has level-based accuracy
        for level in ["level1", "level2", "level3"]:
            key = f"{level}_accuracy"
            if key in results:
                metrics[level] = results[key]

    elif benchmark == "hle":
        # HLE has multiple choice vs short answer
        if "mc_accuracy" in results:
            metrics["multiple_choice"] = results["mc_accuracy"]
        if "short_answer_accuracy" in results:
            metrics["short_answer"] = results["short_answer_accuracy"]
        if "subject_metrics" in results:
            metrics["by_subject"] = results["subject_metrics"]

    elif benchmark == "frames":
        # FRAMES may have category breakdown
        if "category_metrics" in results:
            metrics["by_category"] = results["category_metrics"]

    # Common metrics
    for key in ["avg_latency", "total_tasks", "correct_tasks", "total_cost"]:
        if key in results:
            metrics[key] = results[key]

    return metrics


def format_accuracy(accuracy: Optional[float]) -> str:
    """Format accuracy for display.

    Args:
        accuracy: Accuracy value (0-1) or None

    Returns:
        Formatted string (e.g., "45.2%" or "-")
    """
    if accuracy is None:
        return "-"
    return f"{accuracy * 100:.1f}%"


def generate_markdown_table(
    configs: Dict[str, Dict[str, Dict[str, Any]]],
    benchmarks: List[str] = None,
) -> str:
    """Generate markdown comparison table.

    Args:
        configs: Dictionary mapping config name to benchmark results
        benchmarks: List of benchmarks to include (default: all)

    Returns:
        Markdown formatted table
    """
    if benchmarks is None:
        benchmarks = ["gaia", "hle", "frames"]

    lines = []

    # Header
    lines.append("# Orchestrator SFT Evaluation Results")
    lines.append("")
    lines.append("## 3-Way Comparison")
    lines.append("")
    lines.append("| Configuration | Model | Tools | GAIA | HLE | FRAMES |")
    lines.append("|---------------|-------|-------|------|-----|--------|")

    # Config descriptions
    config_info = {
        "baseline": ("Qwen3-8B (base)", "None"),
        "orchestrator": ("Qwen3-8B (base)", "Full toolkit"),
        "sft": ("Qwen3-8B (SFT'd)", "Full toolkit"),
    }

    # Data rows
    for config_name in ["baseline", "orchestrator", "sft"]:
        model, tools = config_info.get(config_name, (config_name, "-"))
        config_results = configs.get(config_name, {})

        accuracies = []
        for benchmark in benchmarks:
            bench_results = config_results.get(benchmark, {})
            acc = extract_accuracy(bench_results, benchmark)
            accuracies.append(format_accuracy(acc))

        row = f"| {config_name.upper()} | {model} | {tools} | {' | '.join(accuracies)} |"
        lines.append(row)

    lines.append("")

    # Add detailed breakdown for each benchmark
    for benchmark in benchmarks:
        lines.append(f"## {benchmark.upper()} Detailed Results")
        lines.append("")

        has_data = False
        for config_name in ["baseline", "orchestrator", "sft"]:
            config_results = configs.get(config_name, {})
            bench_results = config_results.get(benchmark, {})

            if bench_results:
                has_data = True
                metrics = extract_detailed_metrics(bench_results, benchmark)

                lines.append(f"### {config_name.upper()}")
                lines.append("")

                # Overall accuracy
                acc = extract_accuracy(bench_results, benchmark)
                lines.append(f"- **Overall Accuracy**: {format_accuracy(acc)}")

                # Detailed metrics
                if benchmark == "gaia" and any(k in metrics for k in ["level1", "level2", "level3"]):
                    lines.append(f"- **Level 1**: {format_accuracy(metrics.get('level1'))}")
                    lines.append(f"- **Level 2**: {format_accuracy(metrics.get('level2'))}")
                    lines.append(f"- **Level 3**: {format_accuracy(metrics.get('level3'))}")

                if benchmark == "hle":
                    if "multiple_choice" in metrics:
                        lines.append(f"- **Multiple Choice**: {format_accuracy(metrics.get('multiple_choice'))}")
                    if "short_answer" in metrics:
                        lines.append(f"- **Short Answer**: {format_accuracy(metrics.get('short_answer'))}")

                if "avg_latency" in metrics:
                    lines.append(f"- **Avg Latency**: {metrics['avg_latency']:.2f}s")

                if "total_tasks" in metrics:
                    correct = metrics.get("correct_tasks", 0)
                    total = metrics["total_tasks"]
                    lines.append(f"- **Tasks**: {correct}/{total}")

                lines.append("")

        if not has_data:
            lines.append("*No results available*")
            lines.append("")

    # Add summary section
    lines.append("## Summary")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    # Calculate improvements
    improvements = []
    for benchmark in benchmarks:
        baseline_acc = extract_accuracy(configs.get("baseline", {}).get(benchmark, {}), benchmark)
        sft_acc = extract_accuracy(configs.get("sft", {}).get(benchmark, {}), benchmark)

        if baseline_acc is not None and sft_acc is not None and baseline_acc > 0:
            improvement = (sft_acc - baseline_acc) / baseline_acc * 100
            improvements.append(f"- **{benchmark.upper()}**: {improvement:+.1f}% improvement from baseline to SFT")

    if improvements:
        for imp in improvements:
            lines.append(imp)
    else:
        lines.append("*Improvement analysis requires complete results*")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by aggregate_eval_results.py*")

    return "\n".join(lines)


def generate_json_summary(configs: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate JSON summary of results.

    Args:
        configs: Dictionary mapping config name to benchmark results

    Returns:
        Summary dictionary
    """
    summary = {
        "configs": {},
        "improvements": {},
    }

    benchmarks = ["gaia", "hle", "frames"]

    for config_name, config_results in configs.items():
        summary["configs"][config_name] = {}
        for benchmark in benchmarks:
            bench_results = config_results.get(benchmark, {})
            acc = extract_accuracy(bench_results, benchmark)
            metrics = extract_detailed_metrics(bench_results, benchmark)

            summary["configs"][config_name][benchmark] = {
                "accuracy": acc,
                "details": metrics,
            }

    # Calculate improvements
    for benchmark in benchmarks:
        baseline_acc = summary["configs"].get("baseline", {}).get(benchmark, {}).get("accuracy")
        orch_acc = summary["configs"].get("orchestrator", {}).get(benchmark, {}).get("accuracy")
        sft_acc = summary["configs"].get("sft", {}).get(benchmark, {}).get("accuracy")

        summary["improvements"][benchmark] = {
            "orchestrator_vs_baseline": None,
            "sft_vs_baseline": None,
            "sft_vs_orchestrator": None,
        }

        if baseline_acc is not None and baseline_acc > 0:
            if orch_acc is not None:
                summary["improvements"][benchmark]["orchestrator_vs_baseline"] = (orch_acc - baseline_acc) / baseline_acc
            if sft_acc is not None:
                summary["improvements"][benchmark]["sft_vs_baseline"] = (sft_acc - baseline_acc) / baseline_acc

        if orch_acc is not None and orch_acc > 0 and sft_acc is not None:
            summary["improvements"][benchmark]["sft_vs_orchestrator"] = (sft_acc - orch_acc) / orch_acc

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results for 3-way comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Result directories
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evals/results",
        help="Base directory containing result subdirectories",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Directory with baseline (no tools) results",
    )
    parser.add_argument(
        "--orchestrator-dir",
        type=str,
        default=None,
        help="Directory with orchestrator (no SFT) results",
    )
    parser.add_argument(
        "--sft-dir",
        type=str,
        default=None,
        help="Directory with SFT + orchestrator results",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="evals/results/comparison_table.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional JSON output file path",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Determine result directories
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else results_dir / "baseline_notool"
    orchestrator_dir = Path(args.orchestrator_dir) if args.orchestrator_dir else results_dir / "orchestrator_notrain"
    sft_dir = Path(args.sft_dir) if args.sft_dir else results_dir / "orchestrator_sft"

    print("=" * 60)
    print("Aggregating Evaluation Results")
    print("=" * 60)
    print(f"Baseline dir: {baseline_dir}")
    print(f"Orchestrator dir: {orchestrator_dir}")
    print(f"SFT dir: {sft_dir}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load results
    configs = {}

    print("\nLoading results...")
    for name, dir_path in [
        ("baseline", baseline_dir),
        ("orchestrator", orchestrator_dir),
        ("sft", sft_dir),
    ]:
        if dir_path.exists():
            results = load_results(dir_path)
            configs[name] = results
            print(f"  {name}: {len(results)} benchmarks loaded")
        else:
            print(f"  {name}: Directory not found ({dir_path})")
            configs[name] = {}

    # Generate markdown table
    print("\nGenerating comparison table...")
    markdown = generate_markdown_table(configs)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(markdown)
    print(f"Markdown saved to: {output_path}")

    # Optional JSON output
    if args.json_output:
        json_summary = generate_json_summary(configs)
        json_path = Path(args.json_output)
        with open(json_path, "w") as f:
            json.dump(json_summary, f, indent=2)
        print(f"JSON saved to: {json_path}")

    # Print preview
    print("\n" + "=" * 60)
    print("Preview:")
    print("=" * 60)
    print(markdown[:1500])
    if len(markdown) > 1500:
        print("...")
        print(f"(Total: {len(markdown)} characters)")

    print("\nDone!")


if __name__ == "__main__":
    main()
