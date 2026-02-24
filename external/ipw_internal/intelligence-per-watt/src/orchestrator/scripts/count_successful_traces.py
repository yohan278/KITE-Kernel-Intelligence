#!/usr/bin/env python3
"""Count successful traces in a trajectory dataset.

Example usage:
    python scripts/count_successful_traces.py --path data/trajectories/checkpoint
    python scripts/count_successful_traces.py --path data/trajectories/my_dataset --detailed
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.data.trajectory_dataset import TrajectoryDataset


def main():
    parser = argparse.ArgumentParser(
        description="Count successful traces in a trajectory dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic count
  python scripts/count_successful_traces.py --path data/trajectories/checkpoint
  
  # Detailed statistics
  python scripts/count_successful_traces.py --path data/trajectories/my_dataset --detailed
  
  # Filter by source dataset
  python scripts/count_successful_traces.py --path data/trajectories/checkpoint --source generalthought
        """,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to trajectory dataset directory (supports JSONL, Arrow, Parquet formats)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics including breakdowns by source, tools, etc.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by source dataset name (e.g., 'generalthought', 'adp')",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only count successful traces",
    )
    parser.add_argument(
        "--failure-only",
        action="store_true",
        help="Only count failed traces",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return 1

    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = TrajectoryDataset.load(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Apply filters if specified
    if args.source or args.success_only or args.failure_only:
        filtered_dataset = dataset.filter(
            success_only=args.success_only,
            failure_only=args.failure_only,
            source_dataset=args.source,
        )
        dataset = filtered_dataset

    # Get statistics
    stats = dataset.get_statistics()

    # Print results
    print("\n" + "=" * 70)
    print("TRACE STATISTICS")
    print("=" * 70)
    print(f"Total traces:        {stats['total_samples']:,}")
    print(f"Successful traces:    {stats['success_count']:,}")
    print(f"Failed traces:        {stats['failure_count']:,}")
    print(f"Success rate:         {stats['success_rate']:.2%}")
    print()

    if args.detailed:
        print("=" * 70)
        print("DETAILED STATISTICS")
        print("=" * 70)
        print(f"Average turns:        {stats['avg_turns']:.2f}")
        print(f"Average energy:       {stats['avg_energy_joules']:.4f} J")
        print(f"Average latency:      {stats['avg_latency_seconds']:.2f} s")
        print(f"Average cost:         ${stats['avg_cost_usd']:.6f}")
        print()
        print(f"Total energy:         {stats['total_energy_joules']:.2f} J")
        print(f"Total latency:        {stats['total_latency_seconds']:.2f} s")
        print(f"Total cost:           ${stats['total_cost_usd']:.4f}")
        print()

        # Source breakdown
        if stats.get('sources'):
            print("=" * 70)
            print("BREAKDOWN BY SOURCE DATASET")
            print("=" * 70)
            total = stats['total_samples']
            for source, count in sorted(stats['sources'].items(), key=lambda x: -x[1]):
                pct = count / total * 100
                print(f"  {source:30s} {count:8,} ({pct:5.1f}%)")
            print()

        # Tools breakdown
        if stats.get('tools'):
            print("=" * 70)
            print("BREAKDOWN BY TOOLS USED")
            print("=" * 70)
            for tool, count in sorted(stats['tools'].items(), key=lambda x: -x[1]):
                print(f"  {tool:30s} {count:8,} uses")
            print()

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
