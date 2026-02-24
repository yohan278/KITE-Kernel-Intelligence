#!/usr/bin/env python3
"""Filter trajectory datasets for successful traces with tool usage.

Filters datasets based on:
- success == True
- tools_used contains at least one tool  
- num_turns > 0
- 0 < total_tokens < 256k

Supports:
- Local files: JSON, JSONL, Parquet
- HuggingFace datasets (cloud or local)

Usage:
    # Filter a HuggingFace dataset
    python scripts/filter_dataset.py \
        --input ScalingIntelligence/gpt-oss-20b-sft-traces-new-system-prompt-no-icl \
        --output ./data/filtered_traces \
        --format parquet
    
    # Filter a local JSONL file
    python scripts/filter_dataset.py \
        --input ./data/raw_traces.jsonl \
        --output ./data/filtered_traces \
        --format jsonl
    
    # Preview without saving
    python scripts/filter_dataset.py \
        --input ./data/raw_traces.parquet \
        --preview
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dataset_from_source(source: str, split: str = "train", streaming: bool = False):
    """Load dataset from various sources.
    
    Args:
        source: HuggingFace dataset name, or path to local file/directory
        split: Dataset split to load
        streaming: Whether to use streaming mode for large datasets
    
    Returns:
        HuggingFace Dataset object
    """
    from datasets import Dataset, load_dataset, load_from_disk
    
    source_path = Path(source)
    
    # Check if it's a local file
    if source_path.exists():
        if source_path.is_file():
            suffix = source_path.suffix.lower()
            if suffix == ".jsonl":
                return load_dataset("json", data_files=str(source_path), split="train")
            elif suffix == ".json":
                return load_dataset("json", data_files=str(source_path), split="train")
            elif suffix == ".parquet":
                return Dataset.from_parquet(str(source_path))
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        elif source_path.is_dir():
            # Try loading as HuggingFace dataset directory
            return load_from_disk(str(source_path))
    
    # Assume it's a HuggingFace dataset name
    print(f"Loading from HuggingFace Hub: {source}")
    return load_dataset(source, split=split, streaming=streaming)


def filter_row(row: Dict[str, Any],
               min_tokens: int = 0,
               max_tokens: int = 256_000,
               min_turns: int = 1,
               min_tools: int = 1,
               success_only: bool = True) -> bool:
    """Check if a row passes the filter criteria.

    Args:
        row: Dataset row as dictionary
        min_tokens: Minimum total tokens (exclusive)
        max_tokens: Maximum total tokens (exclusive)
        min_turns: Minimum num_turns (exclusive, i.e. num_turns > min_turns when > 0, or no filter when 0)
        min_tools: Minimum tools used (exclusive, i.e. len(tools_used) > min_tools when > 0, or no filter when 0)
        success_only: Only keep successful traces

    Returns:
        True if row passes all filters
    """
    # Check success
    if success_only:
        success = row.get("success", False)
        if not success:
            return False

    # Check tools_used - can be list of strings, empty list, or None
    if min_tools > 0:
        tools_used = row.get("tools_used", [])
        if tools_used is None:
            tools_used = []
        if isinstance(tools_used, str):
            # Handle case where tools_used might be a JSON string
            try:
                tools_used = json.loads(tools_used) if tools_used else []
            except (json.JSONDecodeError, TypeError):
                tools_used = [tools_used] if tools_used else []
        if len(tools_used) < min_tools:
            return False

    # Check num_turns
    if min_turns > 0:
        num_turns = row.get("num_turns", 0)
        if num_turns is None:
            num_turns = 0
        if num_turns < min_turns:
            return False

    # Check total_tokens - must be between min_tokens and max_tokens
    total_tokens = row.get("total_tokens", 0)
    if total_tokens is None:
        total_tokens = 0
    if not (min_tokens < total_tokens < max_tokens):
        return False

    return True


def save_dataset(dataset, output_path: str, format: str = "parquet"):
    """Save dataset to specified format.
    
    Args:
        dataset: HuggingFace Dataset
        output_path: Output directory or file path
        format: Output format - "parquet", "jsonl", "arrow"
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        output_file = output_path / "filtered_traces.parquet"
        dataset.to_parquet(str(output_file))
        print(f"Saved to: {output_file}")
        
    elif format == "jsonl":
        output_file = output_path / "filtered_traces.jsonl"
        with open(output_file, "w") as f:
            for row in dataset:
                f.write(json.dumps(row) + "\n")
        print(f"Saved to: {output_file}")
        
    elif format == "arrow":
        output_dir = output_path / "dataset"
        dataset.save_to_disk(str(output_dir))
        print(f"Saved to: {output_dir}")
        
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Save metadata
    metadata = {
        "num_samples": len(dataset),
        "format": format,
        "filter_criteria": {
            "success_only": True,
            "min_tools": 1,
            "min_turns": 1,
            "min_tokens": 0,
            "max_tokens": 256000,
        }
    }
    with open(output_path / "filter_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def print_statistics(dataset, prefix: str = ""):
    """Print dataset statistics.
    
    Args:
        dataset: HuggingFace Dataset
        prefix: Prefix for the statistics label
    """
    # Handle both regular and streaming datasets
    try:
        total = len(dataset)
    except TypeError:
        # Streaming dataset - convert to list first (sample)
        print(f"{prefix}Dataset is streaming, showing sample statistics...")
        return
    
    if total == 0:
        print(f"{prefix}Dataset is empty")
        return
    
    success_count = 0
    total_tokens_sum = 0
    total_turns_sum = 0
    tools_count = 0
    
    for row in dataset:
        if row.get("success", False):
            success_count += 1
        total_tokens_sum += row.get("total_tokens", 0) or 0
        total_turns_sum += row.get("num_turns", 0) or 0
        tools_used = row.get("tools_used", [])
        if tools_used:
            tools_count += 1
    
    print(f"\n{prefix}Dataset Statistics:")
    print(f"  Total samples:     {total:,}")
    print(f"  Successful:        {success_count:,} ({100*success_count/total:.1f}%)")
    print(f"  With tools used:   {tools_count:,} ({100*tools_count/total:.1f}%)")
    print(f"  Avg tokens:        {total_tokens_sum/total:,.0f}")
    print(f"  Avg turns:         {total_turns_sum/total:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter trajectory datasets for successful traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter a HuggingFace dataset
    python scripts/filter_dataset.py \\
        --input ScalingIntelligence/gpt-oss-20b-sft-traces \\
        --output ./data/filtered_traces
    
    # Filter a local JSONL file
    python scripts/filter_dataset.py \\
        --input ./data/raw_traces.jsonl \\
        --output ./data/filtered \\
        --format jsonl
    
    # Preview filtering results without saving
    python scripts/filter_dataset.py \\
        --input ./data/raw.parquet \\
        --preview
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input source: HuggingFace dataset name or local file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory path (required unless --preview)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["parquet", "jsonl", "arrow"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Minimum total_tokens (exclusive, default: 0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256_000,
        help="Maximum total_tokens (exclusive, default: 256000)"
    )
    parser.add_argument(
        "--no-success-filter",
        action="store_true",
        help="Don't filter by success=True"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Minimum num_turns (keeps rows with num_turns >= N, 0 to disable, default: 1)"
    )
    parser.add_argument(
        "--min-tools",
        type=int,
        default=1,
        help="Minimum tools used (keeps rows with len(tools_used) >= N, 0 to disable, default: 1)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview filtering results without saving"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large HuggingFace datasets"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    if not args.preview and args.output is None:
        parser.error("--output is required unless --preview is specified")
    
    # Print configuration
    print("=" * 70)
    print("Dataset Filter")
    print("=" * 70)
    print(f"Input:  {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print()
    print("Filter Criteria:")
    print(f"  success == True:        {not args.no_success_filter}")
    print(f"  len(tools_used) >= {args.min_tools}:   {args.min_tools > 0} (min_tools={args.min_tools})")
    print(f"  num_turns >= {args.min_turns}:          {args.min_turns > 0} (min_turns={args.min_turns})")
    print(f"  {args.min_tokens} < total_tokens < {args.max_tokens}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = load_dataset_from_source(
            args.input, 
            split=args.split,
            streaming=args.streaming
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Print original statistics
    if not args.streaming:
        print_statistics(dataset, "Original ")
    
    # Apply limit if specified
    if args.limit and not args.streaming:
        print(f"\nLimiting to {args.limit} samples...")
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    # Filter dataset
    print("\nFiltering dataset...")
    
    def filter_fn(row):
        return filter_row(
            row,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            min_turns=args.min_turns,
            min_tools=args.min_tools,
            success_only=not args.no_success_filter,
        )
    
    if args.streaming:
        # For streaming, we need to iterate and filter
        filtered_rows = []
        for i, row in enumerate(dataset):
            if args.limit and i >= args.limit:
                break
            if filter_fn(row):
                filtered_rows.append(row)
            if args.verbose and i % 1000 == 0:
                print(f"  Processed {i:,} rows, kept {len(filtered_rows):,}")
        
        from datasets import Dataset
        filtered_dataset = Dataset.from_list(filtered_rows)
    else:
        filtered_dataset = dataset.filter(filter_fn)
    
    # Print filtered statistics
    print_statistics(filtered_dataset, "Filtered ")
    
    # Calculate reduction
    try:
        orig_size = len(dataset) if not args.streaming else "N/A"
        new_size = len(filtered_dataset)
        if isinstance(orig_size, int):
            reduction = 100 * (1 - new_size / orig_size)
            print(f"\nReduction: {orig_size:,} -> {new_size:,} ({reduction:.1f}% removed)")
    except:
        pass
    
    # Save if not preview
    if not args.preview:
        print(f"\nSaving to {args.output}...")
        save_dataset(filtered_dataset, args.output, args.format)
        print("\n✅ Dataset filtered and saved successfully!")
    else:
        print("\n(Preview mode - no data saved)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
