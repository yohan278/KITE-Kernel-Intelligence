#!/usr/bin/env python3
"""Upload trajectory dataset to HuggingFace Hub under ScalingIntelligence organization.

Usage:
    uv run python -m orchestrator.cli.upload_to_hf --dataset-path data/trajectories --repo-name my-dataset
    uv run python -m orchestrator.cli.upload_to_hf --dataset-path data/trajectories --repo-name my-dataset --public
    uv run python -m orchestrator.cli.upload_to_hf --dataset-path data/trajectories --repo-name my-dataset --commit-message "Initial upload"
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Upload trajectory dataset to HuggingFace Hub (ScalingIntelligence organization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset path
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset directory (saved TrajectoryDataset)",
    )

    # Repository name
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository name (will be uploaded as ScalingIntelligence/repo-name)",
    )

    # Privacy
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default: private)",
    )

    # HuggingFace token
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or use HF_TOKEN env var)",
    )

    # Commit message
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload trajectory dataset",
        help="Commit message for the upload",
    )

    # Organization override (defaults to ScalingIntelligence)
    parser.add_argument(
        "--org",
        type=str,
        default="ScalingIntelligence",
        help="HuggingFace organization (default: ScalingIntelligence)",
    )

    args = parser.parse_args()

    # Check dataset path exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HuggingFace token required.")
        print("Provide via --hf-token or set HF_TOKEN environment variable.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Construct repo ID
    repo_id = f"{args.org}/{args.repo_name}"

    print("=" * 60)
    print("Upload Dataset to HuggingFace Hub")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Repository: {repo_id}")
    print(f"Privacy: {'Public' if args.public else 'Private'}")
    print(f"Commit message: {args.commit_message}")
    print("=" * 60)

    # Load HuggingFace dataset
    print(f"\nLoading dataset from {dataset_path}...")
    try:
        from datasets import load_from_disk
    except ImportError:
        print("Error: datasets library required. Install with: pip install datasets")
        sys.exit(1)
    
    try:
        hf_dataset = load_from_disk(str(dataset_path))
        print(f"Loaded {len(hf_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print basic info
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(hf_dataset)}")
    if hasattr(hf_dataset, 'info') and hf_dataset.info:
        if hf_dataset.info.features:
            print(f"  Features: {list(hf_dataset.info.features.keys())}")

    # Upload to Hub
    print(f"\nUploading to HuggingFace Hub: {repo_id}...")
    try:
        hf_dataset.push_to_hub(
            repo_id=repo_id,
            private=not args.public,
            token=hf_token,
            commit_message=args.commit_message,
        )
        print(f"\n✓ Successfully uploaded dataset!")
        print(f"  View at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
