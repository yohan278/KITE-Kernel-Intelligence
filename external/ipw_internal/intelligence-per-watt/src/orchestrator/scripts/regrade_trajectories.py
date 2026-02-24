"""Re-grade existing trajectories with fixed grading function."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datasets import load_from_disk, Dataset
from orchestrator.data.episode_builder import grade_answer_llm


def grade_sample(args):
    """Grade a single sample. Returns (index, sample_dict, success)."""
    idx, sample = args
    new_success = grade_answer_llm(
        sample['final_answer'] or "",
        sample['ground_truth']
    )
    sample_dict = dict(sample)
    sample_dict['success'] = new_success
    return idx, sample_dict, new_success


def regrade_checkpoint(checkpoint_dir: str, num_workers: int = 8):
    """Re-grade all trajectories in a checkpoint using parallel workers."""
    print(f"Loading dataset from {checkpoint_dir}/dataset...", flush=True)
    ds = load_from_disk(f"{checkpoint_dir}/dataset")
    print(f"Loaded {len(ds)} samples, using {num_workers} workers", flush=True)

    # Prepare work items
    work_items = [(i, ds[i]) for i in range(len(ds))]

    # Results storage (indexed to maintain order)
    results = [None] * len(ds)
    success_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(grade_sample, item): item[0] for item in work_items}

        for future in as_completed(futures):
            idx, sample_dict, success = future.result()
            results[idx] = sample_dict
            if success:
                success_count += 1
            completed += 1

            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(ds)}, Success rate: {success_count/completed*100:.1f}%", flush=True)

    # Save updated dataset
    print("Saving updated dataset...", flush=True)
    new_ds = Dataset.from_list(results)
    new_ds.save_to_disk(f"{checkpoint_dir}/dataset")

    # Update metadata
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata['statistics']['success_count'] = success_count
    metadata['statistics']['failure_count'] = len(ds) - success_count
    metadata['statistics']['success_rate'] = success_count / len(ds)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! New success rate: {success_count}/{len(ds)} ({success_count/len(ds)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-grade trajectories with fixed grading function")
    parser.add_argument("--checkpoint-dir", default="data/trajectories/checkpoint", help="Checkpoint directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    regrade_checkpoint(args.checkpoint_dir, args.workers)
