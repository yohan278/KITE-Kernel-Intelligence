import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

DATASET_PATH = "gaia-benchmark/GAIA"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gaia_benchmark"


@dataclass
class GAIASample:
    """A single GAIA benchmark sample."""
    
    task_id: str
    question: str
    final_answer: str
    level: int
    annotator_metadata: str
    file_name: Optional[str] = None
    file_path: Optional[Path] = None
    
    def get_prompt(self, input_prompt: Optional[str] = None) -> str:
        """Get the formatted prompt for this sample."""
        prompt = input_prompt or DEFAULT_INPUT_PROMPT
        
        if self.file_name and self.file_path:
            file_info = (
                f"The following file is referenced in the question below and you will "
                f"likely need to use it in order to find the correct answer.\n"
                f"File name: {self.file_name}\n"
                f"File path: {self.file_path}\n"
                f"Use the file reading tools to access this file."
            )
        elif self.file_name:
            file_info = (
                f"The following file is referenced in the question: {self.file_name}\n"
                f"(Note: File path not available)"
            )
        else:
            file_info = ""
        
        return prompt.format(file=file_info, question=self.question)


def load_gaia_samples(
    subset: Literal[
        "2023_all", "2023_level1", "2023_level2", "2023_level3"
    ] = "2023_all",
    split: Literal["test", "validation"] = "validation",
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Iterator[GAIASample]:
    """
    Load GAIA benchmark samples from HuggingFace.
    
    Args:
        subset: Dataset subset (level filtering)
        split: Dataset split ("test" or "validation")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        limit: Maximum number of samples to load
        cache_dir: Cache directory for dataset files
    
    Yields:
        GAIASample objects
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dataset_location = cache_dir / "GAIA"
    
    # Download dataset and its artifacts if required
    if not dataset_location.exists():
        dataset_location.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=DATASET_PATH,
                repo_type="dataset",
                local_dir=str(dataset_location),
            )
        except Exception as ex:
            shutil.rmtree(dataset_location, ignore_errors=True)
            raise ex
    
    # Load dataset from HuggingFace
    dataset = load_dataset(
        str(dataset_location),
        name=subset,
        split=split,
    )
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    # Apply limit if specified
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    # Files location
    files_location = dataset_location / "2023" / split
    
    # Convert records to GAIASample objects
    for record in dataset:
        task_id = record["task_id"]
        
        # Discover associated files
        file_name = None
        file_path = None
        if files_location.exists():
            files = [f for f in os.listdir(files_location) if str(task_id) in f]
            if files:
                file_name = files[0]
                file_path = files_location / file_name
        
        yield GAIASample(
            task_id=task_id,
            question=record["Question"],
            final_answer=record["Final answer"],
            level=record["Level"],
            annotator_metadata=record["Annotator Metadata"],
            file_name=file_name,
            file_path=file_path,
        )


DEFAULT_INPUT_PROMPT = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

{file}

Here is the question:

{question}"""
