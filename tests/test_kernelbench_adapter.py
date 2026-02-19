from pathlib import Path

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.utils.serialization import save_jsonl


def test_kernelbench_adapter_loads_tasks_and_builds_splits(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    save_jsonl(
        root / "tasks.jsonl",
        [
            {
                "task_id": "a",
                "level": 1,
                "prompt": "p",
                "reference_kernel": "def kernel(): return 1",
            },
            {
                "task_id": "b",
                "level": 2,
                "prompt": "q",
                "reference_kernel": "def kernel(): return 2",
            },
        ],
    )

    adapter = KernelBenchAdapter(root)
    tasks = adapter.discover_tasks()
    assert len(tasks) == 2

    out = tmp_path / "processed"
    paths = adapter.build_splits(out)
    assert paths["train"].exists()
    assert paths["all"].exists()
