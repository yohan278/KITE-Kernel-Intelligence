"""KernelBench adapter utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from kite.types import KernelCandidate, KernelTask
from kite.utils.serialization import load_jsonl, save_jsonl


class KernelBenchAdapter:
    """Adapter around KernelBench-formatted tasks and evaluations."""

    def __init__(self, kernelbench_root: Path) -> None:
        self.kernelbench_root = kernelbench_root

    def discover_tasks(self) -> List[KernelTask]:
        """Load tasks from known JSONL files, or return defaults."""
        candidates = [
            self.kernelbench_root / "tasks.jsonl",
            self.kernelbench_root / "data" / "tasks.jsonl",
            self.kernelbench_root / "kernelbench_tasks.jsonl",
        ]
        for path in candidates:
            if path.exists():
                return self._read_tasks_jsonl(path)

        if self.kernelbench_root.exists():
            for path in self.kernelbench_root.rglob("*.jsonl"):
                if "task" in path.name.lower():
                    return self._read_tasks_jsonl(path)

        return self._default_tasks()

    def build_splits(
        self,
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> dict[str, Path]:
        tasks = self.discover_tasks()
        n = len(tasks)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = tasks[:n_train]
        val = tasks[n_train : n_train + n_val]
        test = tasks[n_train + n_val :]

        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train": output_dir / "train.jsonl",
            "val": output_dir / "val.jsonl",
            "test": output_dir / "test.jsonl",
            "all": output_dir / "all.jsonl",
        }

        save_jsonl(paths["train"], (asdict(task) for task in train))
        save_jsonl(paths["val"], (asdict(task) for task in val))
        save_jsonl(paths["test"], (asdict(task) for task in test))
        save_jsonl(paths["all"], (asdict(task) for task in tasks))
        return paths

    def evaluate_candidate(
        self,
        task: KernelTask,
        candidate_code: str,
        baseline_runtime_ms: float = 100.0,
    ) -> KernelCandidate:
        """Lightweight local proxy evaluator.

        This function does not replace official KernelBench scoring; it is used
        for fast local loops and smoke tests.
        """
        compile_ok = "TODO" not in candidate_code and "pass" not in candidate_code
        correct = compile_ok and "return" in candidate_code and "error" not in candidate_code.lower()

        runtime_ms = max(1.0, baseline_runtime_ms * (0.6 if correct else 1.4))
        speedup = baseline_runtime_ms / runtime_ms if runtime_ms > 0 else None

        return KernelCandidate(
            task_id=task.task_id,
            code=candidate_code,
            compile_ok=compile_ok,
            correct=correct,
            runtime_ms=runtime_ms,
            speedup=speedup,
            logs={"proxy_eval": True},
        )

    @staticmethod
    def _read_tasks_jsonl(path: Path) -> List[KernelTask]:
        rows = load_jsonl(path)
        tasks: list[KernelTask] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            tasks.append(
                KernelTask(
                    task_id=str(row.get("task_id", f"task_{i}")),
                    level=int(row.get("level", 1)),
                    prompt=str(row.get("prompt", row.get("problem", ""))),
                    reference_kernel=str(row.get("reference_kernel", row.get("reference", ""))),
                    metadata=dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {},
                )
            )
        return tasks or KernelBenchAdapter._default_tasks()

    @staticmethod
    def _default_tasks() -> List[KernelTask]:
        return [
            KernelTask(
                task_id="kb_default_1",
                level=1,
                prompt="Implement a fused layernorm + residual kernel.",
                reference_kernel="def kernel(x, y): return x + y",
                metadata={"source": "default"},
            ),
            KernelTask(
                task_id="kb_default_2",
                level=2,
                prompt="Implement attention score matmul tile kernel.",
                reference_kernel="def kernel(q, k): return q @ k.T",
                metadata={"source": "default"},
            ),
            KernelTask(
                task_id="kb_default_3",
                level=3,
                prompt="Implement decode-step KV cache update kernel.",
                reference_kernel="def kernel(cache, token): return cache",
                metadata={"source": "default"},
            ),
        ]


def load_tasks(path: Path) -> List[KernelTask]:
    rows = load_jsonl(path)
    tasks: list[KernelTask] = []
    for row in rows:
        tasks.append(
            KernelTask(
                task_id=row["task_id"],
                level=int(row["level"]),
                prompt=row["prompt"],
                reference_kernel=row["reference_kernel"],
                metadata=dict(row.get("metadata", {})),
            )
        )
    return tasks


def export_candidates(path: Path, candidates: Iterable[KernelCandidate]) -> None:
    save_jsonl(path, (asdict(candidate) for candidate in candidates))
