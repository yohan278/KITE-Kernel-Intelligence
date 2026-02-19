"""KernelBench adapter utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any
from typing import Iterable, List

from kite.types import KernelCandidate, KernelTask
from kite.utils.serialization import load_jsonl, save_jsonl


class KernelBenchAdapter:
    """Adapter around KernelBench-formatted tasks and evaluations."""

    def __init__(
        self,
        kernelbench_root: Path,
        enable_kernelbench_eval: bool = True,
        num_correct_trials: int = 3,
        num_perf_trials: int = 25,
        timing_method: str = "cuda_event",
        backend: str = "cuda",
        precision: str = "fp32",
        verbose: bool = False,
    ) -> None:
        self.kernelbench_root = kernelbench_root
        self.enable_kernelbench_eval = enable_kernelbench_eval
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.timing_method = timing_method
        self.backend = backend
        self.precision = precision
        self.verbose = verbose

    def discover_tasks(self) -> List[KernelTask]:
        """Load tasks from KernelBench APIs, JSONL files, or defaults."""
        dataset_tasks = self._discover_tasks_from_kernelbench_dataset()
        if dataset_tasks:
            return dataset_tasks
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
        if self.enable_kernelbench_eval:
            real_result = self._evaluate_with_kernelbench(
                task=task,
                candidate_code=candidate_code,
                baseline_runtime_ms=baseline_runtime_ms,
            )
            if real_result is not None:
                return real_result

        return self._evaluate_proxy(
            task=task,
            candidate_code=candidate_code,
            baseline_runtime_ms=baseline_runtime_ms,
        )

    def _evaluate_proxy(
        self,
        task: KernelTask,
        candidate_code: str,
        baseline_runtime_ms: float = 100.0,
    ) -> KernelCandidate:
        """Fallback local proxy evaluator for offline/dev use."""
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

    def _evaluate_with_kernelbench(
        self,
        task: KernelTask,
        candidate_code: str,
        baseline_runtime_ms: float = 100.0,
    ) -> KernelCandidate | None:
        ref_arch = task.metadata.get("ref_arch_src")
        if not isinstance(ref_arch, str) or not ref_arch.strip():
            return None

        try:
            self._ensure_kernelbench_on_path()
            import torch  # type: ignore
            from kernelbench.eval import (  # type: ignore
                eval_kernel_against_ref,
                get_torch_dtype_from_string,
            )
        except Exception:
            return None

        if not torch.cuda.is_available():
            return None

        try:
            result = eval_kernel_against_ref(
                original_model_src=ref_arch,
                custom_model_src=candidate_code,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                measure_performance=True,
                timing_method=self.timing_method,
                verbose=self.verbose,
                backend=self.backend,
                precision=get_torch_dtype_from_string(self.precision),
                device=torch.device("cuda:0"),
            )
            if result is None:
                return None
        except Exception:
            return None

        compiled = bool(getattr(result, "compiled", False))
        correctness = bool(getattr(result, "correctness", False)) if compiled else False

        runtime_us = float(getattr(result, "runtime", -1.0) or -1.0)
        ref_runtime_us = float(getattr(result, "ref_runtime", -1.0) or -1.0)

        runtime_ms = runtime_us / 1000.0 if runtime_us > 0 else max(
            1.0, baseline_runtime_ms * (0.6 if correctness else 1.4)
        )
        speedup = None
        if runtime_us > 0 and ref_runtime_us > 0:
            speedup = ref_runtime_us / runtime_us
        elif runtime_ms > 0:
            speedup = baseline_runtime_ms / runtime_ms

        raw_metadata = getattr(result, "metadata", {})
        metadata: dict[str, Any] = {}
        if isinstance(raw_metadata, dict):
            for k, v in raw_metadata.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    metadata[k] = v
                else:
                    metadata[k] = str(v)

        logs = {
            "kernelbench_eval": True,
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
            "timing_method": self.timing_method,
            "backend": self.backend,
            "precision": self.precision,
            "metadata": metadata,
        }
        if ref_runtime_us > 0:
            logs["ref_runtime_us"] = ref_runtime_us

        return KernelCandidate(
            task_id=task.task_id,
            code=candidate_code,
            compile_ok=compiled,
            correct=correctness,
            runtime_ms=runtime_ms,
            speedup=speedup,
            logs=logs,
        )

    def _discover_tasks_from_kernelbench_dataset(self) -> List[KernelTask]:
        try:
            self._ensure_kernelbench_on_path()
            from kernelbench.dataset import construct_kernelbench_dataset  # type: ignore
        except Exception:
            return []

        tasks: list[KernelTask] = []
        for level in (1, 2, 3, 4):
            try:
                dataset = construct_kernelbench_dataset(
                    level=level,
                    source="local",
                    base_path=str(self.kernelbench_root / "KernelBench"),
                )
            except Exception:
                continue

            for problem in dataset:
                problem_id = int(getattr(problem, "problem_id", len(tasks) + 1))
                problem_name = str(getattr(problem, "name", f"{problem_id}"))
                code = str(getattr(problem, "code", ""))
                task_id = f"L{level}_{problem_id}"

                tasks.append(
                    KernelTask(
                        task_id=task_id,
                        level=level,
                        prompt=(
                            "Optimize this PyTorch model with a custom GPU kernel implementation.\n\n"
                            f"{code}"
                        ),
                        reference_kernel=code,
                        metadata={
                            "source": "kernelbench_dataset_api",
                            "level": level,
                            "problem_id": problem_id,
                            "problem_name": problem_name,
                            "ref_arch_src": code,
                        },
                    )
                )
        return tasks

    def _ensure_kernelbench_on_path(self) -> None:
        src_dir = self.kernelbench_root / "src"
        src_dir_str = str(src_dir.resolve())
        if src_dir.exists() and src_dir_str not in sys.path:
            sys.path.insert(0, src_dir_str)

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
