"""KernelBench adapter utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any
from typing import Iterable, List

from kite.classification.kernel_classifier import classify_kernel
from kite.measurement.energy_integrate import integrate_energy, integrate_rich_energy
from kite.measurement.nvml_power import NvmlPowerSampler, NvmlRichSampler, PowerSample
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
        levels: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self.kernelbench_root = kernelbench_root
        self.enable_kernelbench_eval = enable_kernelbench_eval
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.timing_method = timing_method
        self.backend = backend
        self.precision = precision
        self.verbose = verbose
        self.levels = tuple(levels) if levels else (1, 2, 3, 4)

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
            compile_log=None if compile_ok else "proxy compile check failed",
            correctness_log=None if correct else "proxy correctness check failed",
            reference_runtime_ms=baseline_runtime_ms,
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

        sampler = NvmlRichSampler(device_index=0, sampling_interval_ms=50.0)
        sampler.start()
        eval_t0 = time.perf_counter()
        eval_error: Exception | None = None
        result = None
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
        except Exception as exc:
            eval_error = exc
        eval_t1 = time.perf_counter()
        rich_samples = sampler.stop()
        duration_s = max(0.0, eval_t1 - eval_t0)
        if len(rich_samples) < 2:
            fallback = sampler.read_sample()
            from kite.measurement.nvml_power import GpuSample
            rich_samples = [
                GpuSample(timestamp_s=0.0, power_w=fallback.power_w,
                          gpu_util_pct=fallback.gpu_util_pct, mem_util_pct=fallback.mem_util_pct,
                          temp_c=fallback.temp_c, sm_clock_mhz=fallback.sm_clock_mhz,
                          mem_clock_mhz=fallback.mem_clock_mhz, mem_used_mb=fallback.mem_used_mb),
                GpuSample(timestamp_s=duration_s, power_w=fallback.power_w,
                          gpu_util_pct=fallback.gpu_util_pct, mem_util_pct=fallback.mem_util_pct,
                          temp_c=fallback.temp_c, sm_clock_mhz=fallback.sm_clock_mhz,
                          mem_clock_mhz=fallback.mem_clock_mhz, mem_used_mb=fallback.mem_used_mb),
            ]
        window = integrate_rich_energy(rich_samples)
        sampler.close()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        def _base_logs() -> dict[str, Any]:
            return {
                "kernelbench_eval": True,
                "num_correct_trials": self.num_correct_trials,
                "num_perf_trials": self.num_perf_trials,
                "timing_method": self.timing_method,
                "backend": self.backend,
                "precision": self.precision,
                "avg_power_w": window.avg_power_w,
                "energy_j": window.energy_j,
                "eval_wall_ms": duration_s * 1000.0,
                "avg_gpu_util_pct": window.avg_gpu_util_pct,
                "avg_mem_util_pct": window.avg_mem_util_pct,
                "avg_temp_c": window.avg_temp_c,
                "avg_sm_clock_mhz": window.avg_sm_clock_mhz,
                "avg_mem_clock_mhz": window.avg_mem_clock_mhz,
                "avg_mem_used_mb": window.avg_mem_used_mb,
            }

        if eval_error is not None:
            err_name = type(eval_error).__name__
            err_text = f"{err_name}: {eval_error}"
            err_lower = err_text.lower()
            logs = _base_logs()
            logs["metadata"] = {
                "exception_name": err_name,
                "exception": str(eval_error),
                "oom": ("out of memory" in err_lower),
            }
            return KernelCandidate(
                task_id=task.task_id,
                code=candidate_code,
                compile_ok=False,
                correct=False,
                runtime_ms=duration_s * 1000.0,
                speedup=0.0,
                compile_log=f"kernelbench eval exception: {err_text}",
                correctness_log=str(eval_error),
                reference_runtime_ms=baseline_runtime_ms,
                logs=logs,
            )
        if result is None:
            logs = _base_logs()
            logs["metadata"] = {"error": "kernelbench returned no result"}
            return KernelCandidate(
                task_id=task.task_id,
                code=candidate_code,
                compile_ok=False,
                correct=False,
                runtime_ms=duration_s * 1000.0,
                speedup=0.0,
                compile_log="kernelbench returned no result",
                correctness_log=None,
                reference_runtime_ms=baseline_runtime_ms,
                logs=logs,
            )

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

        logs = _base_logs()
        logs["metadata"] = metadata
        if ref_runtime_us > 0:
            logs["ref_runtime_us"] = ref_runtime_us

        return KernelCandidate(
            task_id=task.task_id,
            code=candidate_code,
            compile_ok=compiled,
            correct=correctness,
            runtime_ms=runtime_ms,
            speedup=speedup,
            compile_log=None if compiled else str(metadata.get("compile_error", "kernelbench compilation failed")),
            correctness_log=None if correctness else str(metadata.get("correctness_error", "kernelbench correctness failed")),
            reference_runtime_ms=(ref_runtime_us / 1000.0) if ref_runtime_us > 0 else baseline_runtime_ms,
            logs=logs,
        )

    def _discover_tasks_from_kernelbench_dataset(self) -> List[KernelTask]:
        try:
            self._ensure_kernelbench_on_path()
            from kernelbench.dataset import construct_kernelbench_dataset  # type: ignore
        except Exception:
            return []

        tasks: list[KernelTask] = []
        for level in self.levels:
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

                ktype = classify_kernel(
                    task_name=task_id,
                    problem_name=problem_name,
                    reference_code=code,
                    level=level,
                )
                tasks.append(
                    KernelTask(
                        task_id=task_id,
                        level=level,
                        prompt=(
                            "Optimize this PyTorch model with a custom GPU kernel implementation.\n\n"
                            f"{code}"
                        ),
                        reference_kernel=code,
                        kernel_type=ktype,
                        metadata={
                            "source": "kernelbench_dataset_api",
                            "level": level,
                            "problem_id": problem_id,
                            "problem_name": problem_name,
                            "ref_arch_src": code,
                            "kernel_type": ktype,
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
            task_id = str(row.get("task_id", f"task_{i}"))
            level = int(row.get("level", 1))
            ref_kernel = str(row.get("reference_kernel", row.get("reference", "")))
            meta = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {}
            ktype = str(row.get("kernel_type", "")) or meta.get("kernel_type", "")
            if not ktype:
                ktype = classify_kernel(
                    task_name=task_id,
                    problem_name=meta.get("problem_name", ""),
                    reference_code=ref_kernel,
                    level=level,
                )
            meta.setdefault("kernel_type", ktype)
            tasks.append(
                KernelTask(
                    task_id=task_id,
                    level=level,
                    prompt=str(row.get("prompt", row.get("problem", ""))),
                    reference_kernel=ref_kernel,
                    kernel_type=ktype,
                    metadata=meta,
                )
            )
        return tasks or KernelBenchAdapter._default_tasks()

    @staticmethod
    def _default_tasks() -> List[KernelTask]:
        from kite.types import KERNEL_TYPE_NORM, KERNEL_TYPE_ATTENTION, KERNEL_TYPE_MODEL
        return [
            KernelTask(
                task_id="kb_default_1",
                level=1,
                prompt="Implement a fused layernorm + residual kernel.",
                reference_kernel="def kernel(x, y): return x + y",
                kernel_type=KERNEL_TYPE_NORM,
                metadata={"source": "default", "kernel_type": KERNEL_TYPE_NORM},
            ),
            KernelTask(
                task_id="kb_default_2",
                level=2,
                prompt="Implement attention score matmul tile kernel.",
                reference_kernel="def kernel(q, k): return q @ k.T",
                kernel_type=KERNEL_TYPE_ATTENTION,
                metadata={"source": "default", "kernel_type": KERNEL_TYPE_ATTENTION},
            ),
            KernelTask(
                task_id="kb_default_3",
                level=3,
                prompt="Implement decode-step KV cache update kernel.",
                reference_kernel="def kernel(cache, token): return cache",
                kernel_type=KERNEL_TYPE_MODEL,
                metadata={"source": "default", "kernel_type": KERNEL_TYPE_MODEL},
            ),
        ]


def load_tasks(path: Path) -> List[KernelTask]:
    rows = load_jsonl(path)
    tasks: list[KernelTask] = []
    for row in rows:
        task_id = row["task_id"]
        level = int(row["level"])
        ref_kernel = row["reference_kernel"]
        meta = dict(row.get("metadata", {}))
        ktype = row.get("kernel_type", "") or meta.get("kernel_type", "")
        if not ktype:
            ktype = classify_kernel(
                task_name=task_id,
                problem_name=meta.get("problem_name", ""),
                reference_code=ref_kernel,
                level=level,
            )
        tasks.append(
            KernelTask(
                task_id=task_id,
                level=level,
                prompt=row["prompt"],
                reference_kernel=ref_kernel,
                kernel_type=ktype,
                metadata=meta,
            )
        )
    return tasks


def export_candidates(path: Path, candidates: Iterable[KernelCandidate]) -> None:
    save_jsonl(path, (asdict(candidate) for candidate in candidates))
