"""GRPO-style kernel trainer with real trl.GRPOTrainer integration."""

from __future__ import annotations

import ast
import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import time as _time
from typing import Any, Optional
import warnings

from kite.adapters.ipw_adapter import IPWAdapter
from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.classification.prompt_hints import build_energy_aware_prompt
from kite.types import KernelCandidate, KernelTask
from kite.adapters.kevin_style_rollouts import RolloutConfig, filter_trajectories, grouped_rollouts
from kite.policies.qwen_policy import QwenPolicy
from kite.rewards.grpo_reward import GRPOMultiMetricRewardConfig, compute_grpo_multi_metric_reward
from kite.rewards.ipw_reward import IPWRewardConfig, compute_ipw_reward
from kite.telemetry.energy_capture import EnergyCapture
from kite.telemetry.phase_attribution import attribute_prefill_decode
from kite.utils.logging import get_logger
from kite.utils.serialization import load_yaml, save_json, save_jsonl

logger = get_logger(__name__)


@dataclass(slots=True)
class GRPOKernelConfig:
    output_dir: Path = Path("checkpoints/kernel_grpo")
    epochs: int = 3
    group_size: int = 8
    keep_top_k: int = 4
    energy_aware: bool = False
    telemetry_trace_dir: Optional[Path] = Path("data/telemetry/runs")
    ipw_profile_dir: Optional[Path] = None
    allow_synthetic_fallback: bool = True
    model_config_path: Optional[Path] = None
    lora_rank: int = 64
    lora_alpha: int = 16
    learning_rate: float = 5e-6
    batch_size: int = 4
    max_tasks: Optional[int] = None
    eval_num_correct_trials: int = 3
    eval_num_perf_trials: int = 25
    max_completion_length: int = 1024
    beta: float = 0.04
    correctness_bias_epochs: int = 2
    failure_log_every_steps: int = 10
    reward_alpha_speedup: float = 1.0
    reward_beta_joules: float = 0.0
    reward_gamma_latency: float = 0.25
    reward_delta_avg_power: float = 0.01
    reward_eta_runtime: float = 0.10
    reward_correctness_bonus: float = 0.0
    reward_compile_fail: float = -1.0
    reward_incorrect: float = -0.5
    reward_oom_penalty: float = 0.5
    reward_sla_latency_s: float = 1.0
    reward_ipw_blend_weight: float = 0.0
    resume_from_checkpoint: Optional[str] = None
    eval_timeout_seconds: float = 120.0


class GRPOKernelTrainer:
    def __init__(
        self,
        adapter: KernelBenchAdapter,
        policy: QwenPolicy,
        config: GRPOKernelConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.config = config or GRPOKernelConfig()
        self.energy_capture = EnergyCapture()
        self.ipw_adapter = IPWAdapter()
        if self.config.model_config_path:
            self._load_yaml_overrides()

    def _load_yaml_overrides(self) -> None:
        try:
            cfg = load_yaml(self.config.model_config_path)
            lora = cfg.get("model", {}).get("lora", {})
            train = cfg.get("training", {})
            if lora.get("rank"):
                self.config.lora_rank = int(lora["rank"])
            if lora.get("alpha"):
                self.config.lora_alpha = int(lora["alpha"])
            if train.get("learning_rate"):
                self.config.learning_rate = float(train["learning_rate"])
            if train.get("batch_size"):
                self.config.batch_size = int(train["batch_size"])
        except Exception:
            pass

    def run(self) -> dict[str, object]:
        if self._can_train_real():
            return self._run_real_grpo()
        return self._run_stub_grpo()

    @staticmethod
    def _can_train_real() -> bool:
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401
            import trl  # type: ignore  # noqa: F401
            import peft  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _run_real_grpo(self) -> dict[str, object]:
        """Real GRPO training using trl.GRPOTrainer."""
        warnings.filterwarnings(
            "ignore",
            message=r".*Passing `generation_config` together with generation-related arguments=.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Passing `generation_config` together with generation-related arguments=.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*generation_config.*disable_compile.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*generation_config.*disable_compile.*",
            category=UserWarning,
        )

        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig, TaskType  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import GRPOTrainer, GRPOConfig  # type: ignore

        model_name = self.policy.config.model_name
        dtype = torch.bfloat16
        cache_dir = self.policy.config.hf_cache_dir or os.environ.get("KITE_HF_CACHE")
        local_files_only = bool(self.policy.config.local_files_only)
        if not local_files_only:
            local_env = os.environ.get("KITE_HF_LOCAL_FILES_ONLY", "").strip().lower()
            local_files_only = local_env in {"1", "true", "yes", "on"}
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if cache_dir:
            cache_dir = str(Path(cache_dir).expanduser().resolve())
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        model_source = self._resolve_model_source(
            model_name=model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        logger.info(
            "Loading model for GRPO: %s (source=%s, cache_dir=%s, local_files_only=%s)",
            model_name,
            model_source,
            cache_dir or "default",
            local_files_only,
        )
        hf_kwargs: dict[str, object] = {"trust_remote_code": True}
        if cache_dir:
            hf_kwargs["cache_dir"] = cache_dir
        if local_files_only:
            hf_kwargs["local_files_only"] = True

        tokenizer = AutoTokenizer.from_pretrained(model_source, padding_side="left", **hf_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs: dict[str, object] = {"dtype": dtype, "trust_remote_code": True}
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if local_files_only:
            model_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

        vocab_size = getattr(model.config, "vocab_size", None) or len(tokenizer)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None)
        if pad_id is None or (vocab_size is not None and (pad_id < 0 or pad_id >= vocab_size)):
            pad_id = 0
        eos_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", None)
        if eos_id is None or (vocab_size is not None and (eos_id < 0 or eos_id >= vocab_size)):
            eos_id = pad_id
        tokenizer.pad_token_id = pad_id
        tokenizer.eos_token_id = eos_id
        model.config.pad_token_id = pad_id
        model.config.eos_token_id = eos_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = pad_id
            model.generation_config.eos_token_id = eos_id
        logger.info("Token IDs: pad=%d, eos=%d, vocab_size=%d", pad_id, eos_id, vocab_size)

        if self.policy.config.lora_weights_path:
            from peft import PeftModel  # type: ignore

            logger.info("Loading SFT LoRA checkpoint: %s", self.policy.config.lora_weights_path)
            model = PeftModel.from_pretrained(model, self.policy.config.lora_weights_path)
            model = model.merge_and_unload()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )

        tasks = self.adapter.discover_tasks()
        if self.config.max_tasks is not None:
            tasks = tasks[: max(1, int(self.config.max_tasks))]
        telemetry_corpus = self._load_telemetry_corpus()
        if self.config.energy_aware and not telemetry_corpus and not self.config.allow_synthetic_fallback:
            raise RuntimeError(
                "Energy-aware GRPO requested with --no-synthetic-fallback but no telemetry traces were found. "
                "Provide --telemetry-trace-dir and/or --ipw-profile-dir with valid traces."
            )
        telemetry_idx = 0

        adapter = KernelBenchAdapter(
            kernelbench_root=self.adapter.kernelbench_root,
            enable_kernelbench_eval=self.adapter.enable_kernelbench_eval,
            num_correct_trials=self.config.eval_num_correct_trials,
            num_perf_trials=self.config.eval_num_perf_trials,
            timing_method=self.adapter.timing_method,
            backend=self.adapter.backend,
            precision=self.adapter.precision,
            verbose=self.adapter.verbose,
        )
        energy_aware = self.config.energy_aware
        energy_capture = self.energy_capture
        ipw_adapter = self.ipw_adapter
        config = self.config
        policy = self.policy
        reward_config = self._build_grpo_reward_config(energy_aware=energy_aware)
        ipw_config = self._build_ipw_reward_config(energy_aware=energy_aware)
        ipw_blend_weight = float(self.config.reward_ipw_blend_weight)
        reward_steps = 0
        failure_counts: dict[str, int] = {}

        def _record_failure(reason: str) -> None:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

        def _completion_to_code(completion: Any) -> str:
            text: str
            if isinstance(completion, list):
                if completion and isinstance(completion[0], dict):
                    text = str(completion[0].get("content", ""))
                elif completion:
                    text = str(completion[0])
                else:
                    text = ""
            elif isinstance(completion, dict):
                text = str(completion.get("content", ""))
            else:
                text = str(completion)

            code = policy.extract_code(text).strip()
            return code if code else text.strip()

        eval_timeout = config.eval_timeout_seconds
        _eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _evaluate_with_timeout(task: KernelTask, code: str) -> KernelCandidate:
            fut = _eval_executor.submit(adapter.evaluate_candidate, task, code)
            try:
                return fut.result(timeout=eval_timeout)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                return KernelCandidate(
                    task_id=task.task_id,
                    code=code,
                    compile_ok=False, correct=False,
                    runtime_ms=None, speedup=None,
                    compile_log=f"evaluation timed out after {eval_timeout}s",
                    correctness_log=None, reference_runtime_ms=None,
                    logs={"timeout": True},
                )

        def kernel_reward_fn(completions: list[str], **kwargs) -> list[float]:
            nonlocal telemetry_idx, reward_steps
            rewards = []
            step_t0 = _time.perf_counter()
            n_precheck_fail = 0
            n_compile_fail = 0
            n_correct = 0
            n_timeout = 0
            for i, completion in enumerate(completions):
                code = _completion_to_code(completion)
                task_idx = i % len(tasks)
                task = tasks[task_idx]
                precheck_error = self._precheck_candidate_code(task, code)
                if precheck_error is not None:
                    n_precheck_fail += 1
                    _record_failure(precheck_error)
                    candidate = KernelCandidate(
                        task_id=task.task_id,
                        code=code,
                        compile_ok=False,
                        correct=False,
                        runtime_ms=None,
                        speedup=None,
                        compile_log=precheck_error,
                        correctness_log=None,
                        reference_runtime_ms=None,
                        logs={"precheck_error": precheck_error},
                    )
                else:
                    candidate = _evaluate_with_timeout(task, code)
                    if candidate.logs.get("timeout"):
                        n_timeout += 1
                        _record_failure("kernelbench:timeout")
                    elif not candidate.compile_ok:
                        n_compile_fail += 1
                        _record_failure("kernelbench:compile_fail")
                    elif not candidate.correct:
                        _record_failure("kernelbench:correctness_fail")
                    else:
                        n_correct += 1

                candidate.logs["kernel_type"] = task.kernel_type
                candidate.logs["level"] = task.level
                avg_power_w = self._maybe_float(candidate.logs.get("avg_power_w"))
                joules = self._maybe_float(candidate.logs.get("energy_j"))
                if energy_aware:
                    if telemetry_corpus:
                        trace = telemetry_corpus[telemetry_idx % len(telemetry_corpus)]
                        telemetry_idx += 1
                    else:
                        trace = energy_capture.synthetic_trace(steps=120)
                    if not trace.phase_segments:
                        trace = attribute_prefill_decode(trace, ttft_s=0.4)
                    summary = ipw_adapter.summarize(trace, input_tokens=512, output_tokens=128)
                    if avg_power_w is None:
                        avg_power_w = self._maybe_float(summary.avg_power_w)
                    if joules is None:
                        joules = self._maybe_float(summary.total_energy_j)

                avg_mem_util = self._maybe_float(candidate.logs.get("avg_mem_util_pct"))
                reward = compute_grpo_multi_metric_reward(
                    compile_ok=candidate.compile_ok,
                    correct=candidate.correct,
                    speedup=candidate.speedup,
                    runtime_ms=candidate.runtime_ms,
                    joules=joules,
                    avg_power_w=avg_power_w,
                    p95_latency_s=(candidate.runtime_ms or 0.0) / 1000.0,
                    compile_log=candidate.compile_log,
                    correctness_log=candidate.correctness_log,
                    config=reward_config,
                    kernel_type=task.kernel_type,
                    avg_mem_util_pct=avg_mem_util,
                )
                total_reward = reward.total
                if ipw_blend_weight != 0.0:
                    ipw_reward = compute_ipw_reward(
                        compile_ok=candidate.compile_ok,
                        correct=candidate.correct,
                        speedup=candidate.speedup,
                        joules=joules,
                        p95_latency_s=(candidate.runtime_ms or 0.0) / 1000.0,
                        sla_latency_s=float(self.config.reward_sla_latency_s),
                        config=ipw_config,
                    )
                    total_reward += ipw_blend_weight * ipw_reward.total
                rewards.append(total_reward)
            reward_steps += 1
            step_elapsed = _time.perf_counter() - step_t0
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            logger.info(
                "Reward step %d: %d completions in %.1fs | correct=%d precheck_fail=%d compile_fail=%d timeout=%d | avg_reward=%.3f",
                reward_steps, len(completions), step_elapsed,
                n_correct, n_precheck_fail, n_compile_fail, n_timeout, avg_reward,
            )
            if config.failure_log_every_steps > 0 and reward_steps % config.failure_log_every_steps == 0:
                if failure_counts:
                    parts = ", ".join(
                        f"{k}={v}" for k, v in sorted(failure_counts.items(), key=lambda kv: kv[1], reverse=True)
                    )
                    logger.info("GRPO failure reasons @step %d: %s", reward_steps, parts)
            return rewards

        prompts = []
        for task in tasks:
            ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
            prompt = build_energy_aware_prompt(ref_src, kernel_type=task.kernel_type)
            prompts.append([{"role": "user", "content": prompt}])

        train_dataset = Dataset.from_dict({"prompt": prompts})

        lora_out = self.config.output_dir / "lora_weights"
        effective_batch_size = self.config.batch_size
        if effective_batch_size % self.config.group_size != 0:
            # trl.GRPOConfig requires generation_batch_size to be divisible by num_generations.
            effective_batch_size = (
                (effective_batch_size + self.config.group_size - 1) // self.config.group_size
            ) * self.config.group_size
            logger.warning(
                "Adjusted per_device_train_batch_size from %d to %d to satisfy divisibility by num_generations=%d",
                self.config.batch_size,
                effective_batch_size,
                self.config.group_size,
            )

        grpo_config = GRPOConfig(
            output_dir=str(self.config.output_dir / "runs"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=effective_batch_size,
            num_generations=self.config.group_size,
            max_completion_length=self.config.max_completion_length,
            beta=self.config.beta,
            learning_rate=self.config.learning_rate,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            seed=42,
            bf16=True,
            temperature=0.8,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            reward_funcs=kernel_reward_fn,
            train_dataset=train_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        resume_ckpt = self.config.resume_from_checkpoint
        if resume_ckpt is None:
            runs_dir = self.config.output_dir / "runs"
            if runs_dir.exists():
                ckpts = sorted(runs_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    resume_ckpt = str(ckpts[-1])
                    logger.info("Auto-detected checkpoint to resume from: %s", resume_ckpt)

        logger.info("Starting GRPO training (%d epochs, %d tasks, group_size=%d)",
                     self.config.epochs, len(tasks), self.config.group_size)
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        logger.info("GRPO training complete: %s", train_result.metrics)

        lora_out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(lora_out))
        tokenizer.save_pretrained(str(lora_out))

        task_type_dist = {}
        for t in tasks:
            task_type_dist[t.kernel_type] = task_type_dist.get(t.kernel_type, 0) + 1
        checkpoint = {
            "stage": "energy_grpo" if self.config.energy_aware else "kernel_grpo",
            "epochs": self.config.epochs,
            "mode": "trained",
            "lora_weights_path": str(lora_out),
            "train_loss": train_result.metrics.get("train_loss"),
            "num_tasks": len(tasks),
            "batch_size": effective_batch_size,
            "group_size": self.config.group_size,
            "kernel_type_distribution": task_type_dist,
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint

    def _run_stub_grpo(self) -> dict[str, object]:
        """Stub fallback: run grouped rollouts and record rewards without gradient updates."""
        tasks = self.adapter.discover_tasks()
        rollout_cfg = RolloutConfig(group_size=self.config.group_size)
        telemetry_corpus = self._load_telemetry_corpus()
        if self.config.energy_aware and not telemetry_corpus and not self.config.allow_synthetic_fallback:
            raise RuntimeError(
                "Energy-aware GRPO requested with --no-synthetic-fallback but no telemetry traces were found. "
                "Provide --telemetry-trace-dir and/or --ipw-profile-dir with valid traces."
            )
        telemetry_idx = 0
        reward_config = self._build_grpo_reward_config(energy_aware=self.config.energy_aware)
        ipw_config = self._build_ipw_reward_config(energy_aware=self.config.energy_aware)
        ipw_blend_weight = float(self.config.reward_ipw_blend_weight)

        history: list[dict[str, object]] = []

        for epoch in range(1, self.config.epochs + 1):
            epoch_rewards: list[float] = []
            for task in tasks:
                candidates = grouped_rollouts(self.policy, task, rollout_cfg)
                shortlisted = filter_trajectories(candidates, keep_top_k=self.config.keep_top_k)

                for cand in shortlisted:
                    avg_power_w = self._maybe_float(cand.logs.get("avg_power_w"))
                    joules = self._maybe_float(cand.logs.get("energy_j"))
                    if self.config.energy_aware:
                        if telemetry_corpus:
                            trace = telemetry_corpus[telemetry_idx % len(telemetry_corpus)]
                            telemetry_idx += 1
                        else:
                            trace = self.energy_capture.synthetic_trace(steps=120)

                        if not trace.phase_segments:
                            trace = attribute_prefill_decode(trace, ttft_s=0.4)

                        summary = self.ipw_adapter.summarize(trace, input_tokens=512, output_tokens=128)
                        if avg_power_w is None:
                            avg_power_w = self._maybe_float(summary.avg_power_w)
                        if joules is None:
                            joules = self._maybe_float(summary.total_energy_j)

                    avg_mem_util_s = self._maybe_float(cand.logs.get("avg_mem_util_pct"))
                    reward = compute_grpo_multi_metric_reward(
                        compile_ok=cand.compile_ok,
                        correct=cand.correct,
                        speedup=cand.speedup,
                        runtime_ms=cand.runtime_ms,
                        joules=joules,
                        avg_power_w=avg_power_w,
                        p95_latency_s=(cand.runtime_ms or 0.0) / 1000.0,
                        compile_log=cand.compile_log,
                        correctness_log=cand.correctness_log,
                        config=reward_config,
                        kernel_type=task.kernel_type,
                        avg_mem_util_pct=avg_mem_util_s,
                    )
                    total_reward = reward.total
                    if ipw_blend_weight != 0.0:
                        ipw_reward = compute_ipw_reward(
                            compile_ok=cand.compile_ok,
                            correct=cand.correct,
                            speedup=cand.speedup,
                            joules=joules,
                            p95_latency_s=(cand.runtime_ms or 0.0) / 1000.0,
                            sla_latency_s=float(self.config.reward_sla_latency_s),
                            config=ipw_config,
                        )
                        total_reward += ipw_blend_weight * ipw_reward.total
                    epoch_rewards.append(total_reward)

                    history.append(
                        {
                            "epoch": epoch,
                            "task_id": task.task_id,
                            "kernel_type": task.kernel_type,
                            "level": task.level,
                            "compile_ok": cand.compile_ok,
                            "correct": cand.correct,
                            "runtime_ms": cand.runtime_ms,
                            "speedup": cand.speedup,
                            "energy_j": joules,
                            "avg_power_w": avg_power_w,
                            "avg_gpu_util_pct": self._maybe_float(cand.logs.get("avg_gpu_util_pct")),
                            "avg_mem_util_pct": self._maybe_float(cand.logs.get("avg_mem_util_pct")),
                            "avg_temp_c": self._maybe_float(cand.logs.get("avg_temp_c")),
                            "avg_sm_clock_mhz": self._maybe_float(cand.logs.get("avg_sm_clock_mhz")),
                            "avg_mem_clock_mhz": self._maybe_float(cand.logs.get("avg_mem_clock_mhz")),
                            "avg_mem_used_mb": self._maybe_float(cand.logs.get("avg_mem_used_mb")),
                            "reward": total_reward,
                        }
                    )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.config.output_dir / "training_history.jsonl", history)

        avg_reward = sum(item["reward"] for item in history) / len(history) if history else 0.0
        by_kernel_type = _aggregate_by_kernel_type(history)
        checkpoint = {
            "stage": "energy_grpo" if self.config.energy_aware else "kernel_grpo",
            "epochs": self.config.epochs,
            "num_records": len(history),
            "avg_reward": avg_reward,
            "mode": "stub",
            "by_kernel_type": by_kernel_type,
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        save_json(self.config.output_dir / "energy_by_kernel_type.json", by_kernel_type)
        return checkpoint

    def _build_grpo_reward_config(self, energy_aware: bool) -> GRPOMultiMetricRewardConfig:
        beta = float(self.config.reward_beta_joules)
        if not energy_aware:
            beta = 0.0
        return GRPOMultiMetricRewardConfig(
            alpha_speedup=float(self.config.reward_alpha_speedup),
            beta_joules=beta,
            gamma_latency=float(self.config.reward_gamma_latency),
            delta_avg_power=float(self.config.reward_delta_avg_power),
            eta_runtime=float(self.config.reward_eta_runtime),
            correctness_bonus=float(self.config.reward_correctness_bonus),
            compile_fail_reward=float(self.config.reward_compile_fail),
            incorrect_reward=float(self.config.reward_incorrect),
            oom_penalty=float(self.config.reward_oom_penalty),
            sla_latency_s=float(self.config.reward_sla_latency_s),
        )

    def _build_ipw_reward_config(self, energy_aware: bool) -> IPWRewardConfig:
        beta = float(self.config.reward_beta_joules)
        if not energy_aware:
            beta = 0.0
        return IPWRewardConfig(
            alpha_speedup=float(self.config.reward_alpha_speedup),
            beta_joules=beta,
            gamma_latency=float(self.config.reward_gamma_latency),
            compile_fail_reward=float(self.config.reward_compile_fail),
            incorrect_reward=float(self.config.reward_incorrect),
        )

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_telemetry_corpus(self):
        return self.energy_capture.load_trace_corpus(
            trace_dir=self.config.telemetry_trace_dir,
            ipw_profile_dir=self.config.ipw_profile_dir,
            allow_synthetic_fallback=self.config.allow_synthetic_fallback,
        )

    @staticmethod
    def _forward_arity_from_source(source: str, class_name: str) -> int | None:
        if not source:
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        positional = [
                            arg.arg for arg in item.args.args
                            if arg.arg not in {"self"}
                        ]
                        return len(positional)
        return None

    def _precheck_candidate_code(self, task: KernelTask, code: str) -> str | None:
        if not code or not code.strip():
            return "precheck: empty completion"
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"precheck: syntax_error: {exc}"

        if "class ModelNew" not in code:
            return "precheck: missing ModelNew class"

        ref_src = str(task.metadata.get("ref_arch_src", "") or "")
        ref_arity = self._forward_arity_from_source(ref_src, "Model")
        cand_arity = self._forward_arity_from_source(code, "ModelNew")
        if cand_arity is None:
            return "precheck: missing ModelNew.forward method"
        if ref_arity is not None and cand_arity is not None and ref_arity != cand_arity:
            return (
                f"precheck: forward_arity_mismatch ref={ref_arity} candidate={cand_arity}"
            )
        return None

    @staticmethod
    def _resolve_model_source(
        model_name: str,
        cache_dir: str | None,
        local_files_only: bool,
    ) -> str:
        if local_files_only and not cache_dir:
            raise FileNotFoundError(
                "local_files_only=True requires a cache directory. "
                "Set KITE_HF_CACHE or pass --hf-cache-dir."
            )
        if not local_files_only:
            return model_name

        repo_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        refs_main = repo_dir / "refs" / "main"

        if refs_main.exists():
            ref = refs_main.read_text().strip()
            candidate = snapshots_dir / ref
            if candidate.exists():
                return str(candidate)

        if snapshots_dir.exists():
            snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if snapshots:
                snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(snapshots[0])

        raise FileNotFoundError(
            f"Local model cache not found for {model_name} under {cache_dir}. "
            "Run once without local-files-only to populate cache."
        )


def _aggregate_by_kernel_type(history: list[dict]) -> dict:
    """Group training history by kernel_type and compute per-type stats.

    Returns a dict keyed by kernel_type with aggregate metrics including
    rich GPU telemetry signals for energy-vs-compute analysis.
    """
    groups: dict[str, list[dict]] = {}
    for row in history:
        kt = row.get("kernel_type", "unknown")
        groups.setdefault(kt, []).append(row)

    def _mean(vals: list) -> float | None:
        return sum(vals) / len(vals) if vals else None

    def _collect(rows: list[dict], key: str) -> list[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    result = {}
    for kt, rows in sorted(groups.items()):
        runtimes = _collect(rows, "runtime_ms")
        energies = _collect(rows, "energy_j")
        powers = _collect(rows, "avg_power_w")
        rewards = _collect(rows, "reward")
        gpu_utils = _collect(rows, "avg_gpu_util_pct")
        mem_utils = _collect(rows, "avg_mem_util_pct")
        temps = _collect(rows, "avg_temp_c")
        sm_clocks = _collect(rows, "avg_sm_clock_mhz")
        mem_clocks = _collect(rows, "avg_mem_clock_mhz")
        mem_used = _collect(rows, "avg_mem_used_mb")
        correct = sum(1 for r in rows if r.get("correct"))
        compiled = sum(1 for r in rows if r.get("compile_ok"))
        n = len(rows)

        avg_rt = _mean(runtimes)
        avg_en = _mean(energies)

        result[kt] = {
            "count": n,
            "compiled": compiled,
            "correct": correct,
            "compile_rate": compiled / n if n else 0.0,
            "correctness_rate": correct / n if n else 0.0,
            "avg_runtime_ms": avg_rt,
            "avg_energy_j": avg_en,
            "avg_power_w": _mean(powers),
            "avg_reward": _mean(rewards),
            "energy_per_ms": avg_en / avg_rt if avg_en and avg_rt and avg_rt > 0 else None,
            "avg_gpu_util_pct": _mean(gpu_utils),
            "avg_mem_util_pct": _mean(mem_utils),
            "avg_temp_c": _mean(temps),
            "avg_sm_clock_mhz": _mean(sm_clocks),
            "avg_mem_clock_mhz": _mean(mem_clocks),
            "avg_mem_used_mb": _mean(mem_used),
            "compute_to_mem_ratio": (
                _mean(gpu_utils) / _mean(mem_utils)
                if gpu_utils and mem_utils and _mean(mem_utils) and _mean(mem_utils) > 0
                else None
            ),
        }
    return result
