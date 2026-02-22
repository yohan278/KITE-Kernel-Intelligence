"""GRPO-style kernel trainer with real trl.GRPOTrainer integration."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
import warnings

from kite.adapters.ipw_adapter import IPWAdapter
from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.types import KernelCandidate, KernelTask
from kite.adapters.kevin_style_rollouts import RolloutConfig, filter_trajectories, grouped_rollouts
from kite.policies.qwen_policy import QwenPolicy
from kite.rewards.energy_reward import EnergyRewardConfig, compute_energy_aware_reward
from kite.rewards.kernel_reward import staged_kernel_reward
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
    max_completion_length: int = 1024
    beta: float = 0.04
    correctness_bias_epochs: int = 2


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

        tokenizer = AutoTokenizer.from_pretrained(model_source, **hf_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs: dict[str, object] = {"dtype": dtype, "device_map": "auto", "trust_remote_code": True}
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if local_files_only:
            model_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

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
        telemetry_corpus = self._load_telemetry_corpus()
        telemetry_idx = 0

        adapter = self.adapter
        energy_aware = self.config.energy_aware
        energy_capture = self.energy_capture
        ipw_adapter = self.ipw_adapter
        config = self.config
        policy = self.policy

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

        def kernel_reward_fn(completions: list[str], **kwargs) -> list[float]:
            nonlocal telemetry_idx
            prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
            rewards = []
            for i, completion in enumerate(completions):
                code = _completion_to_code(completion)
                task_idx = i % len(tasks)
                task = tasks[task_idx]
                precheck_error = self._precheck_candidate_code(task, code)
                if precheck_error is not None:
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
                    candidate = adapter.evaluate_candidate(task, code)

                if energy_aware:
                    if telemetry_corpus:
                        trace = telemetry_corpus[telemetry_idx % len(telemetry_corpus)]
                        telemetry_idx += 1
                    else:
                        trace = energy_capture.synthetic_trace(steps=120)
                    if not trace.phase_segments:
                        trace = attribute_prefill_decode(trace, ttft_s=0.4)
                    summary = ipw_adapter.summarize(trace, input_tokens=512, output_tokens=128)
                    reward = compute_energy_aware_reward(
                        candidate=candidate,
                        summary=summary,
                        p95_latency_s=(candidate.runtime_ms or 0.0) / 1000.0,
                        sla_latency_s=1.0,
                        timeout_ms=500.0,
                        config=EnergyRewardConfig(),
                    )
                else:
                    reward = staged_kernel_reward(
                        candidate,
                        timeout_ms=500.0,
                        epoch=1,
                        correctness_bias_epochs=config.correctness_bias_epochs,
                    )
                rewards.append(reward.total)
            return rewards

        prompts = []
        for task in tasks:
            ref_src = task.metadata.get("ref_arch_src", task.reference_kernel)
            prompt = (
                "You are an expert GPU kernel engineer. "
                "Optimize this PyTorch model with a custom GPU kernel implementation "
                "that produces identical outputs and runs faster on NVIDIA H100.\n\n"
                "Output only valid Python code.\n"
                "Do not include markdown or explanations.\n"
                "Must define class ModelNew(nn.Module).\n"
                "ModelNew.forward must keep the same input argument count/order as the reference Model.forward.\n"
                "Do not use Triton.\n\n"
                f"```python\n{ref_src}\n```\n\n"
                "Write the optimized kernel:"
            )
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
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            reward_funcs=kernel_reward_fn,
            train_dataset=train_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        logger.info("Starting GRPO training (%d epochs, %d tasks, group_size=%d)",
                     self.config.epochs, len(tasks), self.config.group_size)
        train_result = trainer.train()
        logger.info("GRPO training complete: %s", train_result.metrics)

        lora_out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(lora_out))
        tokenizer.save_pretrained(str(lora_out))

        checkpoint = {
            "stage": "energy_grpo" if self.config.energy_aware else "kernel_grpo",
            "epochs": self.config.epochs,
            "mode": "trained",
            "lora_weights_path": str(lora_out),
            "train_loss": train_result.metrics.get("train_loss"),
            "num_tasks": len(tasks),
            "batch_size": effective_batch_size,
            "group_size": self.config.group_size,
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint

    def _run_stub_grpo(self) -> dict[str, object]:
        """Stub fallback: run grouped rollouts and record rewards without gradient updates."""
        tasks = self.adapter.discover_tasks()
        rollout_cfg = RolloutConfig(group_size=self.config.group_size)
        telemetry_corpus = self._load_telemetry_corpus()
        telemetry_idx = 0

        history: list[dict[str, object]] = []

        for epoch in range(1, self.config.epochs + 1):
            epoch_rewards: list[float] = []
            for task in tasks:
                candidates = grouped_rollouts(self.policy, task, rollout_cfg)
                shortlisted = filter_trajectories(candidates, keep_top_k=self.config.keep_top_k)

                for cand in shortlisted:
                    if self.config.energy_aware:
                        if telemetry_corpus:
                            trace = telemetry_corpus[telemetry_idx % len(telemetry_corpus)]
                            telemetry_idx += 1
                        else:
                            trace = self.energy_capture.synthetic_trace(steps=120)

                        if not trace.phase_segments:
                            trace = attribute_prefill_decode(trace, ttft_s=0.4)

                        summary = self.ipw_adapter.summarize(trace, input_tokens=512, output_tokens=128)
                        reward = compute_energy_aware_reward(
                            candidate=cand,
                            summary=summary,
                            p95_latency_s=(cand.runtime_ms or 0.0) / 1000.0,
                            sla_latency_s=1.0,
                            timeout_ms=500.0,
                            config=EnergyRewardConfig(),
                        )
                    else:
                        reward = staged_kernel_reward(
                            cand,
                            timeout_ms=500.0,
                            epoch=epoch,
                            correctness_bias_epochs=self.config.correctness_bias_epochs,
                        )
                    epoch_rewards.append(reward.total)

                    history.append(
                        {
                            "epoch": epoch,
                            "task_id": task.task_id,
                            "compile_ok": cand.compile_ok,
                            "correct": cand.correct,
                            "runtime_ms": cand.runtime_ms,
                            "speedup": cand.speedup,
                            "reward": reward.total,
                        }
                    )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.config.output_dir / "training_history.jsonl", history)

        avg_reward = sum(item["reward"] for item in history) / len(history) if history else 0.0
        checkpoint = {
            "stage": "energy_grpo" if self.config.energy_aware else "kernel_grpo",
            "epochs": self.config.epochs,
            "num_records": len(history),
            "avg_reward": avg_reward,
            "mode": "stub",
        }
        save_json(self.config.output_dir / "checkpoint.json", checkpoint)
        return checkpoint

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
