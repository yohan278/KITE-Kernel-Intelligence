"""Qwen policy wrapper with optional KernelBench generation backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Optional

from kite.types import KernelCandidate, KernelTask
from kite.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class QwenPolicyConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    temperature: float = 0.8
    max_new_tokens: int = 1024
    generation_mode: str = "stub"  # stub | local | kernelbench_server
    server_type: str = "local"
    backend: str = "cuda"
    precision: str = "fp32"
    prompt_option: str = "one_shot"
    include_hardware_info: bool = False
    hardware_gpu_name: Optional[str] = None
    custom_prompt_key: Optional[str] = None
    check_kernel: bool = False
    reasoning_effort: Optional[str] = None
    budget_tokens: int = 0
    is_reasoning_model: bool = False
    kernelbench_root: Path = Path("external/KernelBench")
    lora_weights_path: Optional[str] = None
    load_in_4bit: bool = False
    dtype: str = "bfloat16"
    allow_triton: bool = False


class QwenPolicy:
    """Policy facade for kernel generation.

    `generation_mode` controls execution:
    - `stub`: deterministic local fallback for tests/offline.
    - `local`: load model + LoRA weights locally and generate via transformers.
    - `kernelbench_server`: use KernelBench prompt construction + inference server.
    """

    def __init__(self, config: QwenPolicyConfig | None = None) -> None:
        self.config = config or QwenPolicyConfig()
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None and self.config.generation_mode == "local":
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None and self.config.generation_mode == "local":
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        load_kwargs: dict = {"torch_dtype": dtype, "device_map": "auto"}
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading base model: %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, trust_remote_code=True, **load_kwargs
        )

        if self.config.lora_weights_path:
            from peft import PeftModel  # type: ignore

            logger.info("Loading LoRA weights from: %s", self.config.lora_weights_path)
            self._model = PeftModel.from_pretrained(
                self._model, self.config.lora_weights_path
            )

    def generate_candidate(self, task: KernelTask, attempt: int = 0) -> KernelCandidate:
        if self.config.generation_mode == "stub":
            return self._generate_candidate_stub(task, attempt)
        if self.config.generation_mode == "local":
            try:
                return self._generate_candidate_local(task, attempt)
            except Exception as exc:
                logger.warning("Local generation failed: %s; falling back to stub", exc)
                fallback = self._generate_candidate_stub(task, attempt)
                fallback.logs["generation_fallback"] = True
                fallback.logs["generation_error"] = str(exc)
                return fallback
        if self.config.generation_mode == "kernelbench_server":
            try:
                return self._generate_candidate_kernelbench(task, attempt)
            except Exception as exc:
                fallback = self._generate_candidate_stub(task, attempt)
                fallback.logs["generation_fallback"] = True
                fallback.logs["generation_error"] = str(exc)
                return fallback
        raise ValueError(f"Unsupported generation_mode: {self.config.generation_mode}")

    def generate_text(self, prompt: str) -> str:
        """Raw text generation (used by fix-loop and GRPO reward)."""
        if self.config.generation_mode == "stub":
            return f"def kernel(*args):\n    return args[0] if args else None\n"

        model = self.model
        tokenizer = self.tokenizer
        import torch  # type: ignore

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    def extract_code(self, raw: str) -> str:
        """Public wrapper to normalize model generations into Python source."""
        return self._normalize_modelnew_contract(self._extract_code(raw))

    def _generate_candidate_local(self, task: KernelTask, attempt: int = 0) -> KernelCandidate:
        prompt = self._build_kernel_prompt(task)
        raw = self.generate_text(prompt)
        code = self.extract_code(raw)

        compile_ok = bool(code and "TODO" not in code and len(code.strip()) > 10)

        return KernelCandidate(
            task_id=task.task_id,
            code=code,
            compile_ok=compile_ok,
            correct=False,
            runtime_ms=None,
            speedup=None,
            logs={
                "policy": self.config.model_name,
                "attempt": attempt,
                "generation_mode": "local",
                "lora_weights": self.config.lora_weights_path,
            },
        )

    def _build_kernel_prompt(self, task: KernelTask) -> str:
        ref = task.metadata.get("ref_arch_src", task.reference_kernel)
        target = (
            "CUDA/Triton kernel replacement"
            if self.config.allow_triton
            else "CUDA kernel replacement"
        )
        triton_rule = (
            "- Triton is allowed if needed.\n"
            if self.config.allow_triton
            else "- Do not use Triton. Use PyTorch/CUDA only.\n"
        )
        return (
            "You are an expert GPU kernel engineer. "
            "Given the following PyTorch reference implementation, write an optimized "
            f"{target}.\n\n"
            "Requirements:\n"
            "- The replacement must produce identical outputs to the reference.\n"
            "- Optimize for speed on NVIDIA H100 GPU.\n"
            f"{triton_rule}"
            "- Output only valid Python source code (no markdown, no prose).\n"
            "- Must define class ModelNew(nn.Module).\n"
            "- Return a complete, self-contained Python module.\n\n"
            f"Reference implementation:\n```python\n{ref}\n```\n\n"
            "Write the optimized kernel implementation:"
        )

    @staticmethod
    def _extract_code(raw: str) -> str:
        if "```python" in raw:
            parts = raw.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) > 1:
                code = parts[1]
                if code.startswith(("python\n", "Python\n", "py\n")):
                    code = "\n".join(code.split("\n")[1:])
                return code.strip()
        # Try to salvage raw model output by stripping leading prose.
        text = raw.strip()
        starts = [p for p in (text.find("\nimport "), text.find("\nfrom "), text.find("\nclass ")) if p >= 0]
        if starts:
            start = min(starts) + 1
            return text[start:].strip()
        return text

    @staticmethod
    def _normalize_modelnew_contract(code: str) -> str:
        """Best-effort rewrite to satisfy KernelBench evaluator entry-point."""
        if not code:
            return code
        if "class ModelNew" in code:
            return code
        if re.search(r"^\s*class\s+Model\s*\(", code, re.MULTILINE):
            rewritten = re.sub(
                r"(^\s*class\s+)Model(\s*\()",
                r"\1ModelNew\2",
                code,
                count=1,
                flags=re.MULTILINE,
            )
            rewritten = rewritten.replace("super(Model, self)", "super(ModelNew, self)")
            return rewritten
        return code

    def _generate_candidate_stub(self, task: KernelTask, attempt: int = 0) -> KernelCandidate:
        compile_ok = attempt % 5 != 0
        correct = compile_ok and attempt % 3 != 0

        code = (
            "def kernel(*args, **kwargs):\n"
            "    # generated by qwen policy stub\n"
            f"    # task={task.task_id} attempt={attempt}\n"
        )
        if not compile_ok:
            code += "    TODO\n"
        elif correct:
            code += "    return args[0] if args else None\n"
        else:
            code += "    return None\n"

        runtime_ms = 75.0 + float((attempt % 7) * 5)
        speedup = max(0.2, 100.0 / runtime_ms)

        return KernelCandidate(
            task_id=task.task_id,
            code=code,
            compile_ok=compile_ok,
            correct=correct,
            runtime_ms=runtime_ms,
            speedup=speedup,
            logs={"policy": self.config.model_name, "attempt": attempt},
        )

    def _generate_candidate_kernelbench(self, task: KernelTask, attempt: int = 0) -> KernelCandidate:
        self._ensure_kernelbench_on_path()

        from kernelbench.prompt_constructor_toml import (  # type: ignore
            get_custom_prompt,
            get_prompt_for_backend,
        )
        from kernelbench.utils import (  # type: ignore
            create_inference_server_from_presets,
            extract_first_code,
        )

        reference_src = (
            str(task.metadata.get("ref_arch_src"))
            if task.metadata.get("ref_arch_src")
            else task.reference_kernel
        )
        if not reference_src:
            reference_src = task.prompt

        if self.config.custom_prompt_key:
            prompt = get_custom_prompt(
                self.config.custom_prompt_key,
                ref_arch_src=reference_src,
                backend=self.config.backend,
                option=self.config.prompt_option,
                precision=self.config.precision,
                include_hardware=self.config.include_hardware_info,
                gpu_name=self.config.hardware_gpu_name,
            )
        else:
            prompt = get_prompt_for_backend(
                reference_src,
                self.config.backend,
                option=self.config.prompt_option,
                precision=self.config.precision,
                include_hardware=self.config.include_hardware_info,
                gpu_name=self.config.hardware_gpu_name,
            )

        inference_server = create_inference_server_from_presets(
            server_type=self.config.server_type,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
            is_reasoning_model=self.config.is_reasoning_model,
            reasoning_effort=self.config.reasoning_effort,
            budget_tokens=self.config.budget_tokens,
        )

        raw = inference_server(prompt)
        code = extract_first_code(raw, ["python", "cpp"]) or ""

        compile_ok = bool(code and "TODO" not in code)
        if self.config.check_kernel and compile_ok:
            from kernelbench.kernel_static_checker import validate_kernel_static  # type: ignore

            static_ok, errors, warnings = validate_kernel_static(
                code,
                backend=self.config.backend,
                precision=self.config.precision,
            )
            compile_ok = bool(static_ok)
        else:
            errors = []
            warnings = []

        return KernelCandidate(
            task_id=task.task_id,
            code=code,
            compile_ok=compile_ok,
            correct=False,
            runtime_ms=None,
            speedup=None,
            logs={
                "policy": self.config.model_name,
                "attempt": attempt,
                "generation_mode": self.config.generation_mode,
                "server_type": self.config.server_type,
                "backend": self.config.backend,
                "static_errors": errors,
                "static_warnings": warnings,
            },
        )

    def _ensure_kernelbench_on_path(self) -> None:
        src_dir = self.config.kernelbench_root / "src"
        if not src_dir.exists():
            raise FileNotFoundError(
                f"KernelBench src path not found: {src_dir}. "
                "Set QwenPolicyConfig.kernelbench_root correctly."
            )
        src_dir_str = str(src_dir.resolve())
        if src_dir_str not in sys.path:
            sys.path.insert(0, src_dir_str)
