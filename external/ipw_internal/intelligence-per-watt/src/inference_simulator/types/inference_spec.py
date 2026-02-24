"""Inference configuration specification for Pipeline #2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class InferenceEngine(str, Enum):
    """Supported inference serving engines."""

    VLLM = "vllm"
    TGI = "tgi"  # HuggingFace Text Generation Inference
    TENSORRT_LLM = "tensorrt_llm"  # NVIDIA TensorRT-LLM
    SGLANG = "sglang"
    LLAMA_CPP = "llama_cpp"
    DEEPSPEED_MII = "deepspeed_mii"


@dataclass(frozen=True)
class InferenceSpec:
    """Specification for an inference serving configuration.

    Attributes:
        engine: Inference serving engine (vLLM, TGI, TensorRT-LLM, etc.).
        num_gpus: Total number of GPUs used.
        tensor_parallel: Tensor parallelism degree.
        pipeline_parallel: Pipeline parallelism degree.
        precision: Numeric precision for model weights/compute.
        max_batch_size: Maximum concurrent requests in a batch.
        max_seq_len: Maximum sequence length (input + output).
        engine_config: Engine-specific configuration parameters.
        metadata: Additional user-defined parameters.

    Engine-specific config examples (stored in engine_config):
        vLLM:
            - gpu_memory_utilization: float (0.0-1.0, default 0.9)
            - enforce_eager: bool (disable CUDA graphs)
            - enable_chunked_prefill: bool
            - max_num_seqs: int (max concurrent sequences)
            - block_size: int (KV cache block size, 16 or 32)
            - swap_space_gb: float (CPU swap space for KV cache)
            - quantization: str ("awq", "gptq", "squeezellm", etc.)
            - speculative_model: str (draft model for spec decode)
            - num_speculative_tokens: int
            - enable_prefix_caching: bool
        TGI:
            - quantize: str ("bitsandbytes", "gptq", "awq")
            - max_concurrent_requests: int
            - max_input_length: int
            - max_total_tokens: int
        TensorRT-LLM:
            - use_inflight_batching: bool
            - paged_kv_cache: bool
            - kv_cache_free_gpu_mem_fraction: float
            - max_beam_width: int
        SGLang:
            - chunked_prefill_size: int
            - schedule_policy: str ("lpm", "random", "fcfs")
            - mem_fraction_static: float
    """

    engine: InferenceEngine = InferenceEngine.VLLM
    num_gpus: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    precision: str = "fp16"
    max_batch_size: int = 64
    max_seq_len: int = 131072
    engine_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_vllm(
        cls,
        num_gpus: int = 1,
        tensor_parallel: int = 1,
        precision: str = "fp16",
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 256,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = False,
        **kwargs: Any,
    ) -> InferenceSpec:
        """Create a vLLM inference configuration."""
        engine_config = {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enable_prefix_caching": enable_prefix_caching,
            **kwargs,
        }
        return cls(
            engine=InferenceEngine.VLLM,
            num_gpus=num_gpus,
            tensor_parallel=tensor_parallel,
            precision=precision,
            engine_config=engine_config,
        )

    @classmethod
    def for_tensorrt_llm(
        cls,
        num_gpus: int = 1,
        tensor_parallel: int = 1,
        precision: str = "fp16",
        use_inflight_batching: bool = True,
        paged_kv_cache: bool = True,
        kv_cache_free_gpu_mem_fraction: float = 0.9,
        **kwargs: Any,
    ) -> InferenceSpec:
        """Create a TensorRT-LLM inference configuration."""
        engine_config = {
            "use_inflight_batching": use_inflight_batching,
            "paged_kv_cache": paged_kv_cache,
            "kv_cache_free_gpu_mem_fraction": kv_cache_free_gpu_mem_fraction,
            **kwargs,
        }
        return cls(
            engine=InferenceEngine.TENSORRT_LLM,
            num_gpus=num_gpus,
            tensor_parallel=tensor_parallel,
            precision=precision,
            engine_config=engine_config,
        )

    @classmethod
    def for_sglang(
        cls,
        num_gpus: int = 1,
        tensor_parallel: int = 1,
        precision: str = "fp16",
        chunked_prefill_size: int = 8192,
        schedule_policy: str = "lpm",
        mem_fraction_static: float = 0.88,
        **kwargs: Any,
    ) -> InferenceSpec:
        """Create an SGLang inference configuration."""
        engine_config = {
            "chunked_prefill_size": chunked_prefill_size,
            "schedule_policy": schedule_policy,
            "mem_fraction_static": mem_fraction_static,
            **kwargs,
        }
        return cls(
            engine=InferenceEngine.SGLANG,
            num_gpus=num_gpus,
            tensor_parallel=tensor_parallel,
            precision=precision,
            engine_config=engine_config,
        )
