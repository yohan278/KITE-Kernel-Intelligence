"""Grid configuration dataclasses and enums for evaluation.

Grid Search Loop Order (outermost to innermost):
    1. GpuType (hardware choice) - e.g., A100, H100, MI300X
    2. ResourceConfig (hardware config) - e.g., 1gpu_8cpu, 4gpu_32cpu
    3. AgentType (agent harness) - e.g., react, openhands
    4. ModelType (LMs) - e.g., qwen3-8b, gpt-oss-20b
    5. BenchmarkType (benchmarks) - e.g., gaia, hle [INNERMOST]

This ordering ensures:
    - Hardware changes least frequently (expensive to switch machines)
    - Resource configs change next (may require server restart)
    - Agents change next (lightweight reconfiguration)
    - Models change next (requires vLLM server restart)
    - Benchmarks change most frequently (just loading different data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Dict, Iterator, List, Optional, Tuple


class BenchmarkType(str, Enum):
    """Supported benchmark types (innermost loop - changes most frequently)."""

    GAIA = "gaia"
    HLE = "hle"
    SWEBENCH = "swebench"
    APEX = "apex"
    BROWSECOMP = "browsecomp"
    DEEPRESEARCH = "deepresearch"
    SIMPLEQA = "simpleqa"
    SWEFFICIENCY = "swefficiency"


class ModelType(str, Enum):
    """Supported model types (4th loop - LMs)."""

    # Qwen3 dense models (instruct-tuned)
    QWEN3_0_6B = "qwen3-0.6b"
    QWEN3_1_7B = "qwen3-1.7b"
    QWEN3_4B = "qwen3-4b"
    QWEN3_8B = "qwen3-8b"
    QWEN3_14B = "qwen3-14b"
    QWEN3_32B = "qwen3-32b"
    # Qwen3 MoE models
    QWEN3_30B_A3B = "qwen3-30b-a3b"
    QWEN3_235B_A22B_FP8 = "qwen3-235b-a22b-fp8"
    # Other models
    GPT_OSS_20B = "gpt-oss-20b"
    GPT_OSS_120B = "gpt-oss-120b"
    GLM_4_7_FLASH = "glm-4.7-flash"
    MOONLIGHT_16B_A3B = "moonlight-16b-a3b"
    MINIMAX_M2_5 = "minimax-m2.5"
    KIMI_K2_5 = "kimi-k2.5"
    KIMI_K2_5_GGUF = "kimi-k2.5-gguf"
    QWEN3_5_397B_A17B = "qwen3.5-397b-a17b"
    GLM_4_7_FP8 = "glm-4.7-fp8"
    TRINITY_MINI = "trinity-mini"


class AgentType(str, Enum):
    """Supported agent types (3rd loop - agent harnesses)."""

    REACT = "react"
    OPENHANDS = "openhands"
    ORCHESTRATOR = "orchestrator"


class ResourceConfig(str, Enum):
    """Resource allocation configurations (2nd loop - GPU count + CPU count).

    These define how many GPUs and CPUs to allocate, independent of GPU type.
    """

    ONE_GPU_8CPU = "1gpu_8cpu"       # 1 GPU, 8 CPU threads
    ONE_GPU_16CPU = "1gpu_16cpu"     # 1 GPU, 16 CPU threads
    TWO_GPU_16CPU = "2gpu_16cpu"     # 2 GPUs, 16 CPU threads
    FOUR_GPU_32CPU = "4gpu_32cpu"    # 4 GPUs, 32 CPU threads
    EIGHT_GPU_64CPU = "8gpu_64cpu"   # 8 GPUs, 64 CPU threads
    EIGHT_GPU_240CPU = "8gpu_240cpu" # 8 GPUs, 240 CPU threads


class GpuType(str, Enum):
    """GPU hardware types (outermost loop - hardware choice).

    These define the physical GPU hardware, independent of resource allocation.
    """

    # NVIDIA GPUs
    A100_80GB = "a100_80gb"         # NVIDIA A100 80GB
    H100_80GB = "h100_80gb"         # NVIDIA H100 80GB
    H200 = "h200"                   # NVIDIA H200 141GB
    GH200 = "gh200"                 # NVIDIA GH200 (96GB GPU + 480GB unified)
    B200 = "b200"                   # NVIDIA B200 192GB
    # AMD GPUs
    MI300X = "mi300x"               # AMD Instinct MI300X 192GB
    # Apple Silicon
    M4_MAX = "m4_max"               # Apple M4 Max (up to 40 GPU cores)
    M4_PRO = "m4_pro"               # Apple M4 Pro (up to 20 GPU cores)
    M3_MAX = "m3_max"               # Apple M3 Max (up to 40 GPU cores)
    M3_PRO = "m3_pro"               # Apple M3 Pro (up to 18 GPU cores)


# Legacy HardwareConfig for backwards compatibility
class HardwareConfig(str, Enum):
    """Hardware configuration presets from IPW paper (arxiv:2511.07885).

    DEPRECATED: Use GpuType + ResourceConfig instead for new code.
    These combine GPU type and resource allocation for convenience.
    """

    # Currently available
    A100_1GPU = "a100_1gpu"      # 1× A100 80GB, 8 CPU
    A100_4GPU = "a100_4gpu"      # 4× A100 80GB, 32 CPU
    # Future expansion
    H100_1GPU = "h100_1gpu"      # 1× H100 80GB, 8 CPU
    H100_4GPU = "h100_4gpu"      # 4× H100 80GB, 32 CPU
    GH200_1GPU = "gh200_1gpu"    # 1× GH200 96GB+480GB, 72 CPU
    B200_1GPU = "b200_1gpu"      # 1× B200 192GB, 8 CPU


# Model registry mapping model types to their configuration
MODEL_REGISTRY: Dict[ModelType, Dict[str, Any]] = {
    # Qwen3 dense models (instruct-tuned, no -Instruct suffix needed)
    ModelType.QWEN3_0_6B: {
        "model_id": "Qwen/Qwen3-0.6B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 0.6,
        "active_params_b": 0.6,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 64,
    },
    ModelType.QWEN3_1_7B: {
        "model_id": "Qwen/Qwen3-1.7B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 1.7,
        "active_params_b": 1.7,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 64,
    },
    ModelType.QWEN3_4B: {
        "model_id": "Qwen/Qwen3-4B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 4.0,
        "active_params_b": 4.0,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 64,
    },
    ModelType.QWEN3_8B: {
        "model_id": "Qwen/Qwen3-8B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 8.0,
        "active_params_b": 8.0,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 64,
    },
    ModelType.QWEN3_14B: {
        "model_id": "Qwen/Qwen3-14B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 14.0,
        "active_params_b": 14.0,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 64,
    },
    ModelType.QWEN3_32B: {
        "model_id": "Qwen/Qwen3-32B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 32.0,
        "active_params_b": 32.0,
        "quantization": None,  # H100 (cc 9.0) supports FP8; bf16 fallback for Ampere
        "min_gpus": 1,
        "enforce_eager": True,  # Free memory from torch.compile/CUDA graphs
        "gpu_memory_utilization": 0.95,  # bf16 weights ~64 GiB; need high util for KV cache
        "max_batch_size": 32,
    },
    # Qwen3 MoE models
    ModelType.QWEN3_30B_A3B: {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 30.0,
        "active_params_b": 3.0,
        "is_moe": True,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 16,
        "tool_call_parser": "hermes",  # Qwen3 uses <tool_call> tags (Hermes-compatible)
    },
    ModelType.QWEN3_235B_A22B_FP8: {
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 235.0,
        "active_params_b": 22.0,
        "is_moe": True,
        "quantization": None,  # Already FP8 native
        "min_gpus": 4,
        "max_batch_size": 16,
    },
    ModelType.QWEN3_5_397B_A17B: {
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 397.0,
        "active_params_b": 17.0,
        "is_moe": True,
        "quantization": None,  # FP8 native; ~400 GB weights fit 8×H100 with room for KV cache
        "min_gpus": 8,
        "max_model_len": 32768,  # Native 262K, but reduce for KV cache headroom at TP=8
        "enforce_eager": True,  # Free memory from torch.compile/CUDA graphs for large MoE
        "gpu_memory_utilization": 0.95,
        "max_batch_size": 16,
        "tool_call_parser": "hermes",  # Hermes-style fallback (no qwen3.5-specific parser yet)
    },
    # Other models
    ModelType.GPT_OSS_20B: {
        "model_id": "openai/gpt-oss-20b",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 20.0,
        "active_params_b": 2.4,
        "is_moe": True,
        "quantization": None,  # Model ships with mxfp4; let vLLM use native quant
        "min_gpus": 1,
        "max_batch_size": 64,
    },
    ModelType.GPT_OSS_120B: {
        "model_id": "openai/gpt-oss-120b",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 120.0,
        "active_params_b": 5.6,
        "is_moe": True,
        "quantization": None,  # Model ships with mxfp4; let vLLM use native quant
        "min_gpus": 1,
        "max_model_len": 32768,  # Reduced from 131072 to fit batch KV cache on 8×H100
        "enforce_eager": True,  # Skip torch.compile/CUDA graphs to free memory
        "gpu_memory_utilization": 0.95,  # Model needs >90% of 80 GB; raise cap
        "max_batch_size": 16,
        "tool_call_parser": "seed_oss",  # vLLM native tool call parser for GPT-OSS
    },
    ModelType.GLM_4_7_FLASH: {
        "model_id": "zai-org/GLM-4.7-Flash",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 31.0,
        "active_params_b": 3.0,
        "is_moe": True,
        "quantization": None,
        "min_gpus": 1,  # MoE with 3B active params; TP=1 is valid (no parallelism needed)
        "max_tp": 4,    # Valid TP sizes: 1, 2, 4 (must divide hidden_size=2048, heads=20, intermediate=10240)
        "max_model_len": 32768,  # Reduced from 202752 to fit batch KV cache on 4×H100
        "trust_remote_code": True,
        "max_batch_size": 16,
        "tool_call_parser": "glm47",  # vLLM native tool call parser for GLM-4.7
    },
    ModelType.MOONLIGHT_16B_A3B: {
        "model_id": "moonshotai/Moonlight-16B-A3B-Instruct",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 16.0,
        "active_params_b": 3.0,
        "is_moe": True,
        "quantization": None,
        "min_gpus": 1,
        "max_model_len": 8192,  # Model max_position_embeddings=8192
        "trust_remote_code": True,
        "max_batch_size": 16,
        "tool_call_parser": "hermes",  # Hermes-style fallback (no moonlight-specific parser)
    },
    ModelType.MINIMAX_M2_5: {
        "model_id": "MiniMaxAI/MiniMax-M2.5",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 229.0,
        "active_params_b": 10.0,
        "is_moe": True,
        "quantization": None,  # FP8 native
        "min_gpus": 4,
        "max_model_len": 32768,  # Reduced from 196608 to fit in 8×H100 with EP
        "trust_remote_code": True,
        "enable_expert_parallel": True,  # Required for MoE expert distribution across GPUs
        "enforce_eager": True,  # Per troubleshooting guide for large MoE
        "gpu_memory_utilization": 0.95,
        "env_vars": {"SAFETENSORS_FAST_GPU": "1"},
        "max_batch_size": 16,
        "tool_call_parser": "minimax_m2",  # vLLM native tool call parser for MiniMax M2
    },
    ModelType.KIMI_K2_5: {
        "model_id": "moonshotai/Kimi-K2.5",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 1000.0,
        "active_params_b": 32.0,
        "is_moe": True,
        "quantization": None,  # Native INT4
        "min_gpus": 8,
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
        "max_batch_size": 9,
    },
    ModelType.KIMI_K2_5_GGUF: {
        "model_id": "/home/ubuntu/.cache/huggingface/hub/models--unsloth--Kimi-K2.5-GGUF/snapshots/386fed8b054275941d6a495a9a7010fbf31b560d/Q3_K_S/Kimi-K2.5-Q3_K_S-00001-of-00010.gguf",
        "tokenizer": "moonshotai/Kimi-K2.5",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 1000.0,
        "active_params_b": 32.0,
        "is_moe": True,
        "quantization": "gguf",  # Q3_K_S variant (~443GB)
        "min_gpus": 8,
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 32768,  # Increased from 8192; model supports 262K native
        "max_batch_size": 9,
    },
    ModelType.GLM_4_7_FP8: {
        "model_id": "zai-org/GLM-4.7-FP8",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 353.0,
        "active_params_b": 40.0,
        "is_moe": True,
        "quantization": None,  # FP8 via compressed-tensors (auto-detected)
        "min_gpus": 8,  # 362 GB FP8 weights; needs TP=8 on 80GB GPUs
        "max_model_len": 32768,  # Conservative; native 202752 but KV cache limited at TP=8
        "max_batch_size": 16,
        "tool_call_parser": "glm47",  # vLLM native tool call parser for GLM-4.7
    },
    ModelType.TRINITY_MINI: {
        "model_id": "arcee-ai/Trinity-Mini",
        "type": "vllm",
        "requires_server": True,
        "total_params_b": 3.5,
        "active_params_b": 3.5,
        "quantization": "fp8",
        "min_gpus": 1,
        "max_batch_size": 64,
    },
}

# =============================================================================
# Ollama Model Mapping - Maps ModelType to Ollama model names
# =============================================================================
OLLAMA_MODEL_MAPPING: Dict[ModelType, str] = {
    ModelType.QWEN3_0_6B: "qwen2.5:0.5b",
    ModelType.QWEN3_1_7B: "qwen2.5:1.5b",
    ModelType.QWEN3_4B: "qwen2.5:3b",
    ModelType.QWEN3_8B: "qwen2.5:7b",
    ModelType.QWEN3_14B: "qwen2.5:14b",
    ModelType.QWEN3_32B: "qwen2.5:32b",
}

# =============================================================================
# GPU Type Registry - Hardware specifications per GPU type
# =============================================================================
GPU_TYPE_REGISTRY: Dict[GpuType, Dict[str, Any]] = {
    GpuType.A100_80GB: {
        "vendor": "nvidia",
        "memory_gb": 80,
        "tdp_watts": 400,
        "architecture": "ampere",
        "fp8_support": True,
    },
    GpuType.H100_80GB: {
        "vendor": "nvidia",
        "memory_gb": 80,
        "tdp_watts": 700,
        "architecture": "hopper",
        "fp8_support": True,
    },
    GpuType.H200: {
        "vendor": "nvidia",
        "memory_gb": 141,
        "tdp_watts": 700,
        "architecture": "hopper",
        "fp8_support": True,
    },
    GpuType.GH200: {
        "vendor": "nvidia",
        "memory_gb": 96,  # GPU memory (+ 480GB unified CPU memory)
        "unified_memory_gb": 480,
        "tdp_watts": 900,  # Full superchip
        "architecture": "hopper",
        "fp8_support": True,
    },
    GpuType.B200: {
        "vendor": "nvidia",
        "memory_gb": 192,
        "tdp_watts": 1000,
        "architecture": "blackwell",
        "fp8_support": True,
        "fp4_support": True,
    },
    GpuType.MI300X: {
        "vendor": "amd",
        "memory_gb": 192,
        "tdp_watts": 750,
        "architecture": "cdna3",
        "fp8_support": True,
    },
    # Apple Silicon
    GpuType.M4_MAX: {
        "vendor": "apple",
        "memory_gb": 128,  # Max unified memory config
        "tdp_watts": 40,   # Approximate
        "architecture": "apple_silicon",
        "unified_memory": True,
        "default_backend": "ollama",
    },
    GpuType.M4_PRO: {
        "vendor": "apple",
        "memory_gb": 64,
        "tdp_watts": 30,
        "architecture": "apple_silicon",
        "unified_memory": True,
        "default_backend": "ollama",
    },
    GpuType.M3_MAX: {
        "vendor": "apple",
        "memory_gb": 128,  # Max unified memory config
        "tdp_watts": 40,   # Approximate
        "architecture": "apple_silicon",
        "unified_memory": True,
        "default_backend": "ollama",
    },
    GpuType.M3_PRO: {
        "vendor": "apple",
        "memory_gb": 36,
        "tdp_watts": 30,
        "architecture": "apple_silicon",
        "unified_memory": True,
        "default_backend": "ollama",
    },
}


# =============================================================================
# Resource Config Registry - Environment settings per resource allocation
# =============================================================================
RESOURCE_CONFIG_REGISTRY: Dict[ResourceConfig, Dict[str, Any]] = {
    ResourceConfig.ONE_GPU_8CPU: {
        "gpu_count": 1,
        "cpu_count": 8,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    },
    ResourceConfig.ONE_GPU_16CPU: {
        "gpu_count": 1,
        "cpu_count": 16,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
    },
    ResourceConfig.TWO_GPU_16CPU: {
        "gpu_count": 2,
        "cpu_count": 16,
        "CUDA_VISIBLE_DEVICES": "4,5",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
    },
    ResourceConfig.FOUR_GPU_32CPU: {
        "gpu_count": 4,
        "cpu_count": 32,
        "CUDA_VISIBLE_DEVICES": "0,2,4,5",
        "OMP_NUM_THREADS": "32",
        "MKL_NUM_THREADS": "32",
    },
    ResourceConfig.EIGHT_GPU_64CPU: {
        "gpu_count": 8,
        "cpu_count": 64,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "OMP_NUM_THREADS": "64",
        "MKL_NUM_THREADS": "64",
    },
    ResourceConfig.EIGHT_GPU_240CPU: {
        "gpu_count": 8,
        "cpu_count": 240,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "OMP_NUM_THREADS": "240",
        "MKL_NUM_THREADS": "240",
    },
}


# =============================================================================
# Legacy Hardware Registry (for backwards compatibility)
# =============================================================================
HARDWARE_REGISTRY: Dict[HardwareConfig, Dict[str, Any]] = {
    HardwareConfig.A100_1GPU: {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "gpu_count": 1,
        "gpu_type": "A100",
        "gpu_memory_gb": 80,
    },
    HardwareConfig.A100_4GPU: {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "32",
        "MKL_NUM_THREADS": "32",
        "gpu_count": 4,
        "gpu_type": "A100",
        "gpu_memory_gb": 320,  # Total across 4 GPUs
    },
    HardwareConfig.H100_1GPU: {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "gpu_count": 1,
        "gpu_type": "H100",
        "gpu_memory_gb": 80,
    },
    HardwareConfig.H100_4GPU: {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "32",
        "MKL_NUM_THREADS": "32",
        "gpu_count": 4,
        "gpu_type": "H100",
        "gpu_memory_gb": 320,
    },
    HardwareConfig.GH200_1GPU: {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "72",
        "MKL_NUM_THREADS": "72",
        "gpu_count": 1,
        "gpu_type": "GH200",
        "gpu_memory_gb": 576,  # 96GB GPU + 480GB unified
    },
    HardwareConfig.B200_1GPU: {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "gpu_count": 1,
        "gpu_type": "B200",
        "gpu_memory_gb": 192,
    },
}


# Phase 1A specific hardware configs (GPUs 4-7)
PHASE1A_HARDWARE: Dict[str, Dict[str, Any]] = {
    "1gpu": {
        "CUDA_VISIBLE_DEVICES": "4",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "gpu_count": 1,
    },
    "2gpu": {
        "CUDA_VISIBLE_DEVICES": "4,5",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "gpu_count": 2,
    },
}


@dataclass
class GridConfig:
    """Configuration for grid search evaluation.

    Grid Search Loop Order (outermost to innermost):
        1. gpu_types (hardware choice) - changes least frequently
        2. resource_configs (hardware config) - GPU/CPU allocation
        3. agents (agent harnesses)
        4. models (LMs)
        5. benchmarks - changes most frequently (innermost)

    This ordering minimizes expensive operations:
        - Hardware choice rarely changes (different machines)
        - Resource configs may require server restart
        - Agent changes are lightweight
        - Model changes require vLLM restart
        - Benchmark changes just load different data

    Attributes:
        gpu_types: List of GPU hardware types to test
        resource_configs: List of resource allocations (GPU count + CPU count)
        agents: List of agent types to use
        models: List of models to test
        benchmarks: List of benchmarks to evaluate
        queries_per_benchmark: Number of queries per benchmark
        seed: Random seed for reproducibility
        gaia_level: Optional GAIA level filter (1, 2, or 3). If None, uses all levels.
        use_full_datasets: If True, do not apply sample limit (use entire dataset)
        hle_text_only: If True, filter HLE to text-only samples (no images/audio)
        hardware_configs: DEPRECATED - use gpu_types + resource_configs instead
    """

    # Loop order: outermost (1) to innermost (5)
    # 1. Hardware choice (outermost - changes least frequently)
    gpu_types: List[GpuType] = field(
        default_factory=lambda: [GpuType.A100_80GB]
    )
    # 2. Resource allocation (GPU count + CPU count)
    resource_configs: List[ResourceConfig] = field(
        default_factory=lambda: [
            ResourceConfig.ONE_GPU_8CPU,
            ResourceConfig.FOUR_GPU_32CPU,
        ]
    )
    # 3. Agent harnesses
    agents: List[AgentType] = field(
        default_factory=lambda: [AgentType.REACT, AgentType.OPENHANDS]
    )
    # 4. Language models
    models: List[ModelType] = field(
        default_factory=lambda: [ModelType.QWEN3_8B, ModelType.GPT_OSS_20B]
    )
    # 5. Benchmarks (innermost - changes most frequently)
    benchmarks: List[BenchmarkType] = field(
        default_factory=lambda: [
            BenchmarkType.GAIA,
            BenchmarkType.HLE,
            BenchmarkType.SWEBENCH,
            BenchmarkType.APEX,
            BenchmarkType.BROWSECOMP,
            BenchmarkType.DEEPRESEARCH,
            BenchmarkType.SIMPLEQA,
            BenchmarkType.SWEFFICIENCY,
        ]
    )

    # Other configuration
    queries_per_benchmark: int = 100
    seed: int = 42
    gaia_level: Optional[int] = None
    use_full_datasets: bool = True
    hle_text_only: bool = True

    # LLM Judge configuration
    grader_model: str = "gpt-5-mini-2025-08-07"
    grader_api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY env var

    # Legacy field for backwards compatibility
    hardware_configs: List[HardwareConfig] = field(
        default_factory=lambda: [
            HardwareConfig.A100_1GPU,
            HardwareConfig.A100_4GPU,
        ]
    )

    # LLM Judge configuration for scoring
    grader_model: str = "gpt-5-mini-2025-08-07"  # Cost-effective model for grading
    grader_api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY env var
    use_exact_match: bool = False  # If True, bypass LLM judge and use exact match

    # Tool configuration
    include_cloud_tools: bool = False  # Include cloud LLM tools (gpt-4o, claude, etc.) in agent toolset

    # Parallelization
    num_workers: int = 1  # Number of parallel workers (1 = sequential)
    batch_mode: bool = False  # Batched parallel eval with per-model batch sizes and amortized energy

    @classmethod
    def from_hydra(cls, cfg) -> "GridConfig":
        """Create a GridConfig from a resolved Hydra DictConfig.

        Maps Hydra config names to Python enum values and wraps single
        selections in lists (since Hydra composes a single run, not a grid).

        Args:
            cfg: Resolved Hydra DictConfig matching the RunConfig schema.

        Returns:
            GridConfig with single-element lists for each dimension.

        Raises:
            ValueError: If a config name doesn't match any known enum value.
        """
        # Map model name → ModelType
        model_name = cfg.model.name
        try:
            model = ModelType(model_name)
        except ValueError:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Valid: {[m.value for m in ModelType]}"
            )

        # Map agent name → AgentType
        agent_name = cfg.agent.name
        try:
            agent = AgentType(agent_name)
        except ValueError:
            raise ValueError(
                f"Unknown agent '{agent_name}'. "
                f"Valid: {[a.value for a in AgentType]}"
            )

        # Map benchmark name → BenchmarkType
        benchmark_name = cfg.benchmark.name
        try:
            benchmark = BenchmarkType(benchmark_name)
        except ValueError:
            raise ValueError(
                f"Unknown benchmark '{benchmark_name}'. "
                f"Valid: {[b.value for b in BenchmarkType]}"
            )

        # Map hardware gpu type → GpuType
        gpu_type_str = cfg.hardware.gpu.type
        try:
            gpu_type = GpuType(gpu_type_str)
        except ValueError:
            raise ValueError(
                f"Unknown GPU type '{gpu_type_str}'. "
                f"Valid: {[g.value for g in GpuType]}"
            )

        # Map hardware resources → ResourceConfig by constructing the enum value
        gpu_count = cfg.hardware.resources.gpu_count
        cpu_count = cfg.hardware.resources.cpu_count
        resource_str = f"{gpu_count}gpu_{cpu_count}cpu"
        try:
            resource_config = ResourceConfig(resource_str)
        except ValueError:
            raise ValueError(
                f"Unknown resource config '{resource_str}' "
                f"(from gpu_count={gpu_count}, cpu_count={cpu_count}). "
                f"Valid: {[r.value for r in ResourceConfig]}"
            )

        return cls(
            gpu_types=[gpu_type],
            resource_configs=[resource_config],
            agents=[agent],
            models=[model],
            benchmarks=[benchmark],
            queries_per_benchmark=cfg.eval.queries_per_benchmark,
            seed=cfg.eval.seed,
            use_full_datasets=cfg.eval.use_full_datasets,
            hle_text_only=cfg.eval.hle_text_only,
            gaia_level=cfg.eval.gaia_level,
        )

    def get_all_combinations(
        self,
    ) -> Iterator[Tuple[GpuType, ResourceConfig, AgentType, ModelType, BenchmarkType]]:
        """Generate all configuration combinations in proper loop order.

        Loop order (outermost to innermost):
            1. gpu_type (hardware choice)
            2. resource_config (GPU/CPU allocation)
            3. agent (agent harness)
            4. model (LM)
            5. benchmark (innermost)

        Yields:
            Tuples of (gpu_type, resource_config, agent, model, benchmark)
        """
        # product() iterates rightmost dimension fastest (innermost)
        # So we list dimensions from outermost to innermost
        for gpu_type, resource_config, agent, model, benchmark in product(
            self.gpu_types,           # 1. outermost
            self.resource_configs,    # 2.
            self.agents,              # 3.
            self.models,              # 4.
            self.benchmarks,          # 5. innermost
        ):
            yield gpu_type, resource_config, agent, model, benchmark

    def get_all_combinations_legacy(
        self,
    ) -> Iterator[Tuple[BenchmarkType, ModelType, AgentType, HardwareConfig]]:
        """Generate combinations using legacy HardwareConfig (for backwards compatibility).

        DEPRECATED: Use get_all_combinations() instead.

        Yields:
            Tuples of (benchmark, model, agent, hardware) for each combination
        """
        for benchmark, model, agent, hardware in product(
            self.benchmarks,
            self.models,
            self.agents,
            self.hardware_configs,
        ):
            yield benchmark, model, agent, hardware

    def total_combinations(self) -> int:
        """Return total number of configuration combinations."""
        return (
            len(self.gpu_types)
            * len(self.resource_configs)
            * len(self.agents)
            * len(self.models)
            * len(self.benchmarks)
        )

    def total_queries(self) -> int:
        """Return total number of queries to run."""
        return self.total_combinations() * self.queries_per_benchmark

    def describe(self) -> str:
        """Return human-readable description of the grid."""
        lines = [
            "Grid Search Configuration:",
            "",
            "Loop order (outermost → innermost):",
            f"  1. GPU Types: {[g.value for g in self.gpu_types]}",
            f"  2. Resource Configs: {[r.value for r in self.resource_configs]}",
            f"  3. Agents: {[a.value for a in self.agents]}",
            f"  4. Models: {[m.value for m in self.models]}",
            f"  5. Benchmarks: {[b.value for b in self.benchmarks]}",
            "",
            "Settings:",
            f"  Queries per benchmark: {self.queries_per_benchmark}",
            f"  Seed: {self.seed}",
            f"  Use full datasets: {self.use_full_datasets}",
            f"  HLE text only: {self.hle_text_only}",
        ]
        if self.gaia_level is not None:
            lines.append(f"  GAIA level filter: {self.gaia_level}")
        lines.extend(
            [
                "",
                "Grader:",
                f"  Model: {self.grader_model}",
                f"  Use exact match: {self.use_exact_match}",
                "",
                "Execution:",
                f"  Workers: {self.num_workers}" + (" (parallel)" if self.num_workers > 1 else " (sequential)"),
                f"  Batch mode: {self.batch_mode}",
            ]
        )
        if self.batch_mode:
            lines.append("  Note: Energy is amortized across batch (total GPU energy / batch_size)")
            batch_info = []
            for m in self.models:
                mc = MODEL_REGISTRY.get(m, {})
                bs = mc.get("max_batch_size", 1)
                batch_info.append(f"{m.value}={bs}")
            lines.append(f"  Batch sizes: {', '.join(batch_info)}")
        if self.num_workers > 1:
            lines.append("  Note: Per-query energy attribution disabled in parallel mode")
        lines.extend(
            [
                "",
                "Summary:",
                f"  Total combinations: {self.total_combinations()}",
                f"  Total queries: {self.total_queries()}",
            ]
        )
        return "\n".join(lines)


__all__ = [
    # Enums (in loop order: outermost to innermost)
    "GpuType",           # 1. Hardware choice (outermost)
    "ResourceConfig",    # 2. Resource allocation
    "AgentType",         # 3. Agent harness
    "ModelType",         # 4. Language model
    "BenchmarkType",     # 5. Benchmark (innermost)
    # Legacy enum (deprecated)
    "HardwareConfig",
    # Registries
    "GPU_TYPE_REGISTRY",
    "RESOURCE_CONFIG_REGISTRY",
    "MODEL_REGISTRY",
    "OLLAMA_MODEL_MAPPING",
    "HARDWARE_REGISTRY",  # Legacy
    "PHASE1A_HARDWARE",   # Legacy
    # Config class
    "GridConfig",
]
