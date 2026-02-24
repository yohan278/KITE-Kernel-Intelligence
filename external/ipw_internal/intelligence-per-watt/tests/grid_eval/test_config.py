"""Tests for grid_eval.config module."""

import pytest

from grid_eval.config import (
    AgentType,
    BenchmarkType,
    GridConfig,
    GpuType,
    GPU_TYPE_REGISTRY,
    HARDWARE_REGISTRY,
    HardwareConfig,
    MODEL_REGISTRY,
    ModelType,
    ResourceConfig,
    RESOURCE_CONFIG_REGISTRY,
)


class TestBenchmarkType:
    """Tests for BenchmarkType enum."""

    def test_values(self):
        assert BenchmarkType.HLE.value == "hle"
        assert BenchmarkType.GAIA.value == "gaia"
        assert BenchmarkType.SWEBENCH.value == "swebench"
        assert BenchmarkType.APEX.value == "apex"
        assert BenchmarkType.BROWSECOMP.value == "browsecomp"
        assert BenchmarkType.DEEPRESEARCH.value == "deepresearch"
        assert BenchmarkType.SIMPLEQA.value == "simpleqa"
        assert BenchmarkType.SWEFFICIENCY.value == "swefficiency"

    def test_all_values(self):
        assert len(BenchmarkType) == 8


class TestModelType:
    """Tests for ModelType enum."""

    def test_values(self):
        assert ModelType.QWEN3_8B.value == "qwen3-8b"
        assert ModelType.GPT_OSS_20B.value == "gpt-oss-20b"
        assert ModelType.TRINITY_MINI.value == "trinity-mini"
        assert ModelType.QWEN3_235B_A22B_FP8.value == "qwen3-235b-a22b-fp8"

    def test_all_values(self):
        # 6 Qwen3 dense + 2 Qwen3 MoE + 4 other = 12 models
        assert len(ModelType) == 12


class TestAgentType:
    """Tests for AgentType enum."""

    def test_values(self):
        assert AgentType.REACT.value == "react"
        assert AgentType.OPENHANDS.value == "openhands"

    def test_all_values(self):
        assert len(AgentType) == 3


class TestHardwareConfig:
    """Tests for HardwareConfig enum."""

    def test_values(self):
        assert HardwareConfig.A100_1GPU.value == "a100_1gpu"
        assert HardwareConfig.A100_4GPU.value == "a100_4gpu"
        assert HardwareConfig.H100_1GPU.value == "h100_1gpu"
        assert HardwareConfig.H100_4GPU.value == "h100_4gpu"
        assert HardwareConfig.GH200_1GPU.value == "gh200_1gpu"
        assert HardwareConfig.B200_1GPU.value == "b200_1gpu"

    def test_all_values(self):
        # 2 A100 configs + 2 H100 configs + GH200 + B200 = 6
        assert len(HardwareConfig) == 6


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_qwen3_8b(self):
        config = MODEL_REGISTRY[ModelType.QWEN3_8B]
        assert config["model_id"] == "Qwen/Qwen3-8B"
        assert config["type"] == "vllm"
        assert config["requires_server"] is True
        assert config["total_params_b"] == 8.0
        assert config["active_params_b"] == 8.0
        assert config["quantization"] == "fp8"
        assert config["min_gpus"] == 1

    def test_gpt_oss_20b(self):
        config = MODEL_REGISTRY[ModelType.GPT_OSS_20B]
        assert config["model_id"] == "openai/gpt-oss-20b"
        assert config["type"] == "vllm"
        assert config["requires_server"] is True
        assert config["total_params_b"] == 20.0
        assert config["active_params_b"] == 20.0
        assert config["quantization"] == "fp8"
        assert config["min_gpus"] == 1

    def test_qwen3_235b_a22b_fp8(self):
        config = MODEL_REGISTRY[ModelType.QWEN3_235B_A22B_FP8]
        assert config["model_id"] == "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
        assert config["type"] == "vllm"
        assert config["requires_server"] is True
        assert config["total_params_b"] == 235.0
        assert config["active_params_b"] == 22.0
        assert config["is_moe"] is True
        assert config["quantization"] is None  # Already FP8 native
        assert config["min_gpus"] == 4

    def test_trinity_mini(self):
        config = MODEL_REGISTRY[ModelType.TRINITY_MINI]
        assert config["model_id"] == "arcee-ai/Trinity-Mini"
        assert config["type"] == "vllm"
        assert config["total_params_b"] == 3.5
        assert config["quantization"] == "fp8"
        assert config["min_gpus"] == 1

    def test_qwen3_32b_requires_2_gpus(self):
        config = MODEL_REGISTRY[ModelType.QWEN3_32B]
        assert config["min_gpus"] == 2

    def test_all_models_have_entries(self):
        for model in ModelType:
            assert model in MODEL_REGISTRY

    def test_all_models_have_quantization(self):
        """All models should have quantization field (can be None or 'fp8')."""
        for model in ModelType:
            config = MODEL_REGISTRY[model]
            assert "quantization" in config

    def test_all_models_have_min_gpus(self):
        """All models should have min_gpus field."""
        for model in ModelType:
            config = MODEL_REGISTRY[model]
            assert "min_gpus" in config
            assert config["min_gpus"] >= 1


class TestHardwareRegistry:
    """Tests for HARDWARE_REGISTRY."""

    def test_a100_1gpu(self):
        config = HARDWARE_REGISTRY[HardwareConfig.A100_1GPU]
        assert config["CUDA_VISIBLE_DEVICES"] == "0"
        assert config["OMP_NUM_THREADS"] == "8"
        assert config["MKL_NUM_THREADS"] == "8"
        assert config["gpu_count"] == 1
        assert config["gpu_type"] == "A100"
        assert config["gpu_memory_gb"] == 80

    def test_a100_4gpu(self):
        config = HARDWARE_REGISTRY[HardwareConfig.A100_4GPU]
        assert config["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert config["OMP_NUM_THREADS"] == "32"
        assert config["MKL_NUM_THREADS"] == "32"
        assert config["gpu_count"] == 4
        assert config["gpu_type"] == "A100"
        assert config["gpu_memory_gb"] == 320  # Total across 4 GPUs

    def test_h100_1gpu(self):
        config = HARDWARE_REGISTRY[HardwareConfig.H100_1GPU]
        assert config["CUDA_VISIBLE_DEVICES"] == "0"
        assert config["gpu_count"] == 1
        assert config["gpu_type"] == "H100"
        assert config["gpu_memory_gb"] == 80

    def test_h100_4gpu(self):
        config = HARDWARE_REGISTRY[HardwareConfig.H100_4GPU]
        assert config["gpu_count"] == 4
        assert config["gpu_type"] == "H100"
        assert config["gpu_memory_gb"] == 320

    def test_gh200_1gpu(self):
        config = HARDWARE_REGISTRY[HardwareConfig.GH200_1GPU]
        assert config["gpu_count"] == 1
        assert config["gpu_type"] == "GH200"
        assert config["gpu_memory_gb"] == 576  # 96GB GPU + 480GB unified

    def test_b200_1gpu(self):
        config = HARDWARE_REGISTRY[HardwareConfig.B200_1GPU]
        assert config["gpu_count"] == 1
        assert config["gpu_type"] == "B200"
        assert config["gpu_memory_gb"] == 192

    def test_all_configs_have_entries(self):
        for hw in HardwareConfig:
            assert hw in HARDWARE_REGISTRY

    def test_all_configs_have_gpu_type(self):
        """All hardware configs should have gpu_type field."""
        for hw in HardwareConfig:
            config = HARDWARE_REGISTRY[hw]
            assert "gpu_type" in config

    def test_all_configs_have_gpu_memory_gb(self):
        """All hardware configs should have gpu_memory_gb field."""
        for hw in HardwareConfig:
            config = HARDWARE_REGISTRY[hw]
            assert "gpu_memory_gb" in config
            assert config["gpu_memory_gb"] > 0


class TestGpuType:
    """Tests for GpuType enum."""

    def test_values(self):
        assert GpuType.A100_80GB.value == "a100_80gb"
        assert GpuType.H100_80GB.value == "h100_80gb"
        assert GpuType.MI300X.value == "mi300x"

    def test_all_values(self):
        # A100, H100, H200, GH200, B200, MI300X + 4 Apple Silicon = 10
        assert len(GpuType) == 10


class TestResourceConfig:
    """Tests for ResourceConfig enum."""

    def test_values(self):
        assert ResourceConfig.ONE_GPU_8CPU.value == "1gpu_8cpu"
        assert ResourceConfig.FOUR_GPU_32CPU.value == "4gpu_32cpu"

    def test_all_values(self):
        assert len(ResourceConfig) == 6


class TestGpuTypeRegistry:
    """Tests for GPU_TYPE_REGISTRY."""

    def test_a100_80gb(self):
        config = GPU_TYPE_REGISTRY[GpuType.A100_80GB]
        assert config["vendor"] == "nvidia"
        assert config["memory_gb"] == 80
        assert config["fp8_support"] is True

    def test_mi300x(self):
        config = GPU_TYPE_REGISTRY[GpuType.MI300X]
        assert config["vendor"] == "amd"
        assert config["memory_gb"] == 192
        assert config["fp8_support"] is True

    def test_all_gpu_types_have_entries(self):
        for gpu in GpuType:
            assert gpu in GPU_TYPE_REGISTRY


class TestResourceConfigRegistry:
    """Tests for RESOURCE_CONFIG_REGISTRY."""

    def test_1gpu_8cpu(self):
        config = RESOURCE_CONFIG_REGISTRY[ResourceConfig.ONE_GPU_8CPU]
        assert config["gpu_count"] == 1
        assert config["cpu_count"] == 8
        assert config["CUDA_VISIBLE_DEVICES"] == "0"

    def test_4gpu_32cpu(self):
        config = RESOURCE_CONFIG_REGISTRY[ResourceConfig.FOUR_GPU_32CPU]
        assert config["gpu_count"] == 4
        assert config["cpu_count"] == 32
        assert config["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    def test_all_resource_configs_have_entries(self):
        for rc in ResourceConfig:
            assert rc in RESOURCE_CONFIG_REGISTRY


class TestGridConfig:
    """Tests for GridConfig dataclass."""

    def test_default_values(self):
        config = GridConfig()
        # New API fields
        assert len(config.gpu_types) == 1  # Default: A100_80GB
        assert len(config.resource_configs) == 2  # Default: 1gpu_8cpu, 4gpu_32cpu
        assert len(config.agents) == 2
        assert len(config.models) == 2
        assert len(config.benchmarks) == 8  # All 8 benchmarks
        # Other settings
        assert config.queries_per_benchmark == 100
        assert config.seed == 42
        assert config.use_full_datasets is True
        assert config.hle_text_only is True
        # Legacy field still exists
        assert len(config.hardware_configs) == 2

    def test_default_gpu_types(self):
        """Default gpu_types should be A100."""
        config = GridConfig()
        assert GpuType.A100_80GB in config.gpu_types

    def test_default_resource_configs(self):
        """Default resource_configs should be 1gpu and 4gpu."""
        config = GridConfig()
        assert ResourceConfig.ONE_GPU_8CPU in config.resource_configs
        assert ResourceConfig.FOUR_GPU_32CPU in config.resource_configs

    def test_total_combinations_default(self):
        config = GridConfig()
        # 1 gpu_type * 2 resource_configs * 2 agents * 2 models * 8 benchmarks = 64
        assert config.total_combinations() == 64

    def test_total_queries_default(self):
        config = GridConfig()
        # 64 combinations * 100 queries = 6400
        assert config.total_queries() == 6400

    def test_custom_config(self):
        config = GridConfig(
            gpu_types=[GpuType.A100_80GB],
            resource_configs=[ResourceConfig.ONE_GPU_8CPU],
            agents=[AgentType.REACT, AgentType.OPENHANDS],
            models=[ModelType.QWEN3_8B],
            benchmarks=[BenchmarkType.HLE],
            queries_per_benchmark=50,
            seed=123,
        )
        # 1 * 1 * 2 * 1 * 1 = 2
        assert config.total_combinations() == 2
        assert config.total_queries() == 100  # 2 * 50
        assert config.seed == 123

    def test_get_all_combinations(self):
        """Test that get_all_combinations returns correct tuples."""
        config = GridConfig(
            gpu_types=[GpuType.A100_80GB],
            resource_configs=[ResourceConfig.ONE_GPU_8CPU],
            agents=[AgentType.REACT],
            models=[ModelType.QWEN3_8B],
            benchmarks=[BenchmarkType.HLE],
        )
        combos = list(config.get_all_combinations())
        assert len(combos) == 1
        gpu_type, resource_config, agent, model, benchmark = combos[0]
        assert gpu_type == GpuType.A100_80GB
        assert resource_config == ResourceConfig.ONE_GPU_8CPU
        assert agent == AgentType.REACT
        assert model == ModelType.QWEN3_8B
        assert benchmark == BenchmarkType.HLE

    def test_get_all_combinations_loop_order(self):
        """Test that loop order is correct: gpu_type > resource > agent > model > benchmark.

        Benchmark (innermost) should change fastest.
        GPU type (outermost) should change slowest.
        """
        config = GridConfig(
            gpu_types=[GpuType.A100_80GB, GpuType.H100_80GB],  # 2
            resource_configs=[ResourceConfig.ONE_GPU_8CPU],     # 1
            agents=[AgentType.REACT],                           # 1
            models=[ModelType.QWEN3_8B],                        # 1
            benchmarks=[BenchmarkType.HLE, BenchmarkType.GAIA], # 2
        )
        combos = list(config.get_all_combinations())
        assert len(combos) == 4  # 2 * 1 * 1 * 1 * 2

        # First 2 combos should have A100 (outermost), with HLE then GAIA (innermost)
        assert combos[0][0] == GpuType.A100_80GB
        assert combos[0][4] == BenchmarkType.HLE  # benchmark is innermost (index 4)
        assert combos[1][0] == GpuType.A100_80GB
        assert combos[1][4] == BenchmarkType.GAIA

        # Next 2 combos should have H100
        assert combos[2][0] == GpuType.H100_80GB
        assert combos[2][4] == BenchmarkType.HLE
        assert combos[3][0] == GpuType.H100_80GB
        assert combos[3][4] == BenchmarkType.GAIA

    def test_get_all_combinations_full_loop_order(self):
        """Test full 5-level loop ordering."""
        config = GridConfig(
            gpu_types=[GpuType.A100_80GB],                      # 1
            resource_configs=[ResourceConfig.ONE_GPU_8CPU,
                            ResourceConfig.FOUR_GPU_32CPU],     # 2
            agents=[AgentType.REACT],                           # 1
            models=[ModelType.QWEN3_8B, ModelType.GPT_OSS_20B], # 2
            benchmarks=[BenchmarkType.HLE],                     # 1
        )
        combos = list(config.get_all_combinations())
        assert len(combos) == 4  # 1 * 2 * 1 * 2 * 1

        # resource_config is 2nd outermost, model is 4th
        # Order should be: (A100, 1gpu, react, qwen, hle),
        #                  (A100, 1gpu, react, gpt, hle),
        #                  (A100, 4gpu, react, qwen, hle),
        #                  (A100, 4gpu, react, gpt, hle)
        assert combos[0][1] == ResourceConfig.ONE_GPU_8CPU
        assert combos[0][3] == ModelType.QWEN3_8B
        assert combos[1][1] == ResourceConfig.ONE_GPU_8CPU
        assert combos[1][3] == ModelType.GPT_OSS_20B
        assert combos[2][1] == ResourceConfig.FOUR_GPU_32CPU
        assert combos[2][3] == ModelType.QWEN3_8B
        assert combos[3][1] == ResourceConfig.FOUR_GPU_32CPU
        assert combos[3][3] == ModelType.GPT_OSS_20B

    def test_describe(self):
        config = GridConfig()
        desc = config.describe()
        assert "Grid Search Configuration" in desc
        assert "Loop order" in desc
        assert "GPU Types" in desc
        assert "Resource Configs" in desc
        assert "Agents" in desc
        assert "Models" in desc
        assert "Benchmarks" in desc
        assert "Use full datasets: True" in desc
        assert "HLE text only: True" in desc
        assert "64" in desc  # total combinations
        assert "6400" in desc  # total queries

    def test_legacy_hardware_configs_preserved(self):
        """Test that legacy hardware_configs field is preserved."""
        config = GridConfig()
        assert HardwareConfig.A100_1GPU in config.hardware_configs
        assert HardwareConfig.A100_4GPU in config.hardware_configs
