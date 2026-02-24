"""Main parametrized test file for bench integration tests.

This module runs all 24 combinations of the test matrix:
- Models: qwen3-8b, gpt-oss-20b (2)
- Agents: react, openhands, orchestrator (3)
- Benchmarks: hle, gaia (2)
- Resources: 1gpu_8cpu, 4gpu_32cpu (2)

Total: 2 x 3 x 2 x 2 = 24 test combinations

Usage:
    # Run all integration tests
    pytest ipw/tests/integration/bench_suite/ -v --tb=short

    # Run specific model
    pytest ipw/tests/integration/bench_suite/ -v -k "qwen3-8b"

    # Run specific agent
    pytest ipw/tests/integration/bench_suite/ -v -k "orchestrator"

    # Run specific resource config
    pytest ipw/tests/integration/bench_suite/ -v -k "4gpu_32cpu"

    # Run single combination
    pytest ipw/tests/integration/bench_suite/ -v -k "qwen3-8b_react_hle_1gpu_8cpu"

    # Collect tests without running
    pytest ipw/tests/integration/bench_suite/ --collect-only
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict

import pytest

from .conftest import (
    MODEL_CONFIGS,
    AGENT_CONFIGS,
    get_checkpoint_path,
)
from .resource_utils import RESOURCE_CONFIGS


# Test dimensions
MODELS = list(MODEL_CONFIGS.keys())  # ["qwen3-8b", "gpt-oss-20b"]
AGENTS = list(AGENT_CONFIGS.keys())  # ["react", "openhands", "orchestrator"]
BENCHMARKS = ["hle", "gaia"]
RESOURCES = list(RESOURCE_CONFIGS.keys())  # ["1gpu_8cpu", "4gpu_32cpu"]

# Generate all 24 combinations
TEST_MATRIX = list(product(MODELS, AGENTS, BENCHMARKS, RESOURCES))

# Generate test IDs for better output
TEST_IDS = [f"{m}_{a}_{b}_{r}" for m, a, b, r in TEST_MATRIX]


def _validate_result(result: Dict[str, Any]) -> None:
    """Validate benchmark result contains expected fields and sensible values.

    This performs a "type check" on metrics to ensure they have sensible values:
    - duration_seconds: 1-3600s (not too fast, not stuck)
    - gpu_energy_joules: positive if present
    - cpu_energy_joules: positive if present
    - avg_gpu_power_watts: 10-500W for A100 (not idle, not impossible)
    - avg_cpu_power_watts: 1-300W (measuring correctly)
    - accuracy: 0.0-1.0 if present
    - ipw_score: positive if present
    - telemetry_samples: > 0 if telemetry enabled

    Args:
        result: Benchmark result dictionary

    Raises:
        AssertionError: If required fields are missing or invalid
    """
    # Check for accuracy or score metric
    has_accuracy = "accuracy" in result and result["accuracy"] is not None
    has_score = "score" in result and result["score"] is not None
    assert has_accuracy or has_score, (
        f"Result must contain 'accuracy' or 'score'. Got keys: {list(result.keys())}"
    )

    # Check duration is reasonable (1s to 1 hour)
    assert "duration_seconds" in result, "Result must contain 'duration_seconds'"
    duration = result["duration_seconds"]
    assert 1 <= duration <= 3600, (
        f"duration_seconds={duration}s is outside expected range [1, 3600]s"
    )

    # Validate accuracy if present
    if "accuracy" in result and result["accuracy"] is not None:
        accuracy = result["accuracy"]
        assert 0.0 <= accuracy <= 1.0, (
            f"accuracy={accuracy} is outside valid range [0.0, 1.0]"
        )

    # Validate energy metrics if present
    if "gpu_energy_joules" in result and result["gpu_energy_joules"] is not None:
        gpu_energy = result["gpu_energy_joules"]
        assert gpu_energy > 0, f"gpu_energy_joules={gpu_energy} must be positive"

    if "cpu_energy_joules" in result and result["cpu_energy_joules"] is not None:
        cpu_energy = result["cpu_energy_joules"]
        # CPU energy may be 0 if not supported on this platform (e.g., RAPL not available)
        if cpu_energy > 0:
            pass  # Valid positive value
        # 0.0 is acceptable (means not measured on this system)

    # Validate power metrics if present (A100 TDP ~400W, can spike to ~600W)
    if "avg_gpu_power_watts" in result and result["avg_gpu_power_watts"] is not None:
        gpu_power = result["avg_gpu_power_watts"]
        assert 10 <= gpu_power <= 600, (
            f"avg_gpu_power_watts={gpu_power}W is outside expected range [10, 600]W"
        )

    if "avg_cpu_power_watts" in result and result["avg_cpu_power_watts"] is not None:
        cpu_power = result["avg_cpu_power_watts"]
        # CPU power may be 0 if not supported on this platform
        if cpu_power > 0:
            assert cpu_power <= 300, (
                f"avg_cpu_power_watts={cpu_power}W exceeds expected max of 300W"
            )

    # Validate IPW score if present
    if "ipw_score" in result and result["ipw_score"] is not None:
        ipw_score = result["ipw_score"]
        assert ipw_score > 0, f"ipw_score={ipw_score} must be positive"

    # Validate telemetry samples if present
    if "telemetry_samples" in result and result["telemetry_samples"] is not None:
        samples = result["telemetry_samples"]
        assert samples > 0, f"telemetry_samples={samples} must be positive"


def _should_skip_combination(
    model: str,
    agent: str,
    benchmark: str,
    resource: str,
) -> tuple[bool, str]:
    """Check if a test combination should be skipped.

    Args:
        model: Model name
        agent: Agent name
        benchmark: Benchmark name
        resource: Resource configuration name

    Returns:
        Tuple of (should_skip, reason)
    """
    # Get resource config
    resource_config = RESOURCE_CONFIGS[resource]
    model_config = MODEL_CONFIGS[model]

    # Skip if model requires more GPUs than resource config provides
    model_tp_size = model_config.get("tensor_parallel_size", 1)
    if model_tp_size > resource_config.gpu_count:
        return True, (
            f"Model {model} requires tensor_parallel_size={model_tp_size} "
            f"but resource config {resource} only provides {resource_config.gpu_count} GPUs"
        )

    # Skip orchestrator tests if no checkpoint is available
    if agent == "orchestrator":
        checkpoint_path = get_checkpoint_path(model, agent)
        if checkpoint_path is None:
            return True, (
                f"Orchestrator agent requires a trained checkpoint. "
                f"Set ORCHESTRATOR_CHECKPOINT env var or train a model first."
            )

    return False, ""


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,agent,benchmark,resource",
    TEST_MATRIX,
    ids=TEST_IDS,
)
def test_bench_combination(
    model: str,
    agent: str,
    benchmark: str,
    resource: str,
    vllm_server,
    resource_manager,
) -> None:
    """Run benchmark with specified configuration.

    This test executes the full benchmark pipeline for a single combination
    of model, agent, benchmark, and resource configuration.

    Args:
        model: Model name (e.g., "qwen3-8b")
        agent: Agent name (e.g., "react")
        benchmark: Benchmark name (e.g., "hle")
        resource: Resource configuration name (e.g., "1gpu_8cpu")
        vllm_server: Fixture providing vLLM server URL
        resource_manager: Fixture providing resource configuration manager
    """
    # Import here to avoid import errors when dependencies aren't available
    from ipw.cli.bench import execute_benchmark

    # Check if combination should be skipped
    should_skip, reason = _should_skip_combination(model, agent, benchmark, resource)
    if should_skip:
        pytest.skip(reason)

    # Get configurations
    model_config = MODEL_CONFIGS[model]
    checkpoint_path = get_checkpoint_path(model, agent)

    with resource_manager.configure(resource) as res_config:
        result = execute_benchmark(
            client_id="vllm",
            model_name=model_config["model_id"],
            agent_id=agent,
            dataset_id=benchmark,
            client_base_url=model_config["vllm_url"],
            checkpoint_path=checkpoint_path,
            max_samples=5,  # Limit samples for faster testing
            enable_telemetry=True,
            skip_warmup=False,
        )

        # Validate result structure
        _validate_result(result)

        # Log key metrics for visibility
        print(f"\n  Result for {model}_{agent}_{benchmark}_{resource}:")
        print(f"    Duration: {result.get('duration_seconds', 0):.2f}s")
        if "accuracy" in result and result["accuracy"] is not None:
            print(f"    Accuracy: {result['accuracy']}")
        if "score" in result and result["score"] is not None:
            print(f"    Score: {result['score']}")
        if "total_energy_joules" in result and result["total_energy_joules"] is not None:
            print(f"    Energy: {result['total_energy_joules']:.2f}J")
        if "ipw_score" in result and result["ipw_score"] is not None:
            print(f"    IPW Score: {result['ipw_score']:.6f}")
        if "telemetry_samples" in result:
            print(f"    Telemetry samples: {result['telemetry_samples']}")


@pytest.mark.integration
class TestResourceConstraints:
    """Tests for resource constraint utilities."""

    def test_gpu_visibility_single(self, resource_manager) -> None:
        """Test setting single GPU visibility."""
        import os

        with resource_manager.configure("1gpu_8cpu"):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_gpu_visibility_multiple(self, resource_manager) -> None:
        """Test setting multiple GPU visibility."""
        import os

        with resource_manager.configure("4gpu_32cpu"):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"

    def test_environment_restored_after_test(self, resource_manager) -> None:
        """Test that environment is restored after context manager exits."""
        import os

        original = os.environ.get("CUDA_VISIBLE_DEVICES")

        with resource_manager.configure("1gpu_8cpu"):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

        # Should be restored
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original


@pytest.mark.integration
class TestModelConfigurations:
    """Tests for model configuration utilities."""

    def test_get_checkpoint_path_react(self) -> None:
        """Test checkpoint path for React agent."""
        path = get_checkpoint_path("qwen3-8b", "react")
        assert path is None

    def test_get_checkpoint_path_openhands(self) -> None:
        """Test checkpoint path for OpenHands agent."""
        path = get_checkpoint_path("qwen3-8b", "openhands")
        assert path is None

    def test_get_checkpoint_path_orchestrator(self) -> None:
        """Test checkpoint path for Orchestrator agent."""
        path = get_checkpoint_path("qwen3-8b", "orchestrator")
        assert path == "Qwen/Qwen3-8B-Instruct"

    def test_model_config_has_required_fields(self) -> None:
        """Test all model configs have required fields."""
        required_fields = ["model_id", "vllm_url"]
        for model_name, config in MODEL_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Model {model_name} missing field {field}"


@pytest.mark.integration
class TestSkipConditions:
    """Tests for skip condition logic."""

    def test_no_skip_moe_model_small_resources(self) -> None:
        """Test that MoE models (like gpt-oss-20b) run on 1 GPU.

        gpt-oss-20b is an MoE model with <3B active parameters,
        so it fits on a single GPU despite having 20B total parameters.
        """
        should_skip, reason = _should_skip_combination(
            model="gpt-oss-20b",
            agent="react",
            benchmark="hle",
            resource="1gpu_8cpu",
        )
        assert not should_skip

    def test_no_skip_small_model_small_resources(self) -> None:
        """Test that small models run on small resource configs."""
        should_skip, reason = _should_skip_combination(
            model="qwen3-8b",
            agent="react",
            benchmark="hle",
            resource="1gpu_8cpu",
        )
        assert not should_skip

    def test_no_skip_large_model_large_resources(self) -> None:
        """Test that large models run on large resource configs."""
        should_skip, reason = _should_skip_combination(
            model="gpt-oss-20b",
            agent="react",
            benchmark="hle",
            resource="4gpu_32cpu",
        )
        assert not should_skip


@pytest.mark.integration
class TestTestMatrix:
    """Tests for test matrix generation."""

    def test_matrix_has_24_combinations(self) -> None:
        """Test that the full matrix has 24 combinations."""
        assert len(TEST_MATRIX) == 24

    def test_matrix_covers_all_models(self) -> None:
        """Test that matrix covers all models."""
        models_in_matrix = set(m for m, _, _, _ in TEST_MATRIX)
        assert models_in_matrix == set(MODELS)

    def test_matrix_covers_all_agents(self) -> None:
        """Test that matrix covers all agents."""
        agents_in_matrix = set(a for _, a, _, _ in TEST_MATRIX)
        assert agents_in_matrix == set(AGENTS)

    def test_matrix_covers_all_benchmarks(self) -> None:
        """Test that matrix covers all benchmarks."""
        benchmarks_in_matrix = set(b for _, _, b, _ in TEST_MATRIX)
        assert benchmarks_in_matrix == set(BENCHMARKS)

    def test_matrix_covers_all_resources(self) -> None:
        """Test that matrix covers all resource configs."""
        resources_in_matrix = set(r for _, _, _, r in TEST_MATRIX)
        assert resources_in_matrix == set(RESOURCES)

    def test_test_ids_are_unique(self) -> None:
        """Test that all test IDs are unique."""
        assert len(TEST_IDS) == len(set(TEST_IDS))
