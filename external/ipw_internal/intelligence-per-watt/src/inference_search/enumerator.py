"""Configuration enumerator with feasibility filtering."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from inference_simulator.types import HardwareSpec, InferenceSpec, ModelSpec

from inference_search.types import SearchConfig


def enumerate_configurations(
    search_config: SearchConfig,
) -> List[Tuple[ModelSpec, HardwareSpec, InferenceSpec]]:
    """Generate all feasible (model, hardware, inference_spec) triples.

    Produces the Cartesian product of model_specs x hardware_specs x inference_specs,
    then filters out infeasible configurations:
      - Model memory footprint exceeds total GPU memory.
      - tensor_parallel degree exceeds num_gpus.

    Args:
        search_config: Search configuration containing the spec lists.

    Returns:
        List of feasible (ModelSpec, HardwareSpec, InferenceSpec) triples.
    """
    feasible: List[Tuple[ModelSpec, HardwareSpec, InferenceSpec]] = []

    for model in search_config.model_specs:
        for hw in search_config.hardware_specs:
            for inf in search_config.inference_specs:
                if _is_feasible(model, hw, inf):
                    feasible.append((model, hw, inf))

    return feasible


def _is_feasible(
    model: ModelSpec,
    hw: HardwareSpec,
    inf: InferenceSpec,
) -> bool:
    """Check whether a configuration is feasible.

    A configuration is infeasible if:
      1. The model weight memory exceeds total GPU memory across all GPUs.
      2. The tensor_parallel degree exceeds num_gpus.

    Args:
        model: Model architecture spec.
        hw: Hardware spec for a single GPU.
        inf: Inference serving configuration.

    Returns:
        True if the configuration is feasible.
    """
    # Check tensor parallelism fits in available GPUs
    if inf.tensor_parallel > inf.num_gpus:
        return False

    # Estimate model weight memory: total_params * bytes_per_param
    model_memory_gb = model.total_params_billion * hw.bytes_per_param

    # Total GPU memory available across all GPUs
    total_gpu_memory_gb = hw.memory_gb * inf.num_gpus

    if model_memory_gb > total_gpu_memory_gb:
        return False

    return True


__all__ = ["enumerate_configurations"]
