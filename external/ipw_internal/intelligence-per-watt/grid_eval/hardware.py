"""Hardware abstraction for grid evaluation.

Manages environment variables for GPU and CPU configuration.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from grid_eval.config import (
    HARDWARE_REGISTRY,
    RESOURCE_CONFIG_REGISTRY,
    HardwareConfig,
    ResourceConfig,
)


class HardwareManager:
    """Context manager for hardware configuration.

    Sets environment variables for CUDA devices and CPU threads,
    then restores the original environment on exit.

    Note: This controls which GPUs the energy monitor tracks via
    CUDA_VISIBLE_DEVICES. For vLLM, the GridEvalRunner will restart
    the server with matching tensor_parallel_size to ensure the
    inference server uses the correct number of GPUs.

    Example:
        >>> with HardwareManager(HardwareConfig.A100_1GPU) as hw:
        ...     # Run evaluation with 1 A100 GPU and 8 CPUs
        ...     pass
        >>> # Original env vars are restored
    """

    ENV_VARS = ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]

    def __init__(self, config: HardwareConfig) -> None:
        """Initialize hardware manager.

        Args:
            config: Hardware configuration to apply
        """
        self.config = config
        self._original_env: Dict[str, Optional[str]] = {}

    def __enter__(self) -> "HardwareManager":
        """Apply hardware configuration."""
        # Save original environment
        for var in self.ENV_VARS:
            self._original_env[var] = os.environ.get(var)

        # Apply new configuration (only string env vars)
        settings = HARDWARE_REGISTRY[self.config]
        for var, value in settings.items():
            if isinstance(value, str):
                os.environ[var] = value

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original environment."""
        for var in self.ENV_VARS:
            original = self._original_env.get(var)
            if original is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original

    @property
    def cuda_devices(self) -> str:
        """Return the CUDA_VISIBLE_DEVICES setting."""
        return HARDWARE_REGISTRY[self.config]["CUDA_VISIBLE_DEVICES"]

    @property
    def num_threads(self) -> int:
        """Return the number of CPU threads configured."""
        return int(HARDWARE_REGISTRY[self.config]["OMP_NUM_THREADS"])

    def describe(self) -> str:
        """Return human-readable description of hardware config."""
        settings = HARDWARE_REGISTRY[self.config]
        gpus = settings["CUDA_VISIBLE_DEVICES"].split(",")
        threads = settings["OMP_NUM_THREADS"]
        return f"{len(gpus)} GPU(s) [{settings['CUDA_VISIBLE_DEVICES']}], {threads} CPU threads"


class ResourceManager:
    """Context manager for resource configuration (new API).

    Sets environment variables for CUDA devices and CPU threads
    based on ResourceConfig, then restores the original environment on exit.

    This class separates resource allocation (GPU count, CPU count) from
    GPU hardware type, enabling a cleaner 5-level grid search loop.

    For Apple Silicon (gpu_vendor="apple"), CUDA_VISIBLE_DEVICES is skipped
    since Apple devices don't use CUDA.

    Example:
        >>> with ResourceManager(ResourceConfig.ONE_GPU_8CPU) as rm:
        ...     # Run evaluation with 1 GPU and 8 CPUs
        ...     pass
        >>> # Original env vars are restored

        >>> # Apple Silicon mode - skips CUDA env vars
        >>> with ResourceManager(ResourceConfig.ONE_GPU_8CPU, gpu_vendor="apple") as rm:
        ...     pass
    """

    ENV_VARS = ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]

    def __init__(
        self, resource_config: ResourceConfig, gpu_vendor: Optional[str] = None
    ) -> None:
        """Initialize resource manager.

        Args:
            resource_config: Resource configuration to apply
            gpu_vendor: GPU vendor ("nvidia", "amd", "apple"). Defaults to "nvidia".
                       For Apple Silicon, CUDA_VISIBLE_DEVICES is not set.
        """
        self.resource_config = resource_config
        self.gpu_vendor = gpu_vendor or "nvidia"
        self._original_env: Dict[str, Optional[str]] = {}

    def __enter__(self) -> "ResourceManager":
        """Apply resource configuration."""
        # Save original environment
        for var in self.ENV_VARS:
            self._original_env[var] = os.environ.get(var)

        # Apply new configuration (only string env vars)
        settings = RESOURCE_CONFIG_REGISTRY[self.resource_config].copy()

        # Apple Silicon doesn't use CUDA_VISIBLE_DEVICES
        if self.gpu_vendor == "apple":
            settings.pop("CUDA_VISIBLE_DEVICES", None)

        for var, value in settings.items():
            if isinstance(value, str):
                os.environ[var] = value

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original environment."""
        for var in self.ENV_VARS:
            original = self._original_env.get(var)
            if original is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original

    @property
    def cuda_devices(self) -> Optional[str]:
        """Return the CUDA_VISIBLE_DEVICES setting.

        Returns None for Apple Silicon since CUDA is not used.
        """
        if self.gpu_vendor == "apple":
            return None
        return RESOURCE_CONFIG_REGISTRY[self.resource_config]["CUDA_VISIBLE_DEVICES"]

    @property
    def gpu_count(self) -> int:
        """Return the number of GPUs configured."""
        return RESOURCE_CONFIG_REGISTRY[self.resource_config]["gpu_count"]

    @property
    def cpu_count(self) -> int:
        """Return the number of CPU threads configured."""
        return RESOURCE_CONFIG_REGISTRY[self.resource_config]["cpu_count"]

    def describe(self) -> str:
        """Return human-readable description of resource config."""
        settings = RESOURCE_CONFIG_REGISTRY[self.resource_config]
        gpus = settings["gpu_count"]
        cpus = settings["cpu_count"]
        if self.gpu_vendor == "apple":
            return f"{gpus} GPU(s) [Apple Silicon], {cpus} CPU threads"
        cuda = settings["CUDA_VISIBLE_DEVICES"]
        return f"{gpus} GPU(s) [{cuda}], {cpus} CPU threads"


__all__ = ["HardwareManager", "ResourceManager"]
