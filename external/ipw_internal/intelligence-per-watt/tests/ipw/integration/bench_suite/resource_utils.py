"""GPU/CPU resource constraint utilities for integration tests.

This module provides utilities for configuring GPU visibility and CPU affinity
during benchmark tests to simulate different resource configurations.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class ResourceConfig:
    """Configuration for compute resources.

    Attributes:
        name: Human-readable identifier for the configuration
        gpu_count: Number of GPUs to make visible
        cpu_count: Number of CPUs to bind to
        tensor_parallel_size: Tensor parallel size for vLLM
    """

    name: str
    gpu_count: int
    cpu_count: int
    tensor_parallel_size: int


# Available resource configurations for testing
RESOURCE_CONFIGS = {
    "1gpu_8cpu": ResourceConfig(
        name="1gpu_8cpu",
        gpu_count=1,
        cpu_count=8,
        tensor_parallel_size=1,
    ),
    "4gpu_32cpu": ResourceConfig(
        name="4gpu_32cpu",
        gpu_count=4,
        cpu_count=32,
        tensor_parallel_size=4,
    ),
}


def get_resource_config(name: str) -> ResourceConfig:
    """Get resource configuration by name.

    Args:
        name: Configuration name (e.g., "1gpu_8cpu", "4gpu_32cpu")

    Returns:
        ResourceConfig instance

    Raises:
        KeyError: If configuration name not found
    """
    if name not in RESOURCE_CONFIGS:
        available = ", ".join(RESOURCE_CONFIGS.keys())
        raise KeyError(f"Unknown resource config: {name}. Available: {available}")
    return RESOURCE_CONFIGS[name]


def set_gpu_visibility(gpu_count: int) -> str:
    """Set CUDA_VISIBLE_DEVICES to limit visible GPUs.

    Args:
        gpu_count: Number of GPUs to make visible

    Returns:
        The CUDA_VISIBLE_DEVICES value that was set
    """
    if gpu_count <= 0:
        devices = ""
    else:
        devices = ",".join(str(i) for i in range(gpu_count))

    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    return devices


def set_cpu_affinity(cpu_count: int) -> Optional[set]:
    """Set CPU affinity to limit available CPUs.

    Uses os.sched_setaffinity() to bind the process to specific CPUs.
    Note: Only works on Linux systems.

    Args:
        cpu_count: Number of CPUs to bind to (uses CPUs 0 through cpu_count-1)

    Returns:
        Set of CPU IDs that were set, or None if not supported
    """
    if not hasattr(os, "sched_setaffinity"):
        # CPU affinity not supported on this platform (e.g., macOS)
        return None

    if cpu_count <= 0:
        return None

    # Bind to CPUs 0 through cpu_count-1
    cpu_set = set(range(cpu_count))
    try:
        os.sched_setaffinity(0, cpu_set)
        return cpu_set
    except (OSError, PermissionError):
        # May fail if not enough CPUs or insufficient permissions
        return None


def get_current_cpu_affinity() -> Optional[set]:
    """Get the current CPU affinity mask.

    Returns:
        Set of CPU IDs the process is bound to, or None if not supported
    """
    if not hasattr(os, "sched_getaffinity"):
        return None

    try:
        return os.sched_getaffinity(0)
    except (OSError, PermissionError):
        return None


class ResourceManager:
    """Context manager for configuring compute resources during tests.

    Saves the original environment state and restores it after the test.
    """

    def __init__(self) -> None:
        self._original_cuda_visible_devices: Optional[str] = None
        self._original_cpu_affinity: Optional[set] = None

    def _save_state(self) -> None:
        """Save current environment state."""
        self._original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._original_cpu_affinity = get_current_cpu_affinity()

    def _restore_state(self) -> None:
        """Restore original environment state."""
        # Restore CUDA_VISIBLE_DEVICES
        if self._original_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_visible_devices
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # Restore CPU affinity
        if self._original_cpu_affinity is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, self._original_cpu_affinity)
            except (OSError, PermissionError):
                pass

    @contextmanager
    def configure(self, config_name: str) -> Generator[ResourceConfig, None, None]:
        """Configure resources for a test run.

        Args:
            config_name: Name of the resource configuration to apply

        Yields:
            The ResourceConfig that was applied
        """
        config = get_resource_config(config_name)
        self._save_state()

        try:
            set_gpu_visibility(config.gpu_count)
            set_cpu_affinity(config.cpu_count)
            yield config
        finally:
            self._restore_state()


@contextmanager
def resource_context(config_name: str) -> Generator[ResourceConfig, None, None]:
    """Convenience context manager for resource configuration.

    Args:
        config_name: Name of the resource configuration to apply

    Yields:
        The ResourceConfig that was applied

    Example:
        with resource_context("1gpu_8cpu") as config:
            # Run test with limited resources
            print(f"Using {config.gpu_count} GPUs")
    """
    manager = ResourceManager()
    with manager.configure(config_name) as config:
        yield config
