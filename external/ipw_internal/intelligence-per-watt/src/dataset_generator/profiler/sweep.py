"""Sweep configuration for profiling dimension grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterator, List


# Mapping from SweepConfig field names to the singular key used in sweep points
_DIM_KEY_MAP = {
    "batch_sizes": "batch_size",
    "prefill_seq_lengths": "seq_len",
    "kv_cache_sizes": "kv_cache_size",
    "message_sizes_bytes": "message_size_bytes",
    "gpu_topologies": "num_gpus",
}


@dataclass
class SweepConfig:
    """Configuration for profiling sweep dimensions.

    Each list field defines the values to sweep for that dimension.
    """

    batch_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )
    prefill_seq_lengths: List[int] = field(
        default_factory=lambda: [
            128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        ]
    )
    kv_cache_sizes: List[int] = field(
        default_factory=lambda: [
            128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
        ]
    )
    message_sizes_bytes: List[int] = field(
        default_factory=lambda: [64, 256, 1024, 4096, 16384, 65536]
    )
    gpu_topologies: List[int] = field(default_factory=lambda: [1, 8])
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    use_energy: bool = False
    profiling_mode: str = "unfused"  # "unfused" | "fused" | "both"

    @classmethod
    def for_chat(cls, **overrides) -> "SweepConfig":
        """Preset for chat/conversational workloads."""
        defaults = dict(
            prefill_seq_lengths=[256, 512, 1024, 2048, 4096],
            kv_cache_sizes=[128, 256, 512, 1024, 2048, 4096],
            gpu_topologies=[1, 8],
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_reasoning(cls, **overrides) -> "SweepConfig":
        """Preset for reasoning workloads with long decode chains."""
        defaults = dict(
            kv_cache_sizes=[4096, 8192, 16384, 32768],
            gpu_topologies=[1, 8],
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_agentic(cls, **overrides) -> "SweepConfig":
        """Preset for agentic workloads with accumulated context."""
        defaults = dict(
            prefill_seq_lengths=[1024, 1536, 2048, 2560, 3072, 4096, 5120],
            kv_cache_sizes=[1024, 2048, 4096, 8192, 16384, 32768],
            gpu_topologies=[1, 8],
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_rag(cls, **overrides) -> "SweepConfig":
        """Preset for RAG workloads with large retrieved document prefills."""
        defaults = dict(
            prefill_seq_lengths=[5120, 10240, 20480],
            gpu_topologies=[1, 8],
        )
        defaults.update(overrides)
        return cls(**defaults)

    def get_sweep_points(self, dims: List[str]) -> Iterator[Dict[str, Any]]:
        """Generate cartesian product of requested sweep dimensions.

        Args:
            dims: List of dimension field names (e.g., ["batch_sizes", "prefill_seq_lengths"]).

        Yields:
            Dicts mapping singular dimension keys to values.
            E.g., {"batch_size": 1, "seq_len": 128}
        """
        if not dims:
            yield {}
            return

        dim_values = []
        dim_keys = []
        for dim_name in dims:
            values = getattr(self, dim_name)
            if not isinstance(values, list):
                raise ValueError(f"{dim_name} is not a list dimension")
            dim_values.append(values)
            dim_keys.append(_DIM_KEY_MAP.get(dim_name, dim_name))

        for combo in product(*dim_values):
            yield dict(zip(dim_keys, combo))
