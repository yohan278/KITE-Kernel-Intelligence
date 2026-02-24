"""Benchmark registry for managing evaluation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    name: str
    """Benchmark name (e.g., 'gaia', 'hle', 'simpleqa')"""

    description: str
    """Brief description of the benchmark"""

    domains: List[str]
    """Available evaluation domains"""

    default_domain: str
    """Default domain to use"""

    metrics: List[str]
    """Metrics computed by this benchmark"""


class BenchmarkRegistry:
    """Registry for evaluation benchmarks.

    Example:
        registry = BenchmarkRegistry()

        @registry.register("gaia")
        class GAIARunner:
            ...

        # Get benchmark
        runner_cls = registry.get("gaia")
        runner = runner_cls(config)
        results = runner.run(model)
    """

    _benchmarks: Dict[str, Type] = {}
    _configs: Dict[str, BenchmarkConfig] = {}

    @classmethod
    def register(
        cls,
        name: str,
        config: Optional[BenchmarkConfig] = None,
    ) -> Callable:
        """Register a benchmark class.

        Args:
            name: Benchmark identifier
            config: Optional benchmark configuration

        Returns:
            Decorator function
        """
        def decorator(benchmark_cls: Type) -> Type:
            cls._benchmarks[name] = benchmark_cls
            if config:
                cls._configs[name] = config
            return benchmark_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a benchmark class by name."""
        return cls._benchmarks.get(name)

    @classmethod
    def get_config(cls, name: str) -> Optional[BenchmarkConfig]:
        """Get benchmark configuration."""
        return cls._configs.get(name)

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmarks."""
        return list(cls._benchmarks.keys())

    @classmethod
    def list_all(cls) -> Dict[str, BenchmarkConfig]:
        """List all benchmarks with their configurations."""
        return {
            name: cls._configs.get(name)
            for name in cls._benchmarks.keys()
        }


def register_benchmark(
    name: str,
    description: str = "",
    domains: List[str] = None,
    default_domain: str = "",
    metrics: List[str] = None,
) -> Callable:
    """Convenience decorator to register a benchmark.

    Example:
        @register_benchmark(
            name="gaia",
            description="General AI Assistants benchmark",
            domains=["all", "level1", "level2"],
            default_domain="all",
            metrics=["accuracy", "level1_accuracy"],
        )
        class GAIARunner:
            ...
    """
    config = BenchmarkConfig(
        name=name,
        description=description,
        domains=domains or [],
        default_domain=default_domain,
        metrics=metrics or ["accuracy"],
    )
    return BenchmarkRegistry.register(name, config)
