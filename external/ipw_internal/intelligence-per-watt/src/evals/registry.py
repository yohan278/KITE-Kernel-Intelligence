from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Tuple

_BENCHMARKS: Dict[str, Callable[..., Any]] = {}
_DISCOVERED = False


def register_benchmark(key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a benchmark factory under ``key``."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if key in _BENCHMARKS:
            raise ValueError(f"Benchmark '{key}' already registered")
        _BENCHMARKS[key] = factory
        return factory

    return decorator


def _ensure_discovered() -> None:
    """Auto-discover and import benchmark modules (any package with main.py)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    import os
    from pathlib import Path

    # Benchmarks live under evals.src.benchmarks.<name>.main
    src_dir = Path(__file__).resolve().parent
    benchmarks_dir = src_dir / "benchmarks"
    if not benchmarks_dir.is_dir():
        return

    for entry in benchmarks_dir.iterdir():
        if not entry.is_dir():
            continue
        main_file = entry / "main.py"
        if not main_file.is_file():
            continue

        module_name = f"evals.benchmarks.{entry.name}.main"
        try:
            importlib.import_module(module_name)
        except Exception:
            # Best-effort; skip benchmarks that fail to import
            continue


def get_benchmark(key: str) -> Callable[..., Any]:
    _ensure_discovered()
    try:
        return _BENCHMARKS[key]
    except KeyError as exc:
        raise KeyError(f"Benchmark '{key}' is not registered") from exc


def benchmark_keys() -> Tuple[str, ...]:
    _ensure_discovered()
    return tuple(_BENCHMARKS.keys())


