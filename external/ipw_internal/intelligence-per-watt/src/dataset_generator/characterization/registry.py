"""Characterizer registry — maps dataset names to characterizer classes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

from dataset_generator.characterization.base import BaseCharacterizer

# Populated after characterizer modules are defined; they register themselves
# by being imported in this module's _ensure_registered().
CHARACTERIZER_REGISTRY: Dict[str, Type[BaseCharacterizer]] = {}

_registered = False


def _ensure_registered():
    """Lazily import characterizer modules to populate the registry."""
    global _registered
    if _registered:
        return
    _registered = True
    try:
        from dataset_generator.characterization import chat_characterizer  # noqa: F401
        from dataset_generator.characterization import reasoning_characterizer  # noqa: F401
        from dataset_generator.characterization import rag_characterizer  # noqa: F401
        from dataset_generator.characterization import agentic_characterizer  # noqa: F401
        from dataset_generator.characterization import coding_characterizer  # noqa: F401
    except ImportError:
        pass


def register_characterizer(name: str, cls: Type[BaseCharacterizer]):
    """Register a characterizer class under *name*."""
    CHARACTERIZER_REGISTRY[name] = cls


def characterize_workload(name: str, limit: Optional[int] = None):
    """Run a single named characterizer and return its WorkloadProfile."""
    _ensure_registered()
    if name not in CHARACTERIZER_REGISTRY:
        raise KeyError(
            f"Unknown characterizer '{name}'. "
            f"Available: {sorted(CHARACTERIZER_REGISTRY.keys())}"
        )
    cls = CHARACTERIZER_REGISTRY[name]
    return cls().characterize(limit=limit)


def characterize_all(
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run all registered characterizers.

    Args:
        limit: Max samples per characterizer.
        output_dir: If given, save each profile as ``<name>_profile.json``.

    Returns:
        Dict mapping dataset name to WorkloadProfile.
    """
    _ensure_registered()
    results = {}
    for name, cls in CHARACTERIZER_REGISTRY.items():
        profile = cls().characterize(limit=limit)
        if output_dir:
            profile.save(Path(output_dir) / f"{name}_profile.json")
        results[name] = profile
    return results
