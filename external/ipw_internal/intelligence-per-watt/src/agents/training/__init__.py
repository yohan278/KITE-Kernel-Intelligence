"""Training module for agents.

This module contains training pipelines for different agents.
Currently supports:
- orchestrator: Trained policy-based orchestrator (now a top-level package)

Note: The orchestrator package is now a top-level package. Import it directly:
    import orchestrator
or access via this module for backward compatibility:
    from agents.training import orchestrator
"""

__all__ = [
    "orchestrator",
]


def __getattr__(name: str):
    """Lazy import for backward compatibility."""
    if name == "orchestrator":
        import orchestrator
        return orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
