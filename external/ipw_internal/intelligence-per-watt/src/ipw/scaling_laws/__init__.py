"""Scaling-law experiment helpers for sweep, aggregation, and fitting."""

from __future__ import annotations

from typing import Any


def aggregate_summaries(*args: Any, **kwargs: Any):
    from .aggregate import aggregate_summaries as _aggregate

    return _aggregate(*args, **kwargs)


def fit_scaling_laws(*args: Any, **kwargs: Any):
    from .fit import fit_scaling_laws as _fit

    return _fit(*args, **kwargs)


def estimate_hardware_normalization(*args: Any, **kwargs: Any):
    from .hardware_normalize import estimate_hardware_normalization as _estimate

    return _estimate(*args, **kwargs)


__all__ = [
    "aggregate_summaries",
    "fit_scaling_laws",
    "estimate_hardware_normalization",
]
