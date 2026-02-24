"""Fast numpy interpolation on dense .npz lookup tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class OutOfRangeError(ValueError):
    """Query is outside the profiled range and cannot be reliably interpolated."""

    pass


class LUTLookup:
    """Fast numpy interpolation on dense .npz lookup tables.

    Loads a .npz file produced by LUTGenerator and provides N-D linear
    interpolation between nearest grid points.
    """

    def __init__(self, npz_path: Path) -> None:
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        self._grid = data["grid"]
        self._axis_names: List[str] = list(data["axis_names"])
        self._axes: List[np.ndarray] = [
            data[f"axis_{i}"] for i in range(len(self._axis_names))
        ]

    @property
    def axis_names(self) -> List[str]:
        return list(self._axis_names)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._grid.shape

    def lookup(self, **dims: Union[int, float, str]) -> Union[float, np.ndarray]:
        """Interpolate value at arbitrary input dimensions.

        For numeric axes, performs N-D linear interpolation between the
        two nearest grid points. For string/categorical axes (e.g., operator
        names), performs exact match.

        Args:
            **dims: Keyword arguments mapping axis names to query values.
                    E.g., ``lookup(seq_len=1024, batch_tokens=64, tp_size=2)``

        Returns:
            Interpolated scalar value, or array if the grid has a trailing
            dimension (e.g., [seconds, joules]).
        """
        indices_low: List[int] = []
        indices_high: List[int] = []
        fractions: List[float] = []

        for i, name in enumerate(self._axis_names):
            if name not in dims:
                raise ValueError(
                    f"Missing dimension '{name}'. Required: {self._axis_names}"
                )
            val = dims[name]
            axis = self._axes[i]

            # String/categorical axis: exact match
            if axis.dtype.kind in ("U", "S", "O"):
                axis_list = list(axis)
                if val not in axis_list:
                    raise ValueError(
                        f"Value '{val}' not found in axis '{name}'. "
                        f"Available: {axis_list}"
                    )
                idx = axis_list.index(val)
                indices_low.append(idx)
                indices_high.append(idx)
                fractions.append(0.0)
            else:
                # Numeric axis: find bracketing indices for interpolation
                numeric_val = float(val)
                axis_float = axis.astype(float)

                # Reject extreme extrapolation (beyond 1.5x of profiled range)
                if numeric_val < axis_float[0] * 0.5:
                    raise OutOfRangeError(
                        f"Value {numeric_val} for axis '{name}' is below the "
                        f"profiled range [{axis_float[0]}, {axis_float[-1]}] "
                        f"and cannot be reliably interpolated."
                    )
                if numeric_val > axis_float[-1] * 1.5:
                    raise OutOfRangeError(
                        f"Value {numeric_val} for axis '{name}' is above the "
                        f"profiled range [{axis_float[0]}, {axis_float[-1]}] "
                        f"and cannot be reliably interpolated."
                    )

                # Allow minor extrapolation via clamping
                if numeric_val <= axis_float[0]:
                    indices_low.append(0)
                    indices_high.append(0)
                    fractions.append(0.0)
                elif numeric_val >= axis_float[-1]:
                    last = len(axis_float) - 1
                    indices_low.append(last)
                    indices_high.append(last)
                    fractions.append(0.0)
                else:
                    # Find the interval containing the value
                    idx_high = int(np.searchsorted(axis_float, numeric_val))
                    idx_low = idx_high - 1
                    lo_val = axis_float[idx_low]
                    hi_val = axis_float[idx_high]
                    frac = (numeric_val - lo_val) / (hi_val - lo_val) if hi_val != lo_val else 0.0
                    indices_low.append(idx_low)
                    indices_high.append(idx_high)
                    fractions.append(frac)

        # N-D linear interpolation: iterate over 2^N corners of the hypercube
        n_dims = len(self._axis_names)
        result = np.zeros_like(self._grid[tuple(indices_low)])
        result = result.astype(float)

        for corner in range(1 << n_dims):
            weight = 1.0
            idx = []
            for d in range(n_dims):
                if corner & (1 << d):
                    idx.append(indices_high[d])
                    weight *= fractions[d]
                else:
                    idx.append(indices_low[d])
                    weight *= 1.0 - fractions[d]

            if weight > 0:
                result += weight * self._grid[tuple(idx)].astype(float)

        # Return scalar if result is 0-d
        if result.ndim == 0:
            return float(result)
        # Return scalar if result is 1-d with single element
        if result.ndim == 1 and len(result) == 1:
            return float(result[0])
        return result
