"""Tests for OutOfRangeError on LUT extrapolation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from inference_simulator.estimator.lut_lookup import LUTLookup, OutOfRangeError


def _create_lut(axis_values, grid_values, axis_names=("seq_len",)):
    """Create a temporary LUT .npz file for testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    arrays = {
        "grid": np.array(grid_values),
        "axis_names": np.array(axis_names),
    }
    for i, vals in enumerate(
        [axis_values] if not isinstance(axis_values[0], (list, np.ndarray)) else axis_values
    ):
        arrays[f"axis_{i}"] = np.array(vals)
    np.savez(tmp.name, **arrays)
    return Path(tmp.name)


class TestOutOfRangeError:
    def test_is_value_error(self):
        assert issubclass(OutOfRangeError, ValueError)

    def test_query_below_range(self):
        """Query below 0.5x of minimum should raise."""
        path = _create_lut(
            axis_values=[100, 200, 400, 800],
            grid_values=[1.0, 2.0, 4.0, 8.0],
        )
        lut = LUTLookup(path)
        with pytest.raises(OutOfRangeError, match="below"):
            lut.lookup(seq_len=10)  # 10 < 100 * 0.5 = 50

    def test_query_above_range(self):
        """Query above 1.5x of maximum should raise."""
        path = _create_lut(
            axis_values=[100, 200, 400, 800],
            grid_values=[1.0, 2.0, 4.0, 8.0],
        )
        lut = LUTLookup(path)
        with pytest.raises(OutOfRangeError, match="above"):
            lut.lookup(seq_len=2000)  # 2000 > 800 * 1.5 = 1200

    def test_minor_extrapolation_clamped(self):
        """Query within 0.5x-1.5x of range is clamped, not errored."""
        path = _create_lut(
            axis_values=[100, 200, 400, 800],
            grid_values=[1.0, 2.0, 4.0, 8.0],
        )
        lut = LUTLookup(path)
        # 60 is between 50 (0.5*100) and 100, so should clamp
        result = lut.lookup(seq_len=60)
        assert result == pytest.approx(1.0)  # Clamped to min

    def test_minor_extrapolation_above_clamped(self):
        """Query slightly above max is clamped."""
        path = _create_lut(
            axis_values=[100, 200, 400, 800],
            grid_values=[1.0, 2.0, 4.0, 8.0],
        )
        lut = LUTLookup(path)
        result = lut.lookup(seq_len=1000)  # 1000 < 1200 (1.5*800)
        assert result == pytest.approx(8.0)  # Clamped to max

    def test_exact_boundary_works(self):
        """Exact min/max values work fine."""
        path = _create_lut(
            axis_values=[100, 200, 400, 800],
            grid_values=[1.0, 2.0, 4.0, 8.0],
        )
        lut = LUTLookup(path)
        assert lut.lookup(seq_len=100) == pytest.approx(1.0)
        assert lut.lookup(seq_len=800) == pytest.approx(8.0)

    def test_interpolation_in_range(self):
        """Values within range interpolate correctly."""
        path = _create_lut(
            axis_values=[100, 200, 400, 800],
            grid_values=[1.0, 2.0, 4.0, 8.0],
        )
        lut = LUTLookup(path)
        result = lut.lookup(seq_len=150)
        # Linear interpolation between 100->1.0 and 200->2.0
        assert result == pytest.approx(1.5)
