"""Tests for tool distribution fitting and sampling."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestToolDistributionFitter:
    @pytest.fixture
    def _skip_no_scipy(self):
        pytest.importorskip("scipy")

    def test_fit_lognormal(self, _skip_no_scipy):
        """Test fitting a log-normal distribution to latency samples."""
        from inference_simulator.estimator.tool_distribution import ToolDistributionFitter

        rng = np.random.default_rng(42)
        # Generate log-normal samples
        raw_latencies = list(rng.lognormal(mean=-2.0, sigma=0.5, size=200))

        fitter = ToolDistributionFitter()
        result = fitter.fit(raw_latencies)

        assert "latency" in result
        assert result["latency"] is not None

    def test_fit_with_result_tokens(self, _skip_no_scipy):
        """Test fitting with both latency and result token samples."""
        from inference_simulator.estimator.tool_distribution import ToolDistributionFitter

        rng = np.random.default_rng(42)
        raw_latencies = list(rng.lognormal(mean=-2.0, sigma=0.5, size=100))
        raw_result_tokens = list(rng.integers(100, 2000, size=100))

        fitter = ToolDistributionFitter()
        result = fitter.fit(raw_latencies, raw_result_tokens)

        assert "latency" in result
        assert "result_tokens" in result

    def test_save_load_roundtrip(self, _skip_no_scipy):
        """Test pickle serialization round-trip."""
        from inference_simulator.estimator.tool_distribution import ToolDistributionFitter

        rng = np.random.default_rng(42)
        raw_latencies = list(rng.lognormal(mean=-2.0, sigma=0.5, size=100))

        fitter = ToolDistributionFitter()
        result = fitter.fit(raw_latencies)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "distributions.pkl"
            fitter.save({"test_tool": result}, path)
            assert path.exists()

            loaded = fitter.load(path)
            assert "test_tool" in loaded

    def test_fit_all_tools(self, _skip_no_scipy):
        """Test fitting distributions for multiple tools."""
        from inference_simulator.estimator.tool_distribution import ToolDistributionFitter

        rng = np.random.default_rng(42)
        raw_samples = {
            ("calculator", "default"): {
                "latencies": list(rng.lognormal(mean=-3.0, sigma=0.3, size=50)),
                "result_tokens": list(rng.integers(10, 100, size=50)),
            },
            ("web_search", "default"): {
                "latencies": list(rng.lognormal(mean=-1.0, sigma=0.5, size=50)),
                "result_tokens": list(rng.integers(500, 2000, size=50)),
            },
        }

        fitter = ToolDistributionFitter()
        results = fitter.fit_all_tools(agentic_csv_path=None, raw_samples=raw_samples)

        assert ("calculator", "default") in results
        assert ("web_search", "default") in results

    def test_sampling(self, _skip_no_scipy):
        """Test sampling from fitted distributions."""
        from inference_simulator.estimator.tool_distribution import ToolDistributionFitter

        rng = np.random.default_rng(42)
        raw_latencies = list(rng.lognormal(mean=-2.0, sigma=0.5, size=200))

        fitter = ToolDistributionFitter()
        result = fitter.fit(raw_latencies)

        # Sample from the fitted distribution
        dist_info = result["latency"]
        if hasattr(dist_info, "rvs"):
            samples = dist_info.rvs(size=100)
            assert len(samples) == 100
            assert all(s > 0 for s in samples)
