"""Tests for FittedDistribution type."""

import numpy as np
import pytest

from inference_simulator.types.fitted_distribution import FittedDistribution


class TestFit:
    """Tests for FittedDistribution.fit()."""

    def test_fit_lognormal_data(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(mean=1.0, sigma=0.5, size=500)
        fd = FittedDistribution.fit(data)
        assert fd.dist_name == "lognormal"
        assert fd.n_samples == 500
        assert fd.ks_statistic < 0.1
        assert "shape" in fd.params
        assert "scale" in fd.params

    def test_fit_gamma_data(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=2.0, scale=3.0, size=500)
        fd = FittedDistribution.fit(data)
        assert fd.dist_name in ("gamma", "lognormal")
        assert fd.n_samples == 500
        assert fd.ks_statistic < 0.1

    def test_fit_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=5.0, size=500)
        fd = FittedDistribution.fit(data)
        # Normal data may fit best to normal or another dist
        assert fd.dist_name in ("normal", "lognormal", "gamma")
        assert fd.n_samples == 500

    def test_fit_with_specified_candidates(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(mean=1.0, sigma=0.5, size=200)
        fd = FittedDistribution.fit(data, candidates=("gamma",))
        assert fd.dist_name == "gamma"

    def test_empirical_fallback_small_samples(self):
        data = [1.0, 2.5, 3.7, 5.1]
        fd = FittedDistribution.fit(data)
        assert fd.dist_name == "empirical"
        assert fd.n_samples == 4
        assert fd.empirical_samples is not None
        assert len(fd.empirical_samples) == 4

    def test_fit_empty_data(self):
        fd = FittedDistribution.fit([])
        assert fd.dist_name == "empirical"
        assert fd.n_samples == 0


class TestSample:
    """Tests for FittedDistribution.sample()."""

    def test_sample_produces_valid_values(self):
        rng_fit = np.random.default_rng(42)
        data = rng_fit.lognormal(mean=1.0, sigma=0.5, size=200)
        fd = FittedDistribution.fit(data)

        rng_sample = np.random.default_rng(99)
        samples = fd.sample(rng_sample, size=50)
        assert samples.shape == (50,)
        assert np.all(np.isfinite(samples))

    def test_sample_empirical(self):
        fd = FittedDistribution(
            dist_name="empirical",
            n_samples=3,
            mean=2.0,
            std=1.0,
            empirical_samples=[1.0, 2.0, 3.0],
        )
        rng = np.random.default_rng(42)
        samples = fd.sample(rng, size=100)
        assert samples.shape == (100,)
        # All samples should come from the stored values
        assert set(samples.tolist()).issubset({1.0, 2.0, 3.0})

    def test_sample_normal(self):
        fd = FittedDistribution(
            dist_name="normal",
            params={"loc": 10.0, "scale": 1.0},
            n_samples=100,
            mean=10.0,
            std=1.0,
        )
        rng = np.random.default_rng(42)
        samples = fd.sample(rng, size=1000)
        assert samples.shape == (1000,)
        assert abs(np.mean(samples) - 10.0) < 0.5

    def test_sample_empirical_empty_falls_back_to_mean(self):
        fd = FittedDistribution(
            dist_name="empirical",
            n_samples=0,
            mean=5.0,
            empirical_samples=[],
        )
        rng = np.random.default_rng(42)
        samples = fd.sample(rng, size=3)
        assert np.all(samples == 5.0)


class TestSerialization:
    """Tests for to_dict / from_dict round-trip."""

    def test_to_dict_from_dict_roundtrip(self):
        fd = FittedDistribution(
            dist_name="lognormal",
            params={"shape": 0.5, "loc": 0.0, "scale": 2.7},
            ks_statistic=0.03,
            ks_pvalue=0.85,
            n_samples=200,
            mean=3.1,
            std=1.7,
        )
        d = fd.to_dict()
        fd2 = FittedDistribution.from_dict(d)
        assert fd2.dist_name == fd.dist_name
        assert fd2.params == fd.params
        assert fd2.ks_statistic == fd.ks_statistic
        assert fd2.ks_pvalue == fd.ks_pvalue
        assert fd2.n_samples == fd.n_samples
        assert fd2.mean == fd.mean
        assert fd2.std == fd.std
        assert fd2.empirical_samples is None

    def test_to_dict_from_dict_with_empirical_samples(self):
        fd = FittedDistribution(
            dist_name="empirical",
            n_samples=5,
            mean=3.0,
            std=1.0,
            empirical_samples=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        d = fd.to_dict()
        assert "empirical_samples" in d
        fd2 = FittedDistribution.from_dict(d)
        assert fd2.empirical_samples == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_to_dict_omits_empirical_when_none(self):
        fd = FittedDistribution(dist_name="gamma", params={"shape": 2.0, "loc": 0.0, "scale": 1.0})
        d = fd.to_dict()
        assert "empirical_samples" not in d
