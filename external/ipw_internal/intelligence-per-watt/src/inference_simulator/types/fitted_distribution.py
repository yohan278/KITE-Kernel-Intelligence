"""Fitted parametric distribution for workload characterization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class FittedDistribution:
    """A fitted parametric distribution with serialization and sampling support.

    Wraps the result of fitting candidate distributions (lognormal, gamma, normal)
    to empirical data, selecting the best fit via Kolmogorov-Smirnov test.
    For small sample sizes (<10), stores raw samples for empirical resampling.

    Attributes:
        dist_name: Distribution family name ("lognormal", "gamma", "normal", "empirical").
        params: Distribution parameters (shape, loc, scale).
        ks_statistic: KS test statistic (lower is better fit).
        ks_pvalue: KS test p-value.
        n_samples: Number of data points used for fitting.
        mean: Sample mean of the original data.
        std: Sample standard deviation of the original data.
        empirical_samples: Raw samples stored when n_samples < 10 (empirical fallback).
    """

    dist_name: str
    params: Dict[str, float] = field(default_factory=dict)
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0
    n_samples: int = 0
    mean: float = 0.0
    std: float = 0.0
    empirical_samples: Optional[List[float]] = None

    def sample(self, rng: "np.random.Generator", size: int = 1) -> "np.ndarray":
        """Draw samples from the fitted distribution.

        Args:
            rng: NumPy random generator instance.
            size: Number of samples to draw.

        Returns:
            Array of sampled values.
        """
        import numpy as np

        if self.dist_name == "empirical":
            if self.empirical_samples is None or len(self.empirical_samples) == 0:
                return np.full(size, self.mean)
            indices = rng.integers(0, len(self.empirical_samples), size=size)
            return np.array([self.empirical_samples[i] for i in indices])

        from scipy import stats as sp_stats

        dist_map = {
            "lognormal": sp_stats.lognorm,
            "gamma": sp_stats.gamma,
            "normal": sp_stats.norm,
        }
        sp_dist = dist_map.get(self.dist_name)
        if sp_dist is None:
            return np.full(size, self.mean)

        if self.dist_name == "normal":
            frozen = sp_dist(
                loc=self.params.get("loc", 0.0),
                scale=self.params.get("scale", 1.0),
            )
        else:
            frozen = sp_dist(
                self.params.get("shape", 1.0),
                loc=self.params.get("loc", 0.0),
                scale=self.params.get("scale", 1.0),
            )
        return frozen.rvs(size=size, random_state=rng)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {
            "dist_name": self.dist_name,
            "params": dict(self.params),
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "n_samples": self.n_samples,
            "mean": self.mean,
            "std": self.std,
        }
        if self.empirical_samples is not None:
            d["empirical_samples"] = list(self.empirical_samples)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FittedDistribution:
        """Reconstruct a FittedDistribution from a dictionary."""
        return cls(
            dist_name=d["dist_name"],
            params=dict(d.get("params", {})),
            ks_statistic=d.get("ks_statistic", 0.0),
            ks_pvalue=d.get("ks_pvalue", 0.0),
            n_samples=d.get("n_samples", 0),
            mean=d.get("mean", 0.0),
            std=d.get("std", 0.0),
            empirical_samples=d.get("empirical_samples"),
        )

    @classmethod
    def fit(
        cls,
        data: Sequence[float],
        candidates: Tuple[str, ...] = ("lognormal", "gamma", "normal"),
    ) -> FittedDistribution:
        """Fit the best distribution to data using the Kolmogorov-Smirnov test.

        For fewer than 10 samples, returns an empirical distribution that stores
        the raw samples for resampling.

        Args:
            data: Sequence of observed values to fit.
            candidates: Distribution families to try.

        Returns:
            FittedDistribution with the best-fit parameters.
        """
        import numpy as np
        from scipy import stats as sp_stats

        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            return cls(dist_name="empirical", n_samples=0)

        sample_mean = float(np.mean(arr))
        sample_std = float(np.std(arr))

        if len(arr) < 10:
            return cls(
                dist_name="empirical",
                n_samples=len(arr),
                mean=sample_mean,
                std=sample_std,
                empirical_samples=arr.tolist(),
            )

        dist_map = {
            "lognormal": ("lognorm", sp_stats.lognorm),
            "gamma": ("gamma", sp_stats.gamma),
            "normal": ("norm", sp_stats.norm),
        }

        fits: List[Dict[str, Any]] = []
        for name in candidates:
            entry = dist_map.get(name)
            if entry is None:
                continue
            scipy_name, sp_dist = entry
            try:
                if name == "normal":
                    loc, scale = sp_dist.fit(arr)
                    params = {"loc": float(loc), "scale": float(scale)}
                    ks_stat, ks_p = sp_stats.kstest(arr, scipy_name, args=(loc, scale))
                else:
                    # Positive-support distributions: fix loc=0
                    positive = arr[arr > 0]
                    if len(positive) < 3:
                        continue
                    shape, loc, scale = sp_dist.fit(positive, floc=0)
                    params = {
                        "shape": float(shape),
                        "loc": float(loc),
                        "scale": float(scale),
                    }
                    ks_stat, ks_p = sp_stats.kstest(
                        positive, scipy_name, args=(shape, loc, scale)
                    )
                fits.append(
                    {
                        "dist_name": name,
                        "params": params,
                        "ks_statistic": float(ks_stat),
                        "ks_pvalue": float(ks_p),
                    }
                )
            except Exception:
                continue

        if not fits:
            return cls(
                dist_name="empirical",
                n_samples=len(arr),
                mean=sample_mean,
                std=sample_std,
                empirical_samples=arr.tolist(),
            )

        best = min(fits, key=lambda f: f["ks_statistic"])
        return cls(
            dist_name=best["dist_name"],
            params=best["params"],
            ks_statistic=best["ks_statistic"],
            ks_pvalue=best["ks_pvalue"],
            n_samples=len(arr),
            mean=sample_mean,
            std=sample_std,
        )


__all__ = ["FittedDistribution"]
