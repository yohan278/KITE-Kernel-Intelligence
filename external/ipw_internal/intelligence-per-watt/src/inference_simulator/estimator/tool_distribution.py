"""Fit parametric distributions to raw tool latency and result_token samples."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class ToolDistributionFitter:
    """Fits parametric distributions to raw tool latency and result_token samples."""

    def fit(
        self,
        raw_latencies: Sequence[float],
        raw_result_tokens: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Try LogNormal, Gamma, empirical CDF; pick best fit via KS test.

        Args:
            raw_latencies: Raw tool latency samples (seconds).
            raw_result_tokens: Optional raw result token count samples.

        Returns:
            Dict with best-fit distribution params for latency (and optionally tokens).
        """
        import numpy as np
        from scipy import stats

        latencies = np.array(raw_latencies, dtype=np.float64)
        latencies = latencies[latencies > 0]  # Filter non-positive

        if len(latencies) < 3:
            return {
                "latency": {
                    "distribution": "empirical",
                    "samples": latencies.tolist(),
                    "mean": float(np.mean(latencies)) if len(latencies) > 0 else 0.0,
                    "std": float(np.std(latencies)) if len(latencies) > 0 else 0.0,
                }
            }

        result: Dict[str, Any] = {}

        # Fit latency distribution
        result["latency"] = self._fit_best(latencies, stats)

        # Fit result_tokens distribution if provided
        if raw_result_tokens is not None:
            tokens = np.array(raw_result_tokens, dtype=np.float64)
            tokens = tokens[tokens > 0]
            if len(tokens) >= 3:
                result["result_tokens"] = self._fit_best(tokens, stats)
            elif len(tokens) > 0:
                result["result_tokens"] = {
                    "distribution": "empirical",
                    "samples": tokens.tolist(),
                    "mean": float(np.mean(tokens)),
                    "std": float(np.std(tokens)),
                }

        return result

    def _fit_best(self, data: Any, stats: Any) -> Dict[str, Any]:
        """Fit LogNormal and Gamma, pick best via KS test."""
        import numpy as np

        candidates: List[Dict[str, Any]] = []

        # LogNormal
        try:
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            ks_stat, ks_p = stats.kstest(data, "lognorm", args=(shape, loc, scale))
            candidates.append({
                "distribution": "lognormal",
                "params": {"shape": float(shape), "loc": float(loc), "scale": float(scale)},
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            })
        except Exception:
            pass

        # Gamma
        try:
            shape, loc, scale = stats.gamma.fit(data, floc=0)
            ks_stat, ks_p = stats.kstest(data, "gamma", args=(shape, loc, scale))
            candidates.append({
                "distribution": "gamma",
                "params": {"shape": float(shape), "loc": float(loc), "scale": float(scale)},
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            })
        except Exception:
            pass

        if not candidates:
            return {
                "distribution": "empirical",
                "samples": data.tolist(),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
            }

        # Pick candidate with lowest KS statistic (best fit)
        best = min(candidates, key=lambda c: c["ks_statistic"])
        best["mean"] = float(np.mean(data))
        best["std"] = float(np.std(data))
        best["n_samples"] = len(data)
        return best

    def fit_all_tools(
        self,
        agentic_csv_path: Optional[Path] = None,
        raw_samples: Optional[Dict[Any, Any]] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Fit distributions for all (tool_type, config) pairs.

        Args:
            agentic_csv_path: Path to agentic_tool.csv with columns including
                tool_type, time_s, and optionally result_tokens. Can be None
                when raw_samples is provided.
            raw_samples: Optional pre-loaded samples. Values can be either:
                - list of floats (latency samples), or
                - dict with ``"latencies"`` and optional ``"result_tokens"`` keys.

        Returns:
            Dict mapping tool key -> fitted distribution info.
        """
        tool_latencies: Dict[Any, List[float]] = {}
        tool_tokens: Dict[Any, List[float]] = {}

        if raw_samples is not None:
            for key, samples in raw_samples.items():
                if isinstance(samples, dict):
                    # Dict with "latencies" and optional "result_tokens"
                    tool_latencies[key] = list(samples.get("latencies", []))
                    if "result_tokens" in samples:
                        tool_tokens[key] = [float(t) for t in samples["result_tokens"]]
                else:
                    tool_latencies[key] = list(samples)
        elif agentic_csv_path is not None and Path(agentic_csv_path).exists():
            with open(Path(agentic_csv_path), "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tool_type = row.get("tool_type", row.get("operator_name", "unknown"))
                    config = row.get("config", "")
                    key = f"{tool_type}:{config}" if config else tool_type

                    try:
                        latency = float(row["time_s"])
                        tool_latencies.setdefault(key, []).append(latency)
                    except (KeyError, ValueError):
                        continue

                    result_tok = row.get("result_tokens")
                    if result_tok:
                        try:
                            tool_tokens.setdefault(key, []).append(float(result_tok))
                        except (ValueError, TypeError):
                            pass

        results: Dict[Any, Dict[str, Any]] = {}
        for key, latencies in tool_latencies.items():
            tokens = tool_tokens.get(key)
            results[key] = self.fit(latencies, [float(t) for t in tokens] if tokens else None)

        return results

    def save(self, distributions: Dict[str, Dict[str, Any]], path: Path) -> Path:
        """Pickle fitted distributions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(distributions, f)
        return path

    def load(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Load fitted distributions."""
        with open(path, "rb") as f:
            return pickle.load(f)
