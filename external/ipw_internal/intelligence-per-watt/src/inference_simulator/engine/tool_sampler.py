"""Tool latency sampler for multi-step inference simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


# Tool types that use Docker container pools (stateful Markov model)
_CONTAINER_TOOL_TYPES = {"code_interpreter", "docker_exec"}

# Tool types with DNS/CDN warm-up (exponential decay)
_WEB_TOOL_TYPES = {"web_search"}


class ContainerPoolState:
    """Stateful Markov model for Docker container pool warm/cold starts.

    Tracks warm container count and last call time. If the pool has warm
    containers and the cooldown has not expired, samples from the warm
    (low-latency) distribution. Otherwise performs a cold start.
    """

    def __init__(
        self,
        max_pool_size: int = 5,
        cooldown_time_s: float = 60.0,
        warm_mean_s: float = 0.05,
        warm_std_s: float = 0.02,
        cold_mean_s: float = 2.0,
        cold_std_s: float = 1.0,
    ) -> None:
        self.warm_containers = 0
        self.max_pool_size = max_pool_size
        self.cooldown_time_s = cooldown_time_s
        self.warm_mean_s = warm_mean_s
        self.warm_std_s = warm_std_s
        self.cold_mean_s = cold_mean_s
        self.cold_std_s = cold_std_s
        self._last_call_time_s: float = 0.0

    def sample_latency(self, current_time_s: float, rng: Any) -> float:
        """Sample latency based on container pool state.

        Args:
            current_time_s: Current simulation time in seconds.
            rng: numpy random generator.

        Returns:
            Latency in seconds (minimum 0.001).
        """
        time_since_last = current_time_s - self._last_call_time_s
        self._last_call_time_s = current_time_s

        if time_since_last < self.cooldown_time_s and self.warm_containers > 0:
            # Warm container hit
            return max(0.001, float(rng.normal(self.warm_mean_s, self.warm_std_s)))
        else:
            # Cold start — add a warm container to the pool
            self.warm_containers = min(self.warm_containers + 1, self.max_pool_size)
            return max(0.001, float(rng.normal(self.cold_mean_s, self.cold_std_s)))


class ToolLatencySampler:
    """Samples tool latencies from fitted distributions during simulation.

    Uses stateful Markov models for container-based tools (code_interpreter,
    docker_exec) and DNS/CDN warm-up for web search tools. Simple tools
    (calculator, file I/O) use i.i.d. sampling from fitted or default
    distributions.

    If fitted distributions are available (from Pipeline #1), uses them.
    Otherwise falls back to default distributions.
    """

    def __init__(self, distributions_path: Optional[Path] = None) -> None:
        self._dists: Dict[str, Any] = {}
        if distributions_path and distributions_path.exists():
            import pickle

            with open(distributions_path, "rb") as f:
                self._dists = pickle.load(f)

        # Stateful models for container-based tools
        self._container_pools: Dict[str, ContainerPoolState] = {}
        # DNS/CDN warm-up cache for web search
        self._dns_cache: Dict[str, float] = {}

    def sample_latency(
        self,
        tool_type: str,
        tool_config: str,
        rng: Any,
        current_time_s: float = 0.0,
    ) -> float:
        """Sample latency in seconds from the appropriate model.

        Container-based tools (code_interpreter, docker_exec) use a stateful
        Markov model that tracks warm/cold container state. Web search tools
        use DNS/CDN warm-up with exponential decay. All other tools use i.i.d.
        sampling from fitted or default distributions.

        Args:
            tool_type: Tool type identifier.
            tool_config: Tool configuration string.
            rng: numpy random generator.
            current_time_s: Current simulation time in seconds (used for
                stateful models).

        Returns:
            Latency in seconds (minimum 0.001).
        """
        # Container-based tools: Markov model with warm/cold starts
        if tool_type in _CONTAINER_TOOL_TYPES:
            pool_key = f"{tool_type}:{tool_config}"
            if pool_key not in self._container_pools:
                self._container_pools[pool_key] = ContainerPoolState()
            return self._container_pools[pool_key].sample_latency(current_time_s, rng)

        # Web search: DNS/CDN warm-up (exponential decay after first call)
        if tool_type in _WEB_TOOL_TYPES:
            cache_key = f"{tool_type}:{tool_config}"
            if cache_key not in self._dns_cache:
                # First call: cold DNS lookup
                self._dns_cache[cache_key] = current_time_s
                base_latency = float(max(0.001, rng.exponential(0.5)))
                return base_latency
            else:
                # Warm: latency decays exponentially with time since first call
                time_since_first = current_time_s - self._dns_cache[cache_key]
                # Decay from 0.5s mean to 0.05s mean over ~30s
                decay = max(0.05, 0.5 * (0.9 ** time_since_first))
                return float(max(0.001, rng.exponential(decay)))

        # Simple tools: i.i.d. from fitted distribution or default
        key = f"{tool_type}:{tool_config}"
        dist = self._dists.get(key)
        if dist is not None and "latency" in dist:
            params = dist["latency"]
            return float(max(0.001, rng.exponential(params.get("mean", 0.1))))
        return float(max(0.001, rng.exponential(0.1)))

    def sample_result_tokens(self, tool_type: str, tool_config: str, rng: Any) -> int:
        """Sample result token count. Falls back to default 512 tokens."""
        key = f"{tool_type}:{tool_config}"
        dist = self._dists.get(key)
        if dist is not None and "result_tokens" in dist:
            params = dist["result_tokens"]
            mean = params.get("mean", 512)
            std = params.get("std", 128)
            return max(1, int(rng.normal(mean, std)))
        return 512

    def has_distribution(self, tool_type: str, tool_config: str) -> bool:
        """Check if we have fitted distributions for this tool."""
        key = f"{tool_type}:{tool_config}"
        return key in self._dists
