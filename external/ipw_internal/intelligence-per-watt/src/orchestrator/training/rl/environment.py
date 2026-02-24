"""RL environment for orchestrator training with cached telemetry.

The environment simulates multi-turn tool usage using cached telemetry profiles
for fast training. During evaluation, real MCP servers are used for accurate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from orchestrator.data.episode_builder import Episode, Action, Observation
from orchestrator.data.telemetry_cache import TelemetryCache, hash_task
from orchestrator.data.mixed_dataset import UnifiedSample


@dataclass
class EpisodeState:
    """State during episode execution."""

    initial_prompt: str
    """Initial task/question"""

    history: List[Tuple[Action, Observation]] = field(default_factory=list)
    """History of (action, observation) pairs"""

    final_answer: Optional[str] = None
    """Final answer (set when is_final_answer action is taken)"""

    example_reasoning: Optional[str] = None
    """Example chain-of-thought reasoning (from GeneralThought dataset)"""

    source_dataset: str = ""
    """Source dataset name ('toolscale', 'generalthought', etc.)"""

    def add_turn(self, action: Action, observation: Observation):
        """Add a turn to the episode history."""
        self.history.append((action, observation))
        if action.is_final_answer:
            self.final_answer = observation.content

    def num_turns(self) -> int:
        """Return number of turns so far."""
        return len(self.history)

    def to_episode(self, task_id: str, ground_truth: str, correct: bool) -> Episode:
        """Convert state to Episode for reward computation.

        Args:
            task_id: Unique task identifier
            ground_truth: Ground truth answer
            correct: Whether final answer is correct

        Returns:
            Episode object
        """
        episode = Episode(
            task_id=task_id,
            initial_prompt=self.initial_prompt,
            ground_truth=ground_truth,
            final_answer=self.final_answer or "",
            correct=correct,
        )

        # Add all steps
        for action, observation in self.history:
            episode.add_step(action, observation)

        return episode


class OrchestratorEnvironment:
    """RL environment for training orchestrator with cached telemetry.

    Example:
        cache = TelemetryCache("data/telemetry_cache.db")
        env = OrchestratorEnvironment(
            telemetry_cache=cache,
            available_tools=["calculator", "ollama:llama3.2:1b", "openai:gpt-4o"],
            max_turns=10
        )

        # Reset for new episode
        state = env.reset(task="What is 2+2?")

        # Take action
        action = Action(
            thought="Use calculator",
            tool_name="calculator",
            tool_prompt="2+2",
            is_final_answer=True
        )
        state, observation = env.step(state, action)

        # Build episode for reward computation
        episode = state.to_episode(
            task_id="test",
            ground_truth="4",
            correct=(state.final_answer == "4")
        )
    """

    def __init__(
        self,
        telemetry_cache: TelemetryCache,
        available_tools: List[str],
        max_turns: int = 10,
        enable_sampling: bool = True,
    ):
        """Initialize environment.

        Args:
            telemetry_cache: Cache for retrieving telemetry profiles
            available_tools: List of available tool names
            max_turns: Maximum turns per episode
            enable_sampling: Whether to sample from profiles with noise (for diversity)
        """
        self.cache = telemetry_cache
        self.available_tools = available_tools
        self.max_turns = max_turns
        self.enable_sampling = enable_sampling

    def reset(self, task: Union[str, UnifiedSample]) -> EpisodeState:
        """Reset environment for a new episode.

        Args:
            task: Initial task/question (string) or UnifiedSample

        Returns:
            Initial state with optional reasoning context
        """
        if isinstance(task, UnifiedSample):
            return EpisodeState(
                initial_prompt=task.question,
                example_reasoning=task.reasoning,
                source_dataset=task.source,
            )
        else:
            return EpisodeState(initial_prompt=task)

    def step(
        self,
        state: EpisodeState,
        action: Action,
    ) -> Tuple[EpisodeState, Observation]:
        """Execute one step in the environment.

        Args:
            state: Current episode state
            action: Action to take

        Returns:
            Updated state and observation

        Raises:
            ValueError: If tool not available or max turns exceeded
        """
        # Check constraints
        if action.tool_name not in self.available_tools:
            raise ValueError(
                f"Tool {action.tool_name} not available. "
                f"Available: {self.available_tools}"
            )

        if state.num_turns() >= self.max_turns:
            raise ValueError(f"Max turns ({self.max_turns}) exceeded")

        # Get cached telemetry profile
        task_hash = hash_task(action.tool_prompt)
        profile = self.cache.get_profile(action.tool_name, task_hash)

        if not profile:
            # No cached profile - use default/fallback
            from orchestrator.data.telemetry_cache import TelemetryProfile

            profile = TelemetryProfile(
                tool_name=action.tool_name,
                task_hash=task_hash,
                avg_energy_joules=10.0,  # Default values
                avg_power_watts=100.0,
                avg_latency_seconds=1.0,
                avg_cost_usd=0.0,
                avg_tokens=50,
                stddev_energy=1.0,
                stddev_latency=0.1,
            )

        # Sample from profile if enabled (adds noise for diversity)
        if self.enable_sampling:
            profile = profile.sample()

        # Create observation
        # In training, we use placeholder content since we're using cached telemetry
        # In real execution (inference), MCP server provides actual content
        observation = Observation(
            content=f"[Simulated response from {action.tool_name}]",
            telemetry=profile,
        )

        # Update state
        state.add_turn(action, observation)

        return state, observation

    def is_done(self, state: EpisodeState) -> bool:
        """Check if episode is complete.

        Args:
            state: Current state

        Returns:
            True if episode should end (final answer given or max turns reached)
        """
        if state.final_answer is not None:
            return True

        if state.num_turns() >= self.max_turns:
            return True

        return False

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.available_tools.copy()


class OrchestratorEnvironmentReal:
    """Real environment using actual MCP servers (for evaluation).

    This variant executes actual MCP tools instead of using cached telemetry.
    Used during evaluation to get accurate metrics.
    """

    def __init__(
        self,
        mcp_tools: Dict,
        max_turns: int = 10,
    ):
        """Initialize real environment.

        Args:
            mcp_tools: Dictionary mapping tool names to MCP server instances
            max_turns: Maximum turns per episode
        """
        self.mcp_tools = mcp_tools
        self.max_turns = max_turns

    def reset(self, task: Union[str, UnifiedSample]) -> EpisodeState:
        """Reset for new episode.

        Args:
            task: Initial task/question (string) or UnifiedSample

        Returns:
            Initial state with optional reasoning context
        """
        if isinstance(task, UnifiedSample):
            return EpisodeState(
                initial_prompt=task.question,
                example_reasoning=task.reasoning,
                source_dataset=task.source,
            )
        else:
            return EpisodeState(initial_prompt=task)

    def step(
        self,
        state: EpisodeState,
        action: Action,
    ) -> Tuple[EpisodeState, Observation]:
        """Execute step using real MCP server.

        Args:
            state: Current state
            action: Action to take

        Returns:
            Updated state and observation
        """
        # Check constraints
        if action.tool_name not in self.mcp_tools:
            raise ValueError(f"Tool {action.tool_name} not available")

        if state.num_turns() >= self.max_turns:
            raise ValueError(f"Max turns ({self.max_turns}) exceeded")

        # Execute real MCP tool
        tool = self.mcp_tools[action.tool_name]
        result = tool.execute(action.tool_prompt)

        # Convert MCPToolResult to TelemetryProfile
        from orchestrator.data.telemetry_cache import TelemetryProfile

        # Compute energy from telemetry samples
        if result.telemetry_samples:
            energy_readings = [
                s.reading.energy_joules
                for s in result.telemetry_samples
                if hasattr(s.reading, "energy_joules") and s.reading.energy_joules
            ]
            avg_energy = (
                sum(energy_readings) / len(energy_readings) if energy_readings else 0.0
            )

            power_readings = [
                s.reading.power_watts
                for s in result.telemetry_samples
                if hasattr(s.reading, "power_watts") and s.reading.power_watts
            ]
            avg_power = (
                sum(power_readings) / len(power_readings) if power_readings else 0.0
            )
        else:
            avg_energy = 0.0
            avg_power = 0.0

        profile = TelemetryProfile(
            tool_name=action.tool_name,
            task_hash=hash_task(action.tool_prompt),
            avg_energy_joules=avg_energy,
            avg_power_watts=avg_power,
            avg_latency_seconds=result.latency_seconds,
            avg_cost_usd=result.cost_usd or 0.0,
            avg_tokens=result.usage.get("total_tokens", 0),
        )

        # Create observation with real content
        observation = Observation(
            content=result.content,
            telemetry=profile,
        )

        # Update state
        state.add_turn(action, observation)

        return state, observation

    def is_done(self, state: EpisodeState) -> bool:
        """Check if episode is done."""
        return (
            state.final_answer is not None or state.num_turns() >= self.max_turns
        )


# Example usage
if __name__ == "__main__":
    from orchestrator.data.telemetry_cache import TelemetryCache, TelemetryProfile, hash_task

    # Create cache and add sample profile
    cache = TelemetryCache("data/test_cache.db")
    profile = TelemetryProfile(
        tool_name="calculator",
        task_hash=hash_task("2+2"),
        avg_energy_joules=5.0,
        avg_power_watts=50.0,
        avg_latency_seconds=0.1,
        avg_cost_usd=0.0,
        avg_tokens=20,
    )
    cache.save_profile(profile)

    # Create environment
    env = OrchestratorEnvironment(
        telemetry_cache=cache,
        available_tools=["calculator"],
        max_turns=5,
    )

    # Run episode
    state = env.reset("What is 2+2?")
    print(f"Initial task: {state.initial_prompt}")

    action = Action(
        thought="Use calculator to compute 2+2",
        tool_name="calculator",
        tool_prompt="2+2",
        is_final_answer=True,
    )

    state, observation = env.step(state, action)
    print(f"Action: {action.tool_name}")
    print(f"Observation: {observation.content}")
    print(f"Energy: {observation.telemetry.avg_energy_joules}J")
    print(f"Done: {env.is_done(state)}")

    # Convert to episode
    episode = state.to_episode(task_id="test", ground_truth="4", correct=True)
    print(f"\nEpisode:")
    print(f"  Turns: {episode.num_turns()}")
    print(f"  Energy: {episode.total_energy_joules}J")
    print(f"  Latency: {episode.total_latency_seconds}s")
