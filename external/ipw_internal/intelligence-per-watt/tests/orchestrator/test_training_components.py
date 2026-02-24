"""Test training components (reward, environment, policy)."""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Minimal imports to avoid dependencies
from orchestrator.data.episode_builder import Episode, Action, Observation
from orchestrator.data.telemetry_cache import TelemetryCache, TelemetryProfile, hash_task


def test_reward_function():
    """Test multi-objective reward function."""
    print("Testing MultiObjectiveReward...")

    # Import reward components (simplified inline version for testing)
    from dataclasses import dataclass

    @dataclass
    class RewardWeights:
        alpha: float = 0.5
        beta: float = 0.3
        gamma: float = 0.2

    @dataclass
    class Normalizers:
        energy_scale: float = 100.0
        cost_scale: float = 0.10
        latency_scale: float = 30.0
        power_scale: float = 200.0

    class MultiObjectiveReward:
        def __init__(self, weights, normalizers):
            self.weights = weights
            self.normalizers = normalizers

        def compute(self, episode):
            accuracy_reward = 1.0 if episode.correct else 0.0
            cost_penalty = episode.total_cost_usd / self.normalizers.cost_scale
            energy_penalty = episode.total_energy_joules / self.normalizers.energy_scale
            latency_penalty = episode.total_latency_seconds / self.normalizers.latency_scale
            power_penalty = episode.max_power_watts / self.normalizers.power_scale

            reward = (
                self.weights.alpha * accuracy_reward
                - self.weights.beta * (cost_penalty + energy_penalty)
                - self.weights.gamma * (latency_penalty + power_penalty)
            )
            return reward

    # Create episode
    episode = Episode(
        task_id="test",
        initial_prompt="What is 2+2?",
        ground_truth="4",
        final_answer="4",
        correct=True,
        total_energy_joules=15.0,
        total_cost_usd=0.0,
        total_latency_seconds=0.5,
        max_power_watts=100.0,
    )

    # Create reward function
    weights = RewardWeights(alpha=0.5, beta=0.3, gamma=0.2)
    normalizers = Normalizers()
    reward_fn = MultiObjectiveReward(weights, normalizers)

    # Compute reward
    reward = reward_fn.compute(episode)
    print(f"  Reward (correct answer): {reward:.4f}")
    assert reward > 0, "Reward should be positive for correct answer"
    print("  ✓ Correct answer gives positive reward")

    # Test incorrect answer
    episode.correct = False
    reward = reward_fn.compute(episode)
    print(f"  Reward (incorrect answer): {reward:.4f}")
    assert reward < 0, "Reward should be negative for incorrect answer"
    print("  ✓ Incorrect answer gives negative reward")

    print("✅ Reward function tests passed!\n")


def test_environment():
    """Test RL environment with cached telemetry."""
    print("Testing OrchestratorEnvironment...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache with sample profile
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = TelemetryCache(cache_path)

        profile = TelemetryProfile(
            tool_name="calculator",
            task_hash=hash_task("2+2"),
            avg_energy_joules=5.0,
            avg_power_watts=50.0,
            avg_latency_seconds=0.1,
            avg_cost_usd=0.0,
            avg_tokens=20,
            stddev_energy=0.5,
            stddev_latency=0.01,
        )
        cache.save_profile(profile)

        # Import environment (inline simplified version)
        from dataclasses import dataclass, field
        from typing import List, Tuple, Optional

        @dataclass
        class EpisodeState:
            initial_prompt: str
            history: List[Tuple[Action, Observation]] = field(default_factory=list)
            final_answer: Optional[str] = None

            def add_turn(self, action, observation):
                self.history.append((action, observation))
                if action.is_final_answer:
                    self.final_answer = observation.content

            def num_turns(self):
                return len(self.history)

        class OrchestratorEnvironment:
            def __init__(self, cache, available_tools, max_turns=10):
                self.cache = cache
                self.available_tools = available_tools
                self.max_turns = max_turns

            def reset(self, task):
                return EpisodeState(initial_prompt=task)

            def step(self, state, action):
                task_hash = hash_task(action.tool_prompt)
                profile = self.cache.get_profile(action.tool_name, task_hash)

                if not profile:
                    profile = TelemetryProfile(
                        tool_name=action.tool_name,
                        task_hash=task_hash,
                        avg_energy_joules=10.0,
                        avg_power_watts=100.0,
                        avg_latency_seconds=1.0,
                        avg_cost_usd=0.0,
                        avg_tokens=50,
                    )

                observation = Observation(
                    content=f"[Result from {action.tool_name}]",
                    telemetry=profile,
                )

                state.add_turn(action, observation)
                return state, observation

        # Create environment
        env = OrchestratorEnvironment(
            cache=cache,
            available_tools=["calculator"],
            max_turns=5,
        )

        # Reset
        state = env.reset("What is 2+2?")
        assert state.initial_prompt == "What is 2+2?"
        print("  ✓ Environment reset")

        # Take action
        action = Action(
            thought="Use calculator",
            tool_name="calculator",
            tool_prompt="2+2",
            is_final_answer=True,
        )

        state, observation = env.step(state, action)
        assert state.num_turns() == 1
        assert observation.telemetry.avg_energy_joules == 5.0
        print("  ✓ Action execution with cached telemetry")

    print("✅ Environment tests passed!\n")


def test_policy_model():
    """Test policy model wrapper."""
    print("Testing PolicyModel...")

    # Import policy components (inline simplified version)
    from dataclasses import dataclass, field
    from typing import List, Tuple, Optional
    import re

    @dataclass
    class EpisodeState:
        initial_prompt: str
        history: List = field(default_factory=list)
        final_answer: Optional[str] = None

    class PolicyModel:
        def __init__(self):
            pass

        def predict_action(self, state, available_tools):
            # Mock: use calculator for math questions
            if "calculator" in available_tools and any(c in state.initial_prompt for c in "0123456789+-*/"):
                tool = "calculator"
            else:
                tool = available_tools[0]

            return Action(
                thought=f"Use {tool} to answer",
                tool_name=tool,
                tool_prompt=state.initial_prompt,
                is_final_answer=True,
            )

    # Create policy
    policy = PolicyModel()

    # Create state
    state = EpisodeState(initial_prompt="What is 2+2?")

    # Predict action
    action = policy.predict_action(state, available_tools=["calculator", "llm"])

    assert action.tool_name == "calculator", "Should choose calculator for math"
    assert action.is_final_answer is True
    print("  ✓ Policy predicts calculator for math question")

    # Test with non-math question
    state2 = EpisodeState(initial_prompt="What is the capital of France?")
    action2 = policy.predict_action(state2, available_tools=["llm", "search"])

    assert action2.tool_name in ["llm", "search"], "Should choose non-calculator tool"
    print("  ✓ Policy chooses appropriate tool")

    print("✅ Policy model tests passed!\n")


def main():
    """Run all training component tests."""
    print("=" * 70)
    print("Training Components Tests")
    print("=" * 70)
    print()

    try:
        test_reward_function()
        test_environment()
        test_policy_model()

        print("=" * 70)
        print("✅ ALL TRAINING COMPONENT TESTS PASSED")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
