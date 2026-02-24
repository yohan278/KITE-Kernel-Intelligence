"""Test data pipeline end-to-end."""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.data import (
    TelemetryCache,
    TelemetryProfile,
    hash_task,
    Episode,
    EpisodeBuilder,
    Action,
    Observation,
)


def test_telemetry_cache():
    """Test telemetry cache creation and retrieval."""
    print("Testing TelemetryCache...")

    # Create temporary cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = TelemetryCache(cache_path)

        # Save profile
        profile = TelemetryProfile(
            tool_name="test_tool",
            task_hash=hash_task("test task"),
            avg_energy_joules=10.5,
            avg_power_watts=100.0,
            avg_latency_seconds=0.5,
            avg_cost_usd=0.001,
            avg_tokens=50,
            stddev_energy=1.0,
            stddev_latency=0.05,
        )
        cache.save_profile(profile)

        # Retrieve profile
        retrieved = cache.get_profile("test_tool", hash_task("test task"))

        assert retrieved is not None, "Profile should be retrieved"
        assert retrieved.avg_energy_joules == 10.5, "Energy should match"
        assert retrieved.avg_latency_seconds == 0.5, "Latency should match"
        assert retrieved.avg_cost_usd == 0.001, "Cost should match"

        print("  ✓ Save and retrieve profile")

        # Test coverage
        coverage = cache.get_coverage()
        assert "test_tool" in coverage, "Tool should be in coverage"
        assert coverage["test_tool"] == 1, "Should have 1 profile"

        print("  ✓ Coverage statistics")

        # Test statistics
        stats = cache.export_statistics()
        assert stats["total_profiles"] == 1, "Should have 1 total profile"
        assert stats["num_tools"] == 1, "Should have 1 tool"

        print("  ✓ Export statistics")

    print("✅ TelemetryCache tests passed!\n")


def test_hash_task():
    """Test task hashing."""
    print("Testing hash_task...")

    # Same task should produce same hash
    hash1 = hash_task("What is 2+2?")
    hash2 = hash_task("What is 2+2?")
    assert hash1 == hash2, "Same task should produce same hash"
    print("  ✓ Consistent hashing")

    # Different tasks should produce different hashes
    hash3 = hash_task("What is 3+3?")
    assert hash1 != hash3, "Different tasks should produce different hashes"
    print("  ✓ Different hashes for different tasks")

    # Hash should be 16 characters (truncated SHA256)
    assert len(hash1) == 16, "Hash should be 16 characters"
    print("  ✓ Hash length")

    print("✅ hash_task tests passed!\n")


def test_episode_building():
    """Test episode building."""
    print("Testing Episode and EpisodeBuilder...")

    # Create temporary cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = TelemetryCache(cache_path)

        # Add sample profile
        profile = TelemetryProfile(
            tool_name="calculator",
            task_hash=hash_task("What is 2+2?"),
            avg_energy_joules=5.0,
            avg_power_watts=50.0,
            avg_latency_seconds=0.1,
            avg_cost_usd=0.0,
            avg_tokens=20,
            stddev_energy=0.5,
            stddev_latency=0.01,
        )
        cache.save_profile(profile)

        # Create episode manually
        episode = Episode(
            task_id="test_1",
            initial_prompt="What is 2+2?",
            ground_truth="4",
        )

        # Add action and observation
        action = Action(
            thought="Use calculator to compute",
            tool_name="calculator",
            tool_prompt="2+2",
            is_final_answer=True,
        )

        observation = Observation(
            content="4",
            telemetry=profile,
        )

        episode.add_step(action, observation)
        episode.final_answer = "4"
        episode.correct = True

        # Check episode
        assert episode.num_turns() == 1, "Should have 1 turn"
        assert episode.total_energy_joules == 5.0, "Energy should match"
        assert episode.total_latency_seconds == 0.1, "Latency should match"
        assert episode.correct, "Should be correct"

        print("  ✓ Manual episode creation")

        # Test to_dict
        episode_dict = episode.to_dict()
        assert episode_dict["task_id"] == "test_1"
        assert episode_dict["correct"] is True
        assert episode_dict["num_turns"] == 1

        print("  ✓ Episode serialization")

    print("✅ Episode building tests passed!\n")


def test_profile_sampling():
    """Test telemetry profile sampling with noise."""
    print("Testing TelemetryProfile.sample()...")

    profile = TelemetryProfile(
        tool_name="test",
        task_hash="abc123",
        avg_energy_joules=10.0,
        avg_power_watts=100.0,
        avg_latency_seconds=0.5,
        avg_cost_usd=0.001,
        avg_tokens=50,
        stddev_energy=1.0,
        stddev_latency=0.05,
    )

    # Sample multiple times
    samples = [profile.sample() for _ in range(10)]

    # Check that samples have variation
    energies = [s.avg_energy_joules for s in samples]
    assert max(energies) != min(energies), "Samples should vary"

    # Check that values are non-negative
    assert all(e >= 0 for e in energies), "Energy should be non-negative"

    # Cost should remain constant
    costs = [s.avg_cost_usd for s in samples]
    assert all(c == 0.001 for c in costs), "Cost should be deterministic"

    print("  ✓ Sampling with noise")
    print(f"    Energy range: {min(energies):.2f} - {max(energies):.2f}J")

    print("✅ Profile sampling tests passed!\n")


def main():
    """Run all data pipeline tests."""
    print("=" * 70)
    print("Data Pipeline Tests")
    print("=" * 70)
    print()

    try:
        test_hash_task()
        test_telemetry_cache()
        test_profile_sampling()
        test_episode_building()

        print("=" * 70)
        print("✅ ALL DATA PIPELINE TESTS PASSED")
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
