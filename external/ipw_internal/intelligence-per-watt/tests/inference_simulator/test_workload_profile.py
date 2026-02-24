"""Tests for WorkloadProfile type."""

import json

import pytest

from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


def _make_fd(dist_name="lognormal", mean=5.0) -> FittedDistribution:
    """Helper to create a simple FittedDistribution for testing."""
    return FittedDistribution(
        dist_name=dist_name,
        params={"shape": 0.5, "loc": 0.0, "scale": 2.0},
        ks_statistic=0.04,
        ks_pvalue=0.8,
        n_samples=100,
        mean=mean,
        std=1.5,
    )


class TestSaveLoad:
    """Tests for WorkloadProfile save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        profile = WorkloadProfile(
            workload_type="chat",
            source_dataset="wildchat",
            n_samples=1000,
            system_prompt_tokens=128,
            structured_output_fraction=0.1,
            turns_or_steps_dist=_make_fd(mean=3.2),
            input_tokens_dist=_make_fd(mean=500.0),
            thinking_tokens_dist=None,
            answer_tokens_dist=_make_fd(mean=200.0),
            tool_call_probability=0.05,
            tool_type_distribution={"web_search": 0.6, "calculator": 0.4},
            domain_mix={"general": 0.7, "technical": 0.3},
            max_context_observed=32768,
        )

        path = tmp_path / "profile.json"
        profile.save(path)

        loaded = WorkloadProfile.load(path)
        assert loaded.workload_type == "chat"
        assert loaded.source_dataset == "wildchat"
        assert loaded.n_samples == 1000
        assert loaded.system_prompt_tokens == 128
        assert loaded.structured_output_fraction == 0.1
        assert loaded.turns_or_steps_dist is not None
        assert loaded.turns_or_steps_dist.mean == 3.2
        assert loaded.input_tokens_dist is not None
        assert loaded.thinking_tokens_dist is None
        assert loaded.answer_tokens_dist is not None
        assert loaded.tool_call_probability == 0.05
        assert loaded.tool_type_distribution == {"web_search": 0.6, "calculator": 0.4}
        assert loaded.domain_mix == {"general": 0.7, "technical": 0.3}
        assert loaded.max_context_observed == 32768

    def test_all_none_optional_fields(self, tmp_path):
        profile = WorkloadProfile(
            workload_type="reasoning",
            source_dataset="openthoughts",
            n_samples=500,
        )

        path = tmp_path / "minimal.json"
        profile.save(path)

        loaded = WorkloadProfile.load(path)
        assert loaded.workload_type == "reasoning"
        assert loaded.source_dataset == "openthoughts"
        assert loaded.n_samples == 500
        assert loaded.turns_or_steps_dist is None
        assert loaded.input_tokens_dist is None
        assert loaded.thinking_tokens_dist is None
        assert loaded.answer_tokens_dist is None
        assert loaded.input_tokens_by_position == {}
        assert loaded.output_tokens_by_position == {}
        assert loaded.tool_call_probability == 0.0
        assert loaded.tool_call_tokens_dist is None
        assert loaded.tool_type_distribution == {}
        assert loaded.inter_turn_seconds_dist is None
        assert loaded.context_growth_rate_dist is None
        assert loaded.domain_mix == {}

    def test_position_conditioned_distributions(self, tmp_path):
        profile = WorkloadProfile(
            workload_type="agentic",
            source_dataset="agent-data-collection",
            n_samples=200,
            input_tokens_by_position={
                0: _make_fd(mean=800.0),
                1: _make_fd(mean=1200.0),
                2: _make_fd(mean=1500.0),
            },
            output_tokens_by_position={
                0: _make_fd(mean=300.0),
                1: _make_fd(mean=400.0),
            },
        )

        path = tmp_path / "pos.json"
        profile.save(path)

        loaded = WorkloadProfile.load(path)
        assert len(loaded.input_tokens_by_position) == 3
        assert 0 in loaded.input_tokens_by_position
        assert loaded.input_tokens_by_position[0].mean == 800.0
        assert loaded.input_tokens_by_position[2].mean == 1500.0
        assert len(loaded.output_tokens_by_position) == 2
        assert loaded.output_tokens_by_position[1].mean == 400.0

    def test_domain_mix_serialization(self, tmp_path):
        profile = WorkloadProfile(
            workload_type="rag",
            source_dataset="hotpotqa",
            n_samples=300,
            domain_mix={
                "science": 0.3,
                "history": 0.25,
                "technology": 0.25,
                "other": 0.2,
            },
        )

        path = tmp_path / "domain.json"
        profile.save(path)

        loaded = WorkloadProfile.load(path)
        assert loaded.domain_mix == {
            "science": 0.3,
            "history": 0.25,
            "technology": 0.25,
            "other": 0.2,
        }

    def test_saved_file_is_valid_json(self, tmp_path):
        profile = WorkloadProfile(
            workload_type="coding",
            source_dataset="swebench",
            n_samples=100,
            turns_or_steps_dist=_make_fd(),
        )
        path = tmp_path / "check.json"
        profile.save(path)

        with open(path) as f:
            data = json.load(f)
        assert data["workload_type"] == "coding"
        assert isinstance(data["turns_or_steps_dist"], dict)
        assert data["turns_or_steps_dist"]["dist_name"] == "lognormal"
