"""Tests for confidence-aware PPI SLA checker."""
from __future__ import annotations

import pytest

ppi_py = pytest.importorskip("ppi_py")

from inference_search.ppi_sla_checker import check_with_confidence
from inference_search.types import SLAConstraint
from inference_simulator.metrics.ppi_validation import RectifiedSimulationMetrics


class TestConservativeMode:
    """Tests for conservative mode (uses CI bounds)."""

    def test_passes_when_ci_upper_within_threshold(self):
        """Config passes when CI upper bound is within max-constraint threshold."""
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.4,
            ttft_p90_ci=(0.35, 0.45),
        )
        constraints = [SLAConstraint("ttft_p90", 0.5, "max")]
        passed, violations = check_with_confidence(metrics, constraints, "conservative")
        assert passed
        assert violations == []

    def test_fails_when_ci_upper_exceeds_threshold(self):
        """Config fails when CI upper bound exceeds max-constraint threshold."""
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.48,
            ttft_p90_ci=(0.42, 0.55),
        )
        constraints = [SLAConstraint("ttft_p90", 0.5, "max")]
        passed, violations = check_with_confidence(metrics, constraints, "conservative")
        assert not passed
        assert len(violations) == 1

    def test_barely_passes_point_estimate_but_fails_conservative(self):
        """Point estimate passes, but CI upper bound fails."""
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.49,
            ttft_p90_ci=(0.44, 0.54),
        )
        constraints = [SLAConstraint("ttft_p90", 0.5, "max")]

        # Conservative should fail (CI upper 0.54 > 0.5)
        passed_con, _ = check_with_confidence(metrics, constraints, "conservative")
        assert not passed_con

        # Optimistic should pass (point estimate 0.49 < 0.5)
        passed_opt, _ = check_with_confidence(metrics, constraints, "optimistic")
        assert passed_opt

    def test_min_constraint_uses_ci_lower(self):
        """For min-constraints, conservative mode uses CI lower bound."""
        metrics = RectifiedSimulationMetrics(
            ttft_p50=100.0,
            ttft_p50_ci=(80.0, 120.0),
        )
        # min constraint: throughput >= 90
        # Using ttft_p50 as a stand-in since we need an attribute that exists
        constraints = [SLAConstraint("ttft_p50", 90.0, "min")]

        # Conservative: CI lower = 80 < 90 => fails
        passed, violations = check_with_confidence(metrics, constraints, "conservative")
        assert not passed

    def test_no_ci_falls_back_to_point_estimate(self):
        """Without CI, conservative mode uses point estimate."""
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.45,
            ttft_p90_ci=None,
        )
        constraints = [SLAConstraint("ttft_p90", 0.5, "max")]
        passed, _ = check_with_confidence(metrics, constraints, "conservative")
        assert passed


class TestOptimisticMode:
    """Tests for optimistic mode (uses point estimates)."""

    def test_uses_point_estimate(self):
        """Optimistic mode uses the rectified point estimate."""
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.45,
            ttft_p90_ci=(0.40, 0.55),
        )
        constraints = [SLAConstraint("ttft_p90", 0.5, "max")]
        passed, _ = check_with_confidence(metrics, constraints, "optimistic")
        assert passed

    def test_still_fails_when_point_estimate_violates(self):
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.55,
            ttft_p90_ci=(0.50, 0.60),
        )
        constraints = [SLAConstraint("ttft_p90", 0.5, "max")]
        passed, violations = check_with_confidence(metrics, constraints, "optimistic")
        assert not passed


class TestEdgeCases:
    def test_invalid_mode_raises(self):
        metrics = RectifiedSimulationMetrics()
        with pytest.raises(ValueError, match="confidence_mode"):
            check_with_confidence(metrics, [], "invalid")

    def test_missing_metric_reported(self):
        metrics = RectifiedSimulationMetrics()
        constraints = [SLAConstraint("nonexistent_metric", 1.0, "max")]
        passed, violations = check_with_confidence(metrics, constraints, "conservative")
        assert not passed
        assert "not found" in violations[0]

    def test_empty_constraints_pass(self):
        metrics = RectifiedSimulationMetrics()
        passed, violations = check_with_confidence(metrics, [], "conservative")
        assert passed
        assert violations == []

    def test_multiple_constraints(self):
        metrics = RectifiedSimulationMetrics(
            ttft_p90=0.4,
            ttft_p90_ci=(0.35, 0.45),
            e2e_p95=1.0,
            e2e_p95_ci=(0.8, 1.2),
        )
        constraints = [
            SLAConstraint("ttft_p90", 0.5, "max"),  # passes
            SLAConstraint("e2e_p95", 0.9, "max"),  # fails (CI upper 1.2 > 0.9)
        ]
        passed, violations = check_with_confidence(metrics, constraints, "conservative")
        assert not passed
        assert len(violations) == 1
        assert "e2e_p95" in violations[0]
