"""Tests for inference_search.sla_checker."""

from __future__ import annotations

from inference_search.sla_checker import check
from inference_search.types import SLAConstraint


class TestSLAChecker:
    def test_all_pass(self) -> None:
        metrics = {"ttft_s": 0.5, "throughput_tps": 100.0, "avg_power_w": 300.0}
        constraints = [
            SLAConstraint("ttft_s", 1.0, "max"),
            SLAConstraint("throughput_tps", 50.0, "min"),
            SLAConstraint("avg_power_w", 400.0, "max"),
        ]
        passed, violations = check(metrics, constraints)
        assert passed is True
        assert violations == []

    def test_max_violation(self) -> None:
        metrics = {"ttft_s": 2.0}
        constraints = [SLAConstraint("ttft_s", 1.0, "max")]
        passed, violations = check(metrics, constraints)
        assert passed is False
        assert len(violations) == 1
        assert "exceeds max" in violations[0]

    def test_min_violation(self) -> None:
        metrics = {"throughput_tps": 5.0}
        constraints = [SLAConstraint("throughput_tps", 10.0, "min")]
        passed, violations = check(metrics, constraints)
        assert passed is False
        assert len(violations) == 1
        assert "below min" in violations[0]

    def test_exact_threshold_passes(self) -> None:
        metrics = {"ttft_s": 1.0}
        constraints = [SLAConstraint("ttft_s", 1.0, "max")]
        passed, _ = check(metrics, constraints)
        assert passed is True

    def test_exact_threshold_min_passes(self) -> None:
        metrics = {"throughput_tps": 10.0}
        constraints = [SLAConstraint("throughput_tps", 10.0, "min")]
        passed, _ = check(metrics, constraints)
        assert passed is True

    def test_missing_metric(self) -> None:
        metrics = {"ttft_s": 0.5}
        constraints = [SLAConstraint("nonexistent", 1.0, "max")]
        passed, violations = check(metrics, constraints)
        assert passed is False
        assert "not found" in violations[0]

    def test_empty_constraints(self) -> None:
        metrics = {"ttft_s": 999.0}
        passed, violations = check(metrics, [])
        assert passed is True
        assert violations == []

    def test_multiple_violations(self) -> None:
        metrics = {"ttft_s": 2.0, "throughput_tps": 5.0}
        constraints = [
            SLAConstraint("ttft_s", 1.0, "max"),
            SLAConstraint("throughput_tps", 10.0, "min"),
        ]
        passed, violations = check(metrics, constraints)
        assert passed is False
        assert len(violations) == 2
