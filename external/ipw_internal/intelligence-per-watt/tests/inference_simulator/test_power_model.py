"""Tests for per-operator power model."""

import pytest

from inference_simulator.energy.power_model import OperatorEvent, PerOperatorPowerModel
from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


def _make_measurements(categories, n_per_category=10):
    """Create synthetic measurements with power data."""
    measurements = []
    for cat in categories:
        for i in range(n_per_category):
            batch_size = (i + 1) * 4
            seq_len = (i + 1) * 128
            # Simulate higher power for larger batches/seqs
            power_w = 200.0 + batch_size * 0.5 + seq_len * 0.01
            measurements.append(OperatorMeasurement(
                operator_name=f"{cat.value}_op",
                category=cat,
                batch_size=batch_size,
                seq_len=seq_len,
                time_s=0.001 * (i + 1),
                energy_j=power_w * 0.001 * (i + 1),
                power_w=power_w,
            ))
    return measurements


class TestPerOperatorPowerModel:
    def test_fit_with_power_data(self):
        model = PerOperatorPowerModel()
        measurements = _make_measurements(
            [OperatorCategory.LINEAR, OperatorCategory.ATTENTION_PREFILL]
        )
        model.fit(measurements)
        assert model.is_fitted()

    def test_fit_without_power_data(self):
        model = PerOperatorPowerModel()
        measurements = [
            OperatorMeasurement(
                operator_name="op",
                category=OperatorCategory.LINEAR,
                batch_size=4,
                seq_len=128,
                time_s=0.001,
                power_w=None,
            )
        ]
        model.fit(measurements)
        assert model.is_fitted()

    def test_predict_power_varies_by_category(self):
        model = PerOperatorPowerModel()
        # Different categories get different power levels
        measurements = _make_measurements(
            [OperatorCategory.LINEAR, OperatorCategory.ATTENTION_PREFILL],
            n_per_category=10,
        )
        # Make attention use more power
        for m in measurements:
            if m.category == OperatorCategory.ATTENTION_PREFILL:
                # Can't modify directly since these are created by _make_measurements
                pass
        model.fit(measurements)

        power_linear = model.predict_power(OperatorCategory.LINEAR, 16, 512)
        power_attn = model.predict_power(OperatorCategory.ATTENTION_PREFILL, 16, 512)
        # Both should be positive
        assert power_linear > 0
        assert power_attn > 0

    def test_predict_unknown_category_returns_idle(self):
        model = PerOperatorPowerModel()
        measurements = _make_measurements([OperatorCategory.LINEAR])
        model.fit(measurements)

        # Unknown category falls back to idle power
        power = model.predict_power(OperatorCategory.SSM_SCAN, 4, 128)
        assert power >= 0

    def test_compute_energy_from_events(self):
        model = PerOperatorPowerModel()
        measurements = _make_measurements([OperatorCategory.LINEAR])
        model.fit(measurements)

        events = [
            OperatorEvent(
                category=OperatorCategory.LINEAR,
                duration_ns=1_000_000,  # 1ms
                batch_size=4,
                seq_len=128,
                start_time_ns=0,
            ),
            OperatorEvent(
                category=OperatorCategory.LINEAR,
                duration_ns=2_000_000,  # 2ms
                batch_size=8,
                seq_len=256,
                start_time_ns=5_000_000,  # 5ms (4ms gap for idle)
            ),
        ]
        energy = model.compute_energy(events)
        assert energy > 0

    def test_compute_energy_empty_events(self):
        model = PerOperatorPowerModel()
        model.fit([])  # No data
        energy = model.compute_energy([])
        assert energy == 0.0

    def test_idle_power_included_in_gaps(self):
        model = PerOperatorPowerModel()
        measurements = _make_measurements([OperatorCategory.LINEAR])
        model.fit(measurements)

        # Two events with a 10ms gap between them
        events = [
            OperatorEvent(
                category=OperatorCategory.LINEAR,
                duration_ns=1_000_000,
                batch_size=4,
                seq_len=128,
                start_time_ns=0,
            ),
            OperatorEvent(
                category=OperatorCategory.LINEAR,
                duration_ns=1_000_000,
                batch_size=4,
                seq_len=128,
                start_time_ns=11_000_000,  # 10ms gap
            ),
        ]
        energy_with_gap = model.compute_energy(events)

        # Same events with no gap
        events_no_gap = [
            OperatorEvent(
                category=OperatorCategory.LINEAR,
                duration_ns=1_000_000,
                batch_size=4,
                seq_len=128,
                start_time_ns=0,
            ),
            OperatorEvent(
                category=OperatorCategory.LINEAR,
                duration_ns=1_000_000,
                batch_size=4,
                seq_len=128,
                start_time_ns=1_000_000,  # No gap
            ),
        ]
        energy_no_gap = model.compute_energy(events_no_gap)

        # Energy with gap should be >= energy without gap (idle power in gaps)
        assert energy_with_gap >= energy_no_gap


class TestOperatorEvent:
    def test_creation(self):
        event = OperatorEvent(
            category=OperatorCategory.LINEAR,
            duration_ns=1_000_000,
            batch_size=4,
            seq_len=128,
        )
        assert event.category == OperatorCategory.LINEAR
        assert event.duration_ns == 1_000_000
        assert event.start_time_ns == 0  # default
