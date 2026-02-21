from kite.measurement.protocol import MeasurementConfig, MeasurementProtocol


def test_measurement_protocol_returns_stats() -> None:
    counter = {"x": 0}

    def workload() -> None:
        counter["x"] += 1
        _ = sum(i * i for i in range(200))

    protocol = MeasurementProtocol(
        MeasurementConfig(
            warmup_iters=1,
            measure_iters=2,
            repeats=3,
            sampling_interval_ms=5.0,
        )
    )
    result = protocol.measure(workload)
    assert result.repeats == 3
    assert result.runtime_ms_mean >= 0.0
    assert result.energy_j_mean >= 0.0
    assert len(result.runs) == 3
    assert counter["x"] >= 7  # warmup + measured calls

