from kite.telemetry.energy_capture import EnergyCapture
from kite.telemetry.phase_attribution import attribute_prefill_decode


def test_phase_attribution_creates_prefill_decode_segments() -> None:
    capture = EnergyCapture()
    trace = capture.synthetic_trace(steps=50)
    out = attribute_prefill_decode(trace, ttft_s=0.5)
    assert len(out.phase_segments) == 2
    assert out.phase_segments[0].name == "prefill"
    assert out.phase_segments[1].name == "decode"
    assert (out.phase_segments[0].energy_j or 0.0) >= 0.0
    assert (out.phase_segments[1].energy_j or 0.0) >= 0.0
