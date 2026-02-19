from kite.adapters.ipw_adapter import IPWAdapter


def test_ipw_adapter_summary_from_trace_payload() -> None:
    adapter = IPWAdapter()
    trace = adapter.parse_trace(
        {
            "timestamps": [0.0, 1.0],
            "power_w": [200.0, 220.0],
            "energy_j": [0.0, 210.0],
            "phase_segments": [
                {"name": "prefill", "start_s": 0.0, "end_s": 0.3, "energy_j": 60.0},
                {"name": "decode", "start_s": 0.3, "end_s": 1.0, "energy_j": 150.0},
            ],
        }
    )
    summary = adapter.summarize(trace, input_tokens=300, output_tokens=100)
    assert summary.total_energy_j == 210.0
    assert summary.energy_per_output_token_j == 2.1
    assert summary.prefill_energy_per_input_token_j == 0.2
    assert summary.decode_energy_per_output_token_j == 1.5
