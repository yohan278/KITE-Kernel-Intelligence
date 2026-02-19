from kite.adapters.ipw_adapter import IPWSummary
from kite.rewards.energy_reward import EnergyRewardConfig, compute_energy_aware_reward
from kite.types import KernelCandidate


def test_energy_reward_hard_penalty_for_incorrect_kernel() -> None:
    candidate = KernelCandidate(
        task_id="t",
        code="def kernel(): return None",
        compile_ok=True,
        correct=False,
        runtime_ms=100.0,
        speedup=1.0,
    )
    summary = IPWSummary(
        total_energy_j=10.0,
        avg_power_w=200.0,
        energy_per_output_token_j=0.05,
        prefill_energy_per_input_token_j=0.01,
        decode_energy_per_output_token_j=0.03,
    )
    reward = compute_energy_aware_reward(
        candidate=candidate,
        summary=summary,
        p95_latency_s=0.8,
        sla_latency_s=1.0,
        timeout_ms=500.0,
        config=EnergyRewardConfig(),
    )
    assert reward.total < 0.0
