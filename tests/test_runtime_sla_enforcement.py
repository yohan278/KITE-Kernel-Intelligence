from kite.rewards.hrl_reward import HRLRewardConfig, compute_hrl_reward


def test_runtime_reward_sla_penalty_under_bursty_load() -> None:
    cfg = HRLRewardConfig(latency_violation_weight=0.5)
    compliant = compute_hrl_reward(
        throughput_tps=120.0,
        apj=0.08,
        apw=0.004,
        ttft_p95=1.8,
        e2e_p95=20.0,
        ttft_sla=2.0,
        e2e_sla=30.0,
        stability_score=0.9,
        config=cfg,
    )
    violating = compute_hrl_reward(
        throughput_tps=120.0,
        apj=0.08,
        apw=0.004,
        ttft_p95=4.2,
        e2e_p95=55.0,
        ttft_sla=2.0,
        e2e_sla=30.0,
        stability_score=0.9,
        config=cfg,
    )
    assert compliant.total > violating.total

