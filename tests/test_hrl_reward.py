from kite.rewards.hrl_reward import HRLRewardConfig, compute_hrl_reward


def test_hrl_reward_penalizes_latency_violations() -> None:
    cfg = HRLRewardConfig()
    good = compute_hrl_reward(
        throughput_tps=100.0,
        apj=0.04,
        apw=0.003,
        ttft_p95=1.0,
        e2e_p95=5.0,
        ttft_sla=2.0,
        e2e_sla=30.0,
        stability_score=0.9,
        config=cfg,
    )
    bad = compute_hrl_reward(
        throughput_tps=100.0,
        apj=0.04,
        apw=0.003,
        ttft_p95=4.0,
        e2e_p95=40.0,
        ttft_sla=2.0,
        e2e_sla=30.0,
        stability_score=0.9,
        config=cfg,
    )
    assert good.total > bad.total
