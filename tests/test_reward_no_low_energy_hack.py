from kite.rewards.ipw_reward import IPWRewardConfig, compute_ipw_reward


def test_incorrect_kernel_cannot_win_with_low_energy() -> None:
    cfg = IPWRewardConfig(
        alpha_speedup=1.0,
        beta_joules=1.0,
        gamma_latency=0.5,
        compile_fail_reward=-1.0,
        incorrect_reward=-0.5,
    )

    bad = compute_ipw_reward(
        compile_ok=True,
        correct=False,
        speedup=100.0,
        joules=1e-6,
        p95_latency_s=0.001,
        sla_latency_s=1.0,
        config=cfg,
    )
    assert bad.total == -0.5

