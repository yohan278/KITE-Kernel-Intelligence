from kite.rewards.grpo_reward import GRPOMultiMetricRewardConfig, compute_grpo_multi_metric_reward


def test_grpo_reward_penalizes_power_and_energy() -> None:
    cfg = GRPOMultiMetricRewardConfig(
        alpha_speedup=1.0,
        beta_joules=0.5,
        gamma_latency=0.25,
        delta_avg_power=0.01,
        eta_runtime=0.1,
        correctness_bonus=0.0,
    )
    reward = compute_grpo_multi_metric_reward(
        compile_ok=True,
        correct=True,
        speedup=2.0,
        runtime_ms=10.0,
        joules=0.25,
        avg_power_w=150.0,
        p95_latency_s=0.02,
        compile_log=None,
        correctness_log=None,
        config=cfg,
    )
    assert reward.total != 0.0
    assert reward.performance > 0.0
    assert reward.energy < 0.0


def test_grpo_reward_oom_is_harder_failure() -> None:
    cfg = GRPOMultiMetricRewardConfig(
        compile_fail_reward=-1.0,
        incorrect_reward=-0.5,
        oom_penalty=0.75,
    )

    oom = compute_grpo_multi_metric_reward(
        compile_ok=False,
        correct=False,
        speedup=None,
        runtime_ms=None,
        joules=None,
        avg_power_w=None,
        p95_latency_s=None,
        compile_log="CUDA out of memory",
        correctness_log=None,
        config=cfg,
    )
    plain = compute_grpo_multi_metric_reward(
        compile_ok=False,
        correct=False,
        speedup=None,
        runtime_ms=None,
        joules=None,
        avg_power_w=None,
        p95_latency_s=None,
        compile_log="syntax error",
        correctness_log=None,
        config=cfg,
    )
    assert oom.total < plain.total

