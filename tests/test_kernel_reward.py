from kite.rewards.kernel_reward import KernelRewardConfig, compute_kernel_reward
from kite.types import KernelCandidate


def test_kernel_reward_correct_fast_candidate() -> None:
    candidate = KernelCandidate(
        task_id="t1",
        code="def kernel(): return 1",
        compile_ok=True,
        correct=True,
        runtime_ms=100.0,
        speedup=2.0,
    )
    reward = compute_kernel_reward(candidate, timeout_ms=500.0, config=KernelRewardConfig())
    assert reward.correctness == 1.0
    assert reward.performance == 2.0
    assert abs(reward.total - 1.3) < 1e-9


def test_kernel_reward_penalizes_compile_and_timeout() -> None:
    candidate = KernelCandidate(
        task_id="t2",
        code="TODO",
        compile_ok=False,
        correct=False,
        runtime_ms=900.0,
        speedup=0.0,
    )
    reward = compute_kernel_reward(candidate, timeout_ms=500.0, config=KernelRewardConfig())
    assert reward.total < -1.0
