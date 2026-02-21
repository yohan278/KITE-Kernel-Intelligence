from kite.agents.llm_agent import LLMKernelAgent
from kite.policies.qwen_policy import QwenPolicy, QwenPolicyConfig
from kite.types import KernelTask


def test_multiturn_pass_at_k() -> None:
    policy = QwenPolicy(QwenPolicyConfig(generation_mode="stub"))
    agent = LLMKernelAgent(policy)
    task = KernelTask(
        task_id="L1_1",
        level=1,
        prompt="Optimize this kernel",
        reference_kernel="def kernel(x): return x",
    )

    def evaluate_fn(code: str) -> dict:
        ok = "return" in code
        return {
            "compile_ok": ok,
            "correct": ok,
            "reward": 1.0 if ok else -1.0,
            "runtime_ms": 10.0,
            "joules": 2.0,
        }

    result = agent.optimize_task(task, evaluate_fn=evaluate_fn, max_turns=3)
    assert result.pass_at_k
    assert result.turns_to_success > 0
    assert len(result.steps) >= 1

