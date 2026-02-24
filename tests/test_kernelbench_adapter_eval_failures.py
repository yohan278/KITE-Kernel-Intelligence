import sys
import types
from pathlib import Path

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.types import KernelTask


def _install_fake_kernelbench(monkeypatch, raising_fn):
    kb_pkg = types.ModuleType("kernelbench")
    kb_eval = types.ModuleType("kernelbench.eval")
    setattr(kb_eval, "eval_kernel_against_ref", raising_fn)
    setattr(kb_eval, "get_torch_dtype_from_string", lambda _: "fp32")
    monkeypatch.setitem(sys.modules, "kernelbench", kb_pkg)
    monkeypatch.setitem(sys.modules, "kernelbench.eval", kb_eval)


def _install_fake_torch(monkeypatch):
    cuda_ns = types.SimpleNamespace(is_available=lambda: True)
    fake_torch = types.SimpleNamespace(cuda=cuda_ns, device=lambda name: name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_kernelbench_oom_returns_hard_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    adapter = KernelBenchAdapter(root, enable_kernelbench_eval=True)
    monkeypatch.setattr(adapter, "_ensure_kernelbench_on_path", lambda: None)

    def _raise_oom(**kwargs):  # noqa: ANN003
        raise RuntimeError("CUDA out of memory")

    _install_fake_torch(monkeypatch)
    _install_fake_kernelbench(monkeypatch, _raise_oom)

    task = KernelTask(
        task_id="L1_1",
        level=1,
        prompt="p",
        reference_kernel="rk",
        metadata={"ref_arch_src": "import torch\nclass Model: pass"},
    )
    cand = adapter.evaluate_candidate(task, "class ModelNew:\n    pass\n")

    assert cand.compile_ok is False
    assert cand.correct is False
    assert "out of memory" in (cand.compile_log or "").lower()
    assert cand.logs.get("kernelbench_eval") is True
    assert cand.logs.get("proxy_eval") is None
    assert bool(cand.logs.get("metadata", {}).get("oom")) is True

