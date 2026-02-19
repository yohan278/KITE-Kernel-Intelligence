from pathlib import Path

from kite.cli import main


def test_cli_smoke_pipeline(tmp_path: Path) -> None:
    kb_root = tmp_path / "external" / "KernelBench"
    kb_root.mkdir(parents=True, exist_ok=True)

    processed = tmp_path / "data" / "kernelbench" / "processed"
    assert main(["--seed", "123", "data", "build", "--kernelbench-root", str(kb_root), "--output", str(processed)]) == 0

    sft_out = tmp_path / "checkpoints" / "sft"
    assert main(["train", "sft", "--kernelbench-root", str(kb_root), "--output", str(sft_out)]) == 0
    assert (sft_out / "checkpoint.json").exists()

    kernel_out = tmp_path / "checkpoints" / "kernel_grpo"
    assert main(["train", "kernel-grpo", "--kernelbench-root", str(kb_root), "--output", str(kernel_out), "--epochs", "1"]) == 0
    assert (kernel_out / "checkpoint.json").exists()

    runtime_out = tmp_path / "checkpoints" / "runtime_ppo"
    assert main(["train", "runtime-ppo", "--output", str(runtime_out), "--episodes", "2", "--horizon", "2"]) == 0
    assert (runtime_out / "checkpoint.json").exists()

    hrl_out = tmp_path / "checkpoints" / "hrl"
    assert main(["train", "hrl", "--kernelbench-root", str(kb_root), "--output", str(hrl_out), "--rounds", "1"]) == 0
    assert (hrl_out / "checkpoint.json").exists()

    eval_out = tmp_path / "outputs" / "eval"
    assert main(["eval", "suite", "--output", str(eval_out)]) == 0
    assert (eval_out / "suite_results.json").exists()
    assert (eval_out / "report.md").exists()
