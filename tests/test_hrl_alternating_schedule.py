from pathlib import Path

from kite.trainers.hrl_trainer import HRLTrainer, HRLTrainerConfig


def test_hrl_alternating_schedule(monkeypatch, tmp_path: Path) -> None:
    trainer = HRLTrainer(
        kernelbench_root=tmp_path / "KernelBench",
        config=HRLTrainerConfig(
            output_dir=tmp_path / "hrl",
            alternating_rounds=1,
            kernel_epochs_per_round=1,
            runtime_episodes_per_round=1,
            runtime_horizon=2,
            joint_finetune_episodes=1,
        ),
    )

    monkeypatch.setattr(trainer, "_kernel_step", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(trainer, "_runtime_step", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        trainer,
        "_joint_finetune",
        lambda *args, **kwargs: [{"reward": 1.0}, {"reward": 0.5}],
    )

    out = trainer.run()
    assert out["stage"] == "hrl"
    assert out["rounds"] == 1
    assert len(out["summaries"]) == 1
    assert (tmp_path / "hrl" / "checkpoint.json").exists()

