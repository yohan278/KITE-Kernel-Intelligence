"""KITE command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.eval.ablations import run_ablations
from kite.eval.benchmark_runner import BenchmarkRunner
from kite.eval.reports import save_suite_artifacts
from kite.policies.qwen_policy import QwenPolicy
from kite.trainers.grpo_kernel_trainer import GRPOKernelConfig, GRPOKernelTrainer
from kite.trainers.hrl_trainer import HRLTrainer
from kite.trainers.ppo_runtime_trainer import PPORuntimeConfig, PPORuntimeTrainer
from kite.trainers.sft_trainer import SFTConfig, SFTTrainer
from kite.utils.logging import configure_logging, get_logger
from kite.utils.seeds import set_seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kite", description="KITE research CLI")
    parser.add_argument("--seed", type=int, default=42)

    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Data operations")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    data_build = data_sub.add_parser("build", help="Build KernelBench dataset splits")
    data_build.add_argument("--kernelbench-root", type=Path, default=Path("external/KernelBench"))
    data_build.add_argument("--output", type=Path, default=Path("data/kernelbench/processed"))

    train = sub.add_parser("train", help="Training operations")
    train_sub = train.add_subparsers(dest="train_cmd", required=True)

    sft = train_sub.add_parser("sft", help="Run SFT stage")
    sft.add_argument("--kernelbench-root", type=Path, default=Path("external/KernelBench"))
    sft.add_argument("--output", type=Path, default=Path("checkpoints/sft"))

    kernel = train_sub.add_parser("kernel-grpo", help="Run kernel GRPO stage")
    kernel.add_argument("--kernelbench-root", type=Path, default=Path("external/KernelBench"))
    kernel.add_argument("--output", type=Path, default=Path("checkpoints/kernel_grpo"))
    kernel.add_argument("--epochs", type=int, default=3)
    kernel.add_argument("--energy-aware", action="store_true")

    runtime = train_sub.add_parser("runtime-ppo", help="Run runtime PPO stage")
    runtime.add_argument("--output", type=Path, default=Path("checkpoints/runtime_ppo"))
    runtime.add_argument("--episodes", type=int, default=20)
    runtime.add_argument("--horizon", type=int, default=10)

    hrl = train_sub.add_parser("hrl", help="Run hierarchical alternating training")
    hrl.add_argument("--kernelbench-root", type=Path, default=Path("external/KernelBench"))
    hrl.add_argument("--output", type=Path, default=Path("checkpoints/hrl"))
    hrl.add_argument("--rounds", type=int, default=2)

    ev = sub.add_parser("eval", help="Evaluation operations")
    ev_sub = ev.add_subparsers(dest="eval_cmd", required=True)
    suite = ev_sub.add_parser("suite", help="Run benchmark suite and reports")
    suite.add_argument("--output", type=Path, default=Path("outputs/eval"))

    return parser


def _cmd_data_build(args: argparse.Namespace) -> int:
    logger = get_logger("kite.data")
    adapter = KernelBenchAdapter(args.kernelbench_root)
    paths = adapter.build_splits(args.output)
    logger.info("Built dataset splits at %s", args.output)
    for split, path in paths.items():
        logger.info("%s: %s", split, path)
    return 0


def _cmd_train_sft(args: argparse.Namespace) -> int:
    adapter = KernelBenchAdapter(args.kernelbench_root)
    trainer = SFTTrainer(adapter=adapter, policy=QwenPolicy(), config=SFTConfig(output_dir=args.output))
    summary = trainer.run()
    get_logger("kite.train").info("SFT complete: %s", summary)
    return 0


def _cmd_train_kernel_grpo(args: argparse.Namespace) -> int:
    adapter = KernelBenchAdapter(args.kernelbench_root)
    trainer = GRPOKernelTrainer(
        adapter=adapter,
        policy=QwenPolicy(),
        config=GRPOKernelConfig(
            output_dir=args.output,
            epochs=args.epochs,
            energy_aware=bool(args.energy_aware),
        ),
    )
    summary = trainer.run()
    get_logger("kite.train").info("Kernel GRPO complete: %s", summary)
    return 0


def _cmd_train_runtime_ppo(args: argparse.Namespace) -> int:
    trainer = PPORuntimeTrainer(
        config=PPORuntimeConfig(
            output_dir=args.output,
            episodes=args.episodes,
            horizon=args.horizon,
        )
    )
    summary = trainer.run()
    get_logger("kite.train").info("Runtime PPO complete: %s", summary)
    return 0


def _cmd_train_hrl(args: argparse.Namespace) -> int:
    trainer = HRLTrainer(kernelbench_root=args.kernelbench_root)
    trainer.config.output_dir = args.output
    trainer.config.alternating_rounds = args.rounds
    summary = trainer.run()
    get_logger("kite.train").info("HRL complete: %s", summary)
    return 0


def _cmd_eval_suite(args: argparse.Namespace) -> int:
    runner = BenchmarkRunner(output_dir=args.output)
    suite = runner.run()
    ablations = run_ablations(output_dir=args.output)
    artifacts = save_suite_artifacts(args.output, suite)

    get_logger("kite.eval").info("Suite complete (%d experiments)", suite.get("num_experiments", 0))
    get_logger("kite.eval").info("Ablations complete (%d rows)", len(ablations.get("ablations", [])))
    get_logger("kite.eval").info("Artifacts: %s", artifacts)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)

    if args.command == "data" and args.data_cmd == "build":
        return _cmd_data_build(args)
    if args.command == "train" and args.train_cmd == "sft":
        return _cmd_train_sft(args)
    if args.command == "train" and args.train_cmd == "kernel-grpo":
        return _cmd_train_kernel_grpo(args)
    if args.command == "train" and args.train_cmd == "runtime-ppo":
        return _cmd_train_runtime_ppo(args)
    if args.command == "train" and args.train_cmd == "hrl":
        return _cmd_train_hrl(args)
    if args.command == "eval" and args.eval_cmd == "suite":
        return _cmd_eval_suite(args)

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
