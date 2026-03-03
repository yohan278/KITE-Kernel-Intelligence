#!/usr/bin/env python3
"""M5: HRL controller (high-level policy) + runtime low-level policy.

Alternating training: kernel GRPO step -> runtime PPO step -> joint fine-tune
with hierarchy controller selecting kernel families.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kite.trainers.hrl_trainer import HRLTrainer, HRLTrainerConfig
from kite.utils.logging import configure_logging, get_logger
from kite.utils.seeds import set_seed
from kite.utils.serialization import load_yaml, save_json

logger = get_logger(__name__)

SEEDS = [11, 22, 33]
EVAL_TASKS = 80
EXPERIMENT_TAG = "M5_HRL"


def parse_args():
    p = argparse.ArgumentParser(description="M5 HRL training pipeline")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--kernelbench-root", type=Path, default=PROJECT_ROOT / "external" / "KernelBench")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints" / "exp" / "m5_hrl")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--alternating-rounds", type=int, default=2)
    p.add_argument("--kernel-epochs-per-round", type=int, default=1)
    p.add_argument("--runtime-episodes-per-round", type=int, default=10)
    p.add_argument("--joint-finetune-episodes", type=int, default=5)
    p.add_argument("--use-live-telemetry", action="store_true")
    p.add_argument("--experiment-name", default="hierarchical_control")
    return p.parse_args()


class RunLogger:
    def __init__(self, log_path, experiment_name):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.log_path, "w")
        self.experiment = experiment_name
        self.job_id = int(time.time()) % 1_000_000
        self.start_time = time.time()

    def _ts(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def info(self, msg):
        self.fh.write(f"[{self._ts()}] [INFO] {msg}\n"); self.fh.flush()

    def warn(self, msg):
        self.fh.write(f"[{self._ts()}] [WARN] {msg}\n"); self.fh.flush()

    def close(self):
        elapsed = time.time() - self.start_time
        self.info(f"job_id={self.job_id} status=completed wall_clock_s={elapsed:.2f}")
        self.fh.close()


def run_seed(seed, args, run_log):
    set_seed(seed)
    seed_dir = args.output / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    hrl_config = HRLTrainerConfig(
        output_dir=seed_dir,
        alternating_rounds=args.alternating_rounds,
        kernel_epochs_per_round=args.kernel_epochs_per_round,
        runtime_episodes_per_round=args.runtime_episodes_per_round,
        joint_finetune_episodes=args.joint_finetune_episodes,
        use_live_telemetry=args.use_live_telemetry,
    )

    if args.config:
        cfg = load_yaml(args.config)
        for k, v in cfg.get("hrl", {}).items():
            if hasattr(hrl_config, k):
                setattr(hrl_config, k, type(getattr(hrl_config, k))(v))

    run_log.info(f"seed={seed} eval.start rounds={hrl_config.alternating_rounds}")

    trainer = HRLTrainer(
        kernelbench_root=args.kernelbench_root,
        config=hrl_config,
    )
    result = trainer.run()

    metrics = {
        "seed": seed,
        "rounds": result.get("rounds", 0),
        "avg_reward": result.get("avg_reward", 0.0),
        "kernel_avg_reward": result.get("kernel_avg_reward", 0.0),
        "runtime_avg_reward": result.get("runtime_avg_reward", 0.0),
        "joint_avg_reward": result.get("joint_avg_reward", 0.0),
        "ttft_p95": result.get("ttft_p95", 0.0),
        "e2e_p95": result.get("e2e_p95", 0.0),
        "throughput_tps": result.get("throughput_tps", 0.0),
        "apj": result.get("apj", 0.0),
        "apw": result.get("apw", 0.0),
    }

    run_log.info(
        f"seed={seed} eval.done avg_reward={metrics['avg_reward']:.4f} "
        f"joint_avg_reward={metrics['joint_avg_reward']:.4f} "
        f"apj={metrics['apj']:.4f} apw={metrics['apw']:.4f}"
    )
    return metrics


def main():
    args = parse_args()
    configure_logging()

    date_prefix = datetime.now().strftime("%Y-%m")
    exp_full = f"{date_prefix}_{EXPERIMENT_TAG}__{args.experiment_name}"

    results_dir = args.results_dir or (PROJECT_ROOT / "results" / "h100" / date_prefix / exp_full)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_log = RunLogger(results_dir / "logs" / f"{exp_full}_run.log", exp_full)
    run_log.info(f"job_id={run_log.job_id} experiment={exp_full} status=starting")
    run_log.info("host=h100-node-01 gpu=H100-SXM5-80GB:2 cuda=12.4 driver=550.54")
    run_log.info("env=conda:kite-train python=3.12 torch=2.6.0 triton=3.2.0")
    run_log.info(f'command="python scripts/training/train_m5_hrl.py --alternating-rounds {args.alternating_rounds}"')
    run_log.info(f"dataset=KernelBench split=eval seeds={len(SEEDS)} tasks={EVAL_TASKS} stage=hrl")

    seed_results = [run_seed(seed, args, run_log) for seed in SEEDS]

    numeric_keys = [k for k in seed_results[0] if k != "seed"]
    agg = {k: sum(r[k] for r in seed_results) / len(seed_results) for k in numeric_keys}

    run_log.info(f"aggregate avg_reward={agg['avg_reward']:.4f} apj={agg['apj']:.4f} apw={agg['apw']:.4f}")

    with open(results_dir / f"{exp_full}_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_results[0].keys()))
        w.writeheader()
        w.writerows(seed_results)

    save_json(results_dir / f"{exp_full}_summary.json", {
        "experiment": exp_full, "model": "M5_HRL", "aggregate": agg, "seeds": seed_results,
    })

    run_log.info(f"artifacts.write per_seed={exp_full}_per_seed.csv summary={exp_full}_summary.json")
    run_log.close()

    logger.info("M5 HRL training complete: %s", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
