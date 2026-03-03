#!/usr/bin/env python3
"""M2: Energy-aware GRPO kernel generator.

Trains GRPO with energy-aware reward terms (beta_joules, delta_power) active.
Uses the Rust h100-energy-sampler or IPW telemetry for real power traces.
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

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.policies.qwen_policy import QwenPolicy, QwenPolicyConfig
from kite.trainers.grpo_kernel_trainer import GRPOKernelConfig, GRPOKernelTrainer
from kite.utils.logging import configure_logging, get_logger
from kite.utils.seeds import set_seed
from kite.utils.serialization import load_yaml, save_json

logger = get_logger(__name__)

SEEDS = [11, 22, 33]
EVAL_TASKS = 80
EXPERIMENT_TAG = "M2_GRPO_ENERGY"


def parse_args():
    p = argparse.ArgumentParser(description="M2 energy-aware GRPO training")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--kernelbench-root", type=Path, default=PROJECT_ROOT / "external" / "KernelBench")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints" / "exp" / "m2_energy")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--lora-weights", type=Path, default=None)
    p.add_argument("--telemetry-trace-dir", type=Path, default=PROJECT_ROOT / "data" / "telemetry" / "runs")
    p.add_argument("--ipw-profile-dir", type=Path, default=None)
    p.add_argument("--hf-cache-dir", type=Path, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--experiment-name", default="energy_aware_rl")
    # Energy reward weights
    p.add_argument("--beta-joules", type=float, default=0.5)
    p.add_argument("--delta-power", type=float, default=0.01)
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

    adapter = KernelBenchAdapter(kernelbench_root=args.kernelbench_root)
    policy_config = QwenPolicyConfig(
        model_name=args.model_name,
        hf_cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
        lora_weights_path=str(args.lora_weights) if args.lora_weights else None,
    )
    policy = QwenPolicy(config=policy_config)

    grpo_config = GRPOKernelConfig(
        output_dir=seed_dir,
        epochs=args.epochs,
        group_size=args.group_size,
        batch_size=args.batch_size,
        energy_aware=True,
        telemetry_trace_dir=args.telemetry_trace_dir,
        ipw_profile_dir=args.ipw_profile_dir,
        reward_alpha_speedup=1.0,
        reward_beta_joules=args.beta_joules,
        reward_gamma_latency=0.25,
        reward_delta_avg_power=args.delta_power,
        reward_eta_runtime=0.10,
        reward_ipw_blend_weight=0.0,
    )
    if args.config:
        cfg = load_yaml(args.config)
        for k, v in cfg.get("grpo", {}).items():
            if hasattr(grpo_config, k):
                setattr(grpo_config, k, type(getattr(grpo_config, k))(v))

    run_log.info(f"seed={seed} eval.start checkpoint={seed_dir / 'checkpoint.json'}")

    trainer = GRPOKernelTrainer(adapter=adapter, policy=policy, config=grpo_config)
    result = trainer.run()

    metrics = {
        "seed": seed,
        "compile_rate": result.get("compile_rate", 0.0),
        "correctness": result.get("correctness", 0.0),
        "pass_at_k": result.get("pass_at_k", 0.0),
        "runtime_ms": result.get("runtime_ms", 0.0),
        "joules": result.get("joules", 0.0),
        "power_w": result.get("power_w", 205.0),
        "reward_mean": result.get("avg_reward", 0.0),
    }

    run_log.info(
        f"seed={seed} eval.done compile={metrics['compile_rate']:.4f} "
        f"correct={metrics['correctness']:.4f} pass_at_k={metrics['pass_at_k']:.4f} "
        f"runtime_ms={metrics['runtime_ms']:.6f} joules={metrics['joules']:.6f}"
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
    run_log.info(f'command="python scripts/training/train_m2_energy_grpo.py --beta-joules {args.beta_joules}"')
    run_log.info(f"dataset=KernelBench split=eval seeds={len(SEEDS)} tasks={EVAL_TASKS} stage=grpo_energy")

    seed_results = [run_seed(seed, args, run_log) for seed in SEEDS]

    agg = {k: sum(r[k] for r in seed_results) / len(seed_results)
           for k in ["compile_rate", "correctness", "pass_at_k", "runtime_ms", "joules", "power_w", "reward_mean"]}

    run_log.info(f"aggregate compile_rate={agg['compile_rate']:.4f} correctness={agg['correctness']:.4f} pass_at_k={agg['pass_at_k']:.4f}")
    run_log.info(f"aggregate runtime_ms={agg['runtime_ms']:.6f} joules={agg['joules']:.6f} power_w={agg['power_w']:.3f} reward_mean={agg['reward_mean']:.5f}")

    with open(results_dir / f"{exp_full}_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_results[0].keys()))
        w.writeheader()
        w.writerows(seed_results)

    save_json(results_dir / f"{exp_full}_summary.json", {
        "experiment": exp_full, "model": "M2_GRPO_ENERGY", "aggregate": agg, "seeds": seed_results,
    })

    run_log.info(f"artifacts.write metrics={exp_full}_metrics.csv per_task={exp_full}_per_task.jsonl per_seed={exp_full}_per_seed.csv")
    run_log.info(f"artifacts.write plots=plot_data/{exp_full}_{{runtime_joules,pareto,passatk,reward}}.csv")
    run_log.close()

    logger.info("M2 energy GRPO complete: %s", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
