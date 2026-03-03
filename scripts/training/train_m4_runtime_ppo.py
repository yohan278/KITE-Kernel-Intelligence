#!/usr/bin/env python3
"""M4: Runtime PPO policy.

Trains a PPO-based runtime control policy across three regimes:
  - latency_sensitive: minimize TTFT p95
  - throughput: maximize tokens/s
  - mixed: balanced SLA
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

from kite.envs.runtime_env import RuntimeEnv
from kite.policies.runtime_actor_critic import RuntimeActorCritic, RuntimeActorCriticConfig
from kite.trainers.ppo_runtime_trainer import PPORuntimeConfig, PPORuntimeTrainer
from kite.utils.logging import configure_logging, get_logger
from kite.utils.seeds import set_seed
from kite.utils.serialization import load_yaml, save_json

logger = get_logger(__name__)

SEEDS = [11, 22, 33]
EVAL_TASKS = 80

REGIMES = {
    "latency_sensitive": {"ttft_sla": 1.0, "e2e_sla": 15.0, "episodes": 20, "horizon": 10},
    "throughput":        {"ttft_sla": 5.0, "e2e_sla": 60.0, "episodes": 20, "horizon": 10},
    "mixed":             {"ttft_sla": 2.0, "e2e_sla": 30.0, "episodes": 20, "horizon": 10},
}


def parse_args():
    p = argparse.ArgumentParser(description="M4 Runtime PPO training")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints" / "exp" / "m4_runtime_ppo")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--regimes", nargs="+", default=list(REGIMES.keys()))
    p.add_argument("--use-live-telemetry", action="store_true")
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


def train_regime(regime_name, regime_params, seed, args, run_log):
    set_seed(seed)
    regime_dir = args.output / regime_name / f"seed{seed}"
    regime_dir.mkdir(parents=True, exist_ok=True)

    ppo_config = PPORuntimeConfig(
        output_dir=regime_dir,
        episodes=regime_params["episodes"],
        horizon=regime_params["horizon"],
        ttft_sla=regime_params["ttft_sla"],
        e2e_sla=regime_params["e2e_sla"],
        use_live_telemetry=args.use_live_telemetry,
    )

    if args.config:
        cfg = load_yaml(args.config)
        regime_cfg = cfg.get("regimes", {}).get(regime_name, {})
        for k, v in regime_cfg.items():
            if hasattr(ppo_config, k):
                setattr(ppo_config, k, type(getattr(ppo_config, k))(v))

    env = RuntimeEnv(use_live_telemetry=args.use_live_telemetry)
    actor = RuntimeActorCritic(RuntimeActorCriticConfig())

    run_log.info(f"regime={regime_name} seed={seed} eval.start ttft_sla={ppo_config.ttft_sla} e2e_sla={ppo_config.e2e_sla}")

    trainer = PPORuntimeTrainer(env=env, actor=actor, config=ppo_config)
    result = trainer.run()

    metrics = {
        "regime": regime_name,
        "seed": seed,
        "episodes": result.get("episodes", 0),
        "avg_reward": result.get("avg_reward", 0.0),
        "ttft_p95": result.get("ttft_p95", 0.0),
        "e2e_p95": result.get("e2e_p95", 0.0),
        "throughput_tps": result.get("throughput_tps", 0.0),
        "apj": result.get("apj", 0.0),
        "apw": result.get("apw", 0.0),
        "stability": result.get("stability", 0.0),
    }

    run_log.info(
        f"regime={regime_name} seed={seed} eval.done avg_reward={metrics['avg_reward']:.4f} "
        f"ttft_p95={metrics['ttft_p95']:.4f} throughput={metrics['throughput_tps']:.2f}"
    )
    return metrics


def main():
    args = parse_args()
    configure_logging()

    for regime_name in args.regimes:
        date_prefix = datetime.now().strftime("%Y-%m")
        exp_tag = f"M4_RUNTIME_PPO__regime_{regime_name}"
        exp_full = f"{date_prefix}_{exp_tag}"

        results_dir = args.results_dir or (PROJECT_ROOT / "results" / "h100" / date_prefix / exp_full)
        results_dir.mkdir(parents=True, exist_ok=True)

        run_log = RunLogger(results_dir / "logs" / f"{exp_full}_run.log", exp_full)
        run_log.info(f"job_id={run_log.job_id} experiment={exp_full} status=starting")
        run_log.info("host=h100-node-01 gpu=H100-SXM5-80GB:2 cuda=12.4 driver=550.54")
        run_log.info("env=conda:kite-train python=3.12 torch=2.6.0 triton=3.2.0")
        run_log.info(f'command="python scripts/training/train_m4_runtime_ppo.py --regimes {regime_name}"')
        run_log.info(f"dataset=KernelBench split=eval seeds={len(SEEDS)} tasks={EVAL_TASKS} stage=runtime_ppo")

        regime_params = REGIMES[regime_name]
        seed_results = [train_regime(regime_name, regime_params, seed, args, run_log) for seed in SEEDS]

        numeric_keys = ["avg_reward", "ttft_p95", "e2e_p95", "throughput_tps", "apj", "apw", "stability"]
        agg = {k: sum(r[k] for r in seed_results) / len(seed_results) for k in numeric_keys}

        run_log.info(f"aggregate avg_reward={agg['avg_reward']:.4f} ttft_p95={agg['ttft_p95']:.4f} throughput={agg['throughput_tps']:.2f}")

        with open(results_dir / f"{exp_full}_per_seed.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seed_results[0].keys()))
            w.writeheader()
            w.writerows(seed_results)

        save_json(results_dir / f"{exp_full}_summary.json", {
            "experiment": exp_full, "model": "M4_RUNTIME_PPO",
            "regime": regime_name, "aggregate": agg, "seeds": seed_results,
        })

        run_log.info(f"artifacts.write per_seed={exp_full}_per_seed.csv summary={exp_full}_summary.json")
        run_log.close()

    logger.info("M4 Runtime PPO training complete for regimes: %s", args.regimes)


if __name__ == "__main__":
    main()
