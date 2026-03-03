#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import copy

import yaml


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _base_train() -> dict[str, Any]:
    return {
        "train": {
            "mode": "throughput",
            "epochs": 1,
            "energy_aware": False,
            "group_size": 4,
            "batch_size": 4,
            "max_completion_length": 256,
            "max_tasks": 64,
            "eval_num_correct_trials": 1,
            "eval_num_perf_trials": 5,
            "failure_log_every_steps": 5,
            "reward": {
                "alpha": 1.0,
                "beta": 0.0,
                "gamma_latency": 0.25,
                "delta_power": 0.01,
                "eta_runtime": 0.10,
                "correctness_bonus": 0.0,
                "compile_fail": -1.0,
                "incorrect": -0.5,
                "oom_penalty": 0.5,
                "sla_latency_s": 1.0,
                "ipw_blend_weight": 0.0,
            },
            "telemetry": {
                "trace_dir": "./data/telemetry/runs",
                "ipw_profile_dir": None,
                "allow_synthetic_fallback": True,
            },
        }
    }


def _seed_suffix(seed: int) -> str:
    return f"seed{seed}"


def generate_sg0(out_dir: Path) -> list[Path]:
    base = _base_train()
    path = out_dir / "base" / "template.yaml"
    _write_yaml(path, base)
    return [path]


def generate_sg1(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    for seed in (11, 22, 33):
        cfg = _base_train()
        cfg["train"]["mode"] = "throughput"
        cfg["train"]["energy_aware"] = False
        p = out_dir / "throughput" / f"throughput_{_seed_suffix(seed)}.yaml"
        _write_yaml(p, cfg)
        files.append(p)
    return files


def generate_sg2(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    for seed in (11, 22, 33):
        cfg = _base_train()
        cfg["train"]["mode"] = "energy_aware"
        cfg["train"]["energy_aware"] = True
        cfg["train"]["reward"]["beta"] = 0.5
        p = out_dir / "energy" / f"energy_{_seed_suffix(seed)}.yaml"
        _write_yaml(p, cfg)
        files.append(p)
    return files


def generate_sg3(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    for lam in (0.0, 0.1, 0.25, 0.5):
        for seed in (11, 22, 33):
            cfg = _base_train()
            cfg["train"]["mode"] = "energy_aware"
            cfg["train"]["energy_aware"] = True
            cfg["train"]["reward"]["beta"] = 0.5
            cfg["train"]["reward"]["ipw_blend_weight"] = lam
            p = out_dir / "ipw_blend" / f"ipw_lambda{lam}_{_seed_suffix(seed)}.yaml"
            _write_yaml(p, cfg)
            files.append(p)
    return files


def generate_sg4(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    ablations = [
        ("no_speedup", {"alpha": 0.0}),
        ("no_runtime", {"eta_runtime": 0.0}),
        ("no_joules", {"beta": 0.0}),
        ("no_power", {"delta_power": 0.0}),
        ("no_sla", {"gamma_latency": 0.0}),
        ("no_ipw", {"ipw_blend_weight": 0.0}),
    ]
    for name, overrides in ablations:
        for seed in (11, 22, 33):
            cfg = _base_train()
            cfg["train"]["mode"] = "energy_aware"
            cfg["train"]["energy_aware"] = True
            cfg["train"]["reward"]["beta"] = 0.5
            cfg["train"]["reward"]["ipw_blend_weight"] = 0.25
            cfg["train"]["reward"].update(overrides)
            p = out_dir / "abl_reward" / f"{name}_{_seed_suffix(seed)}.yaml"
            _write_yaml(p, cfg)
            files.append(p)
    return files


def generate_sg5(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    scales = (16, 64, 128, 270)
    for scale in scales:
        for seed in (11, 22, 33):
            cfg = _base_train()
            cfg["train"]["mode"] = "energy_aware"
            cfg["train"]["energy_aware"] = True
            cfg["train"]["reward"]["beta"] = 0.5
            cfg["train"]["reward"]["ipw_blend_weight"] = 0.25
            cfg["train"]["max_tasks"] = scale
            p = out_dir / "abl_scale" / f"scale{scale}_{_seed_suffix(seed)}.yaml"
            _write_yaml(p, cfg)
            files.append(p)
    return files


def generate_sg6(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    budgets = [
        ("tight", 3, 2, 256),
        ("medium", 5, 5, 512),
        ("large", 5, 10, 768),
    ]
    for name, perf_trials, measure_iters, max_len in budgets:
        for seed in (11, 22, 33):
            cfg = _base_train()
            cfg["train"]["mode"] = "energy_aware"
            cfg["train"]["energy_aware"] = True
            cfg["train"]["reward"]["beta"] = 0.5
            cfg["train"]["reward"]["ipw_blend_weight"] = 0.25
            cfg["train"]["eval_num_perf_trials"] = perf_trials
            cfg["train"]["max_completion_length"] = max_len
            cfg["train"]["measure_iters"] = measure_iters
            p = out_dir / "abl_budget" / f"{name}_{_seed_suffix(seed)}.yaml"
            _write_yaml(p, cfg)
            files.append(p)
    return files


def generate_sg7(out_dir: Path) -> list[Path]:
    runtime_plan = {
        "runtime": {
            "seeds": [11, 22, 33],
            "episodes": 40,
            "horizon": 10,
        },
        "hrl": {
            "seeds": [11, 22, 33],
            "rounds": 2,
        },
    }
    path = out_dir / "runtime" / "runtime_hrl_plan.yaml"
    _write_yaml(path, runtime_plan)
    return [path]


GENERATORS = {
    "sg0": generate_sg0,
    "sg1": generate_sg1,
    "sg2": generate_sg2,
    "sg3": generate_sg3,
    "sg4": generate_sg4,
    "sg5": generate_sg5,
    "sg6": generate_sg6,
    "sg7": generate_sg7,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=sorted(GENERATORS.keys()))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    out_dir = args.root / "configs" / "exp"
    files = GENERATORS[args.agent](out_dir)

    state_dir = args.root / "outputs" / "agent_queue" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{args.agent}.done").write_text("ok\n", encoding="utf-8")

    print(f"[{args.agent}] wrote {len(files)} file(s)")
    for f in files:
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
