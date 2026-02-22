#!/usr/bin/env python3
"""Phase 4 training wrapper for throughput or energy-aware kernel GRPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.cli import main as kite_main
from kite.utils.serialization import load_yaml, save_json

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _run_with_progress(
    label: str,
    run_fn,
    heartbeat_seconds: float = 15.0,
    show_progress: bool = True,
) -> int:
    result: dict[str, int] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            result["rc"] = int(run_fn())
        except BaseException as exc:
            error["exc"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()

    if tqdm is not None and show_progress:
        pbar = tqdm(total=0, desc=label, bar_format="{desc} | elapsed {elapsed} | {postfix}")
        while not done.wait(timeout=max(1.0, heartbeat_seconds)):
            pbar.set_postfix_str("running")
            pbar.update(0)
        pbar.set_postfix_str("done")
        pbar.update(0)
        pbar.close()
    else:
        start = time.time()
        print(f"[train_rl] started: {label}", flush=True)
        while not done.wait(timeout=max(1.0, heartbeat_seconds)):
            elapsed = int(time.time() - start)
            print(f"[train_rl] {label} running... {elapsed}s elapsed", flush=True)
        elapsed = int(time.time() - start)
        print(f"[train_rl] finished: {label} ({elapsed}s)", flush=True)

    t.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("rc", 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output", type=Path, default=Path("./checkpoints/kernel_grpo"))
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if "sweep" in cfg:
        sweep_cfg = cfg["sweep"]
        combos = sweep_cfg.get("alpha_beta", [])
        epochs = int(sweep_cfg.get("epochs", 3))
        out_root = Path(sweep_cfg.get("output_dir", "./outputs/sweeps"))
        records = []
        if tqdm is not None and not args.no_progress:
            combo_iter = tqdm(
                list(enumerate(combos)),
                total=len(combos),
                desc="train_rl sweep",
                dynamic_ncols=True,
            )
        else:
            combo_iter = enumerate(combos)

        for idx, combo in combo_iter:
            alpha, beta = float(combo[0]), float(combo[1])
            out_dir = out_root / f"alpha_{alpha}_beta_{beta}"
            cmd = [
                "train",
                "kernel-grpo",
                "--kernelbench-root",
                str(args.kernelbench_root),
                "--output",
                str(out_dir),
                "--epochs",
                str(epochs),
                "--energy-aware",
            ]
            label = f"sweep {idx + 1}/{len(combos)} alpha={alpha} beta={beta}"
            if tqdm is not None and not args.no_progress:
                combo_iter.set_postfix(alpha=alpha, beta=beta)
            rc = _run_with_progress(
                label=label,
                run_fn=lambda cmd=cmd: kite_main(cmd),
                heartbeat_seconds=args.heartbeat_seconds,
                show_progress=not args.no_progress,
            )
            records.append({"alpha": alpha, "beta": beta, "return_code": rc, "output_dir": str(out_dir)})
        save_json(out_root / "sweep_manifest.json", {"runs": records})
        return 0

    train = cfg.get("train", {})
    energy_aware = bool(train.get("energy_aware", False))
    epochs = int(train.get("epochs", 3))
    cmd = [
        "train",
        "kernel-grpo",
        "--kernelbench-root",
        str(args.kernelbench_root),
        "--output",
        str(args.output),
        "--epochs",
        str(epochs),
    ]
    if energy_aware:
        cmd.append("--energy-aware")
    label = f"kernel-grpo epochs={epochs} energy_aware={energy_aware}"
    return _run_with_progress(
        label=label,
        run_fn=lambda: kite_main(cmd),
        heartbeat_seconds=args.heartbeat_seconds,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    raise SystemExit(main())
