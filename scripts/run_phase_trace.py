#!/usr/bin/env python3
"""Phase trace experiment runner for runtime-only and hierarchical controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.hierarchical.high_level_kernel_choice import HighLevelKernelChooser
from kite.hierarchical.low_level_runtime_policy import LowLevelRuntimePolicy
from kite.runtime_control.runtime_env import PhasePoint, PhaseTraceRuntimeEnv
from kite.utils.serialization import load_yaml, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("./configs/hierarchical.yaml"))
    parser.add_argument("--output", type=Path, default=Path("./outputs/phase_trace/phase_trace.json"))
    args = parser.parse_args()

    cfg = load_yaml(args.config).get("hierarchical", {})
    regime_cfg = cfg.get("phase_trace", {}).get("regimes", [])
    phase_trace = [
        PhasePoint(
            phase_id=str(r.get("name", "mixed")),
            queue_depth=int(r.get("queue_depth", 16)),
            phase_ratio=float(r.get("phase_ratio", 0.5)),
        )
        for r in regime_cfg
    ] or [PhasePoint("mixed", 16, 0.5)]

    env = PhaseTraceRuntimeEnv(phase_trace=phase_trace)
    hi = HighLevelKernelChooser()
    lo = LowLevelRuntimePolicy()

    state = env.reset()
    rows = []
    for step in range(max(4, len(phase_trace) * 2)):
        choice = hi.choose(state, explore=False)
        action = lo.act(state, explore=False)
        out = env.step(state, action)
        rows.append(
            {
                "step": step,
                "phase_id": state.phase_id,
                "kernel_family": choice.family,
                "action": action,
                "throughput_tps": out.throughput_tps,
                "apj": out.apj,
                "apw": out.apw,
                "latency_s": out.latency_s,
                "energy_j": out.energy_j,
            }
        )
        state = out.next_state

    payload = {"num_steps": len(rows), "rows": rows}
    save_json(args.output, payload)
    print(f"Saved phase trace run to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
