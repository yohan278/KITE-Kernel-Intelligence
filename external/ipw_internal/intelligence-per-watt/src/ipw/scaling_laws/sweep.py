"""Step 3: measurement sweep runner for scaling-law experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence


OOM_PATTERNS = (
    r"out of memory",
    r"cuda out of memory",
    r"oom",
    r"cublas_status_alloc_failed",
    r"std::bad_alloc",
    r"insufficient memory",
)
_OOM_REGEX = re.compile("|".join(OOM_PATTERNS), re.IGNORECASE)


def _slugify_model(model: str) -> str:
    text = model.strip().lower().replace("/", "-")
    return re.sub(r"[^a-z0-9_.-]+", "-", text).strip("-") or "model"


def _normalize_quant(quant: str) -> str:
    return quant.strip().lower()


def _now_epoch() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_kv_pairs(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE, got '{item}'")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty key in '{item}'")
        parsed[key] = value.strip()
    return parsed


def _apply_default_client_params(
    client_params: Mapping[str, str],
    *,
    disable_prefix_caching: bool,
) -> dict[str, str]:
    merged = dict(client_params)

    if disable_prefix_caching:
        has_prefix_caching = any(
            key.strip().lower() == "enable_prefix_caching"
            for key in merged.keys()
        )
        if not has_prefix_caching:
            # Synthetic prompts are often repeated in scaling-law sweeps.
            # Disable vLLM prefix-cache reuse by default so prefill energy
            # remains measurable and not silently amortized.
            merged["enable_prefix_caching"] = "false"

    return merged


def _quant_client_params(quant: str, quantized_dtype: str | None) -> dict[str, str]:
    quant_norm = _normalize_quant(quant)
    params: dict[str, str] = {}

    if quant_norm in {"fp16", "float16"}:
        params["dtype"] = "float16"
    elif quant_norm in {"bf16", "bfloat16"}:
        params["dtype"] = "bfloat16"
    elif quant_norm in {"fp8", "int8", "int4"}:
        params["quantization"] = quant_norm
        if quantized_dtype:
            params["dtype"] = quantized_dtype
    else:
        # Best effort: treat unknown quant string as dtype.
        params["dtype"] = quant_norm

    return params


def _is_oom_output(text: str) -> bool:
    return bool(_OOM_REGEX.search(text))


@dataclass(frozen=True)
class SweepPoint:
    model: str
    quant: str
    batch: int
    seq_in: int
    seq_out: int

    @property
    def config_id(self) -> str:
        return (
            f"{self.model}|{self.quant}|"
            f"B{self.batch}_Sin{self.seq_in}_Sout{self.seq_out}"
        )


def iter_points(
    models: Sequence[str],
    quants: Sequence[str],
    batches: Sequence[int],
    seq_ins: Sequence[int],
    seq_outs: Sequence[int],
) -> Iterator[SweepPoint]:
    for model in models:
        for quant in quants:
            for seq_out in seq_outs:
                for seq_in in seq_ins:
                    for batch in batches:
                        yield SweepPoint(
                            model=model,
                            quant=_normalize_quant(quant),
                            batch=batch,
                            seq_in=seq_in,
                            seq_out=seq_out,
                        )


def output_dir_for(results_root: Path, gpu_label: str, point: SweepPoint) -> Path:
    model_slug = _slugify_model(point.model)
    return (
        results_root
        / gpu_label
        / model_slug
        / point.quant
        / f"B{point.batch}_Sin{point.seq_in}_Sout{point.seq_out}"
    )


def _tail_lines(text: str, max_lines: int = 30) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _find_summary_path(config_output_dir: Path) -> Path | None:
    """Locate summary.json for a config, tolerating nested output layouts."""
    direct = config_output_dir / "summary.json"
    if direct.exists():
        return direct

    candidates = sorted(config_output_dir.rglob("summary.json"))
    if not candidates:
        return None

    # Prefer the shallowest summary under this config directory.
    return min(candidates, key=lambda path: len(path.relative_to(config_output_dir).parts))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": _now_iso(),
            "entries": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, Mapping):
        raise RuntimeError(f"Invalid state file: {path}")
    entries = loaded.get("entries")
    if not isinstance(entries, Mapping):
        loaded["entries"] = {}
    return dict(loaded)


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _build_profile_command(
    *,
    ipw_bin: str,
    point: SweepPoint,
    out_dir: Path,
    num_samples: int,
    max_queries: int | None,
    quantized_dtype: str | None,
    dataset_params: Mapping[str, str],
    client_params: Mapping[str, str],
) -> list[str]:
    command = [
        ipw_bin,
        "profile",
        "--client",
        "vllm",
        "--model",
        point.model,
        "--dataset",
        "synthetic",
        "--dataset-param",
        f"input_tokens={point.seq_in}",
        "--dataset-param",
        f"num_samples={num_samples}",
        "--client-param",
        f"sampling_max_tokens={point.seq_out}",
        "--max-concurrency",
        str(point.batch),
        "--max-queries",
        str(max_queries if max_queries is not None else num_samples),
        "--phased",
        "--output-dir",
        str(out_dir),
    ]

    for key, value in _quant_client_params(point.quant, quantized_dtype).items():
        command.extend(["--client-param", f"{key}={value}"])

    for key, value in dataset_params.items():
        command.extend(["--dataset-param", f"{key}={value}"])
    for key, value in client_params.items():
        command.extend(["--client-param", f"{key}={value}"])

    return command


def run_sweep(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    state_path = results_root / ".sweep_state.json"
    log_path = results_root / "sweep.log"
    state = _load_state(state_path)
    entries: dict[str, Any] = dict(state.get("entries", {}))

    dataset_param_overrides = _parse_kv_pairs(args.dataset_param)
    client_param_overrides = _apply_default_client_params(
        _parse_kv_pairs(args.client_param),
        disable_prefix_caching=args.disable_prefix_caching,
    )

    points = list(
        iter_points(
            models=args.models,
            quants=args.quants,
            batches=args.batches,
            seq_ins=args.seq_ins,
            seq_outs=args.seq_outs,
        )
    )
    total = len(points)
    if total == 0:
        print("No sweep points configured; nothing to run.", file=sys.stderr)
        return 1

    failures = 0
    completed = 0
    oom = 0
    skipped = 0

    for idx, point in enumerate(points, start=1):
        out_dir = output_dir_for(results_root, args.gpu_label, point)
        summary_path = _find_summary_path(out_dir)
        config_id = point.config_id
        entry = dict(entries.get(config_id, {}))
        status = str(entry.get("status", "")).lower()

        if summary_path is not None:
            status = "completed"
            entry["status"] = status
            entry["last_returncode"] = 0
            entry["updated_at"] = _now_iso()
            entry["output_dir"] = str(out_dir)
            entry["summary_path"] = str(summary_path)
            entries[config_id] = entry
            skipped += 1
            print(f"[{idx}/{total}] skip (already complete): {config_id}")
            continue

        if args.resume and status in {"completed", "oom"}:
            skipped += 1
            print(f"[{idx}/{total}] skip ({status}): {config_id}")
            continue
        if args.resume and status == "failed" and not args.retry_failed:
            skipped += 1
            print(f"[{idx}/{total}] skip (failed; use --retry-failed): {config_id}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        command = _build_profile_command(
            ipw_bin=args.ipw_bin,
            point=point,
            out_dir=out_dir,
            num_samples=args.num_samples,
            max_queries=args.max_queries,
            quantized_dtype=args.quantized_dtype,
            dataset_params=dataset_param_overrides,
            client_params=client_param_overrides,
        )

        command_preview = " ".join(subprocess.list2cmdline([c]) for c in command)
        print(f"[{idx}/{total}] run: {config_id}")
        print(f"  cmd: {command_preview}")

        if args.dry_run:
            skipped += 1
            continue

        started = _now_epoch()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed_s = _now_epoch() - started
        combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        output_excerpt = _tail_lines(combined_output)
        summary_path = _find_summary_path(out_dir)

        log_block = (
            f"[{_now_iso()}] config={config_id} rc={proc.returncode} "
            f"elapsed_s={elapsed_s:.2f}\n{output_excerpt}\n"
        )
        _append_log(log_path, log_block)

        result_status: str
        if proc.returncode == 0 and summary_path is not None:
            result_status = "completed"
            completed += 1
            print(f"  -> completed ({elapsed_s:.1f}s)")
        elif _is_oom_output(combined_output):
            result_status = "oom"
            oom += 1
            print(f"  -> OOM (marked infeasible, {elapsed_s:.1f}s)")
        else:
            result_status = "failed"
            failures += 1
            print(f"  -> failed rc={proc.returncode} ({elapsed_s:.1f}s)")

        entry.update(
            {
                "status": result_status,
                "attempts": int(entry.get("attempts", 0)) + 1,
                "last_returncode": proc.returncode,
                "updated_at": _now_iso(),
                "output_dir": str(out_dir),
                "summary_path": str(summary_path) if summary_path is not None else "",
                "elapsed_s": elapsed_s,
                "error_excerpt": output_excerpt if result_status != "completed" else "",
            }
        )
        entries[config_id] = entry
        state["entries"] = entries
        state["last_updated_at"] = _now_iso()
        _save_state(state_path, state)

        if args.stop_after_failures > 0 and failures >= args.stop_after_failures:
            print(
                f"Stopping early after {failures} failures "
                f"(threshold={args.stop_after_failures}).",
                file=sys.stderr,
            )
            break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    state["entries"] = entries
    state["last_updated_at"] = _now_iso()
    _save_state(state_path, state)

    print(
        "Sweep summary: "
        f"completed={completed}, oom={oom}, failed={failures}, skipped={skipped}, total={total}"
    )
    return 0 if failures == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step-3 scaling-law measurement sweep with checkpointing and "
            "OOM-aware skipping."
        )
    )
    parser.add_argument("--gpu-label", required=True, help="Hardware label (e.g., h100)")
    parser.add_argument("--results-root", required=True, help="Root output directory")
    parser.add_argument("--models", nargs="+", required=True, help="Model ids")
    parser.add_argument("--quants", nargs="+", required=True, help="Quant modes")
    parser.add_argument("--batches", nargs="+", type=int, required=True, help="Batch sizes")
    parser.add_argument(
        "--seq-ins",
        nargs="+",
        type=int,
        required=True,
        help="Input sequence lengths",
    )
    parser.add_argument(
        "--seq-outs",
        nargs="+",
        type=int,
        required=True,
        help="Output sequence lengths",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Synthetic dataset sample count per config",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional override for --max-queries (defaults to num-samples)",
    )
    parser.add_argument(
        "--dataset-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra dataset parameter passed to ipw profile",
    )
    parser.add_argument(
        "--client-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra client parameter passed to ipw profile",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable vLLM prefix caching by default for scaling-law sweeps "
            "(set --no-disable-prefix-caching to opt out)."
        ),
    )
    parser.add_argument(
        "--quantized-dtype",
        default="float16",
        help=(
            "dtype to pass for quantized modes (fp8/int8/int4). "
            "Use empty string to omit."
        ),
    )
    parser.add_argument("--ipw-bin", default="ipw", help="ipw executable path")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between runs")
    parser.add_argument(
        "--stop-after-failures",
        type=int,
        default=0,
        help="Stop early after N failed (non-OOM) configs; 0 disables",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from state file and skip completed/OOM points",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, retry configs previously marked as failed",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quantized_dtype is not None and not str(args.quantized_dtype).strip():
        args.quantized_dtype = None
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
