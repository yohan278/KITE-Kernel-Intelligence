"""Fast end-to-end mini pilot for scaling-law pipeline validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ipw.scaling_laws.aggregate import aggregate_summaries
from ipw.scaling_laws.fit import fit_scaling_laws
from ipw.scaling_laws.sweep import run_sweep


def _parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError(f"Expected at least one integer in '{text}'")
    return values


def _estimated_minutes(num_configs: int, sec_per_config: float) -> float:
    return (num_configs * max(sec_per_config, 1.0)) / 60.0


def run_mini_pilot(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    batches = _parse_csv_ints(args.batches)
    seq_ins = _parse_csv_ints(args.seq_ins)
    seq_outs = _parse_csv_ints(args.seq_outs)
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    if not quants:
        raise ValueError("At least one quant is required")

    estimated = _estimated_minutes(
        num_configs=len(quants) * len(batches) * len(seq_ins) * len(seq_outs),
        sec_per_config=args.estimate_sec_per_config,
    )
    print(
        "Mini pilot plan: "
        f"models=1 quants={len(quants)} batches={len(batches)} "
        f"seq_ins={len(seq_ins)} seq_outs={len(seq_outs)} "
        f"configs={len(quants) * len(batches) * len(seq_ins) * len(seq_outs)} "
        f"estimated_time~{estimated:.1f}m"
    )

    if not args.skip_sweep:
        sweep_args = argparse.Namespace(
            gpu_label=args.gpu_label,
            results_root=str(results_root),
            models=[args.model],
            quants=quants,
            batches=batches,
            seq_ins=seq_ins,
            seq_outs=seq_outs,
            num_samples=args.num_samples,
            max_queries=args.max_queries,
            dataset_param=list(args.dataset_param),
            client_param=list(args.client_param),
            quantized_dtype=args.quantized_dtype,
            ipw_bin=args.ipw_bin,
            sleep_seconds=args.sleep_seconds,
            stop_after_failures=args.stop_after_failures,
            disable_prefix_caching=args.disable_prefix_caching,
            resume=args.resume,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
        sweep_rc = run_sweep(sweep_args)
        if sweep_rc != 0:
            print(f"Mini pilot sweep failed (rc={sweep_rc}); skipping fit stages.")
            return sweep_rc
        if args.dry_run:
            print("Mini pilot dry-run completed (no aggregation/fitting executed).")
            return 0

    frame = aggregate_summaries(results_root=results_root)
    if frame.empty:
        print(f"No summary.json files found under {results_root}; nothing to fit.")
        return 1

    parquet_path = Path(args.output_parquet).resolve()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    print(f"Wrote aggregated data: {parquet_path} (rows={len(frame)})")

    if not args.run_fit:
        return 0

    report = fit_scaling_laws(
        frame,
        gpu=args.gpu_label if args.fit_gpu_from_label else args.fit_gpu,
        cv_folds=args.cv_folds,
        cv_seed=args.cv_seed,
        heldout_model_b=args.heldout_model_b,
    )
    report["metadata"]["mini_pilot"] = True
    report["metadata"]["results_root"] = str(results_root)

    report_path = Path(args.output_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(
        f"Wrote fit report: {report_path} "
        f"(rows_fit={report['metadata']['rows_fit']})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fast (~5 minute) end-to-end mini pilot for scaling-law "
            "pipeline validation (Step 3 -> 4 -> 5)."
        )
    )
    parser.add_argument("--gpu-label", required=True, help="GPU label for output layout")
    parser.add_argument("--model", required=True, help="Single model for mini pilot")
    parser.add_argument(
        "--results-root",
        default="results/mini_pilot",
        help="Mini pilot results root",
    )

    # Tiny default grid (4 configs): 1 quant x 2 batches x 2 seq_in x 1 seq_out.
    parser.add_argument("--quants", default="fp16", help="Comma-separated quant list")
    parser.add_argument("--batches", default="1,8", help="Comma-separated batch sizes")
    parser.add_argument("--seq-ins", default="256,1024", help="Comma-separated seq_in values")
    parser.add_argument("--seq-outs", default="128", help="Comma-separated seq_out values")
    parser.add_argument("--num-samples", type=int, default=4, help="Samples per config")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=4,
        help="Queries per config",
    )
    parser.add_argument(
        "--estimate-sec-per-config",
        type=float,
        default=60.0,
        help="Displayed runtime estimate only",
    )

    parser.add_argument(
        "--dataset-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra dataset params for sweep",
    )
    parser.add_argument(
        "--client-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra client params for sweep",
    )
    parser.add_argument(
        "--quantized-dtype",
        default="float16",
        help="dtype used for quantized modes",
    )
    parser.add_argument("--ipw-bin", default="ipw", help="ipw executable path")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--stop-after-failures", type=int, default=1)
    parser.add_argument(
        "--disable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable vLLM prefix caching during mini pilot "
            "(set --no-disable-prefix-caching to opt out)."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume mini pilot sweep if partial state exists",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip Step 3 and only run aggregation/fit on existing results",
    )

    parser.add_argument(
        "--run-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Step 5 fit after aggregation",
    )
    parser.add_argument(
        "--fit-gpu-from-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --gpu-label as fit GPU filter",
    )
    parser.add_argument("--fit-gpu", default=None, help="Override fit GPU filter")
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--heldout-model-b", type=float, default=14.0)
    parser.add_argument(
        "--output-parquet",
        default="results/mini_pilot/scaling_law_data.parquet",
    )
    parser.add_argument(
        "--output-report",
        default="results/mini_pilot/scaling_law_fit_report.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quantized_dtype is not None and not str(args.quantized_dtype).strip():
        args.quantized_dtype = None
    return run_mini_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
