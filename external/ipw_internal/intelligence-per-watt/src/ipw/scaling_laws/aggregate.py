"""Step 4: aggregate scaling-law sweep outputs into a tabular dataset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


_CONFIG_DIR_RE = re.compile(r"^B(?P<batch>\d+)_Sin(?P<seq_in>\d+)_Sout(?P<seq_out>\d+)$")
_MODEL_ACTIVE_RE = re.compile(r"a(?P<active>\d+(?:\.\d+)?)b", re.IGNORECASE)
_MODEL_SIZE_RE = re.compile(r"(?P<size>\d+(?:\.\d+)?)b", re.IGNORECASE)
_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(slots=True)
class ParsedPath:
    gpu: str
    model_slug: str
    quant: str
    batch: int
    seq_in: int
    seq_out: int


def _parse_config_path(summary_path: Path, results_root: Path) -> ParsedPath | None:
    rel = summary_path.relative_to(results_root)
    # expected: {gpu}/{model}/{quant}/B{batch}_Sin{seq_in}_Sout{seq_out}/summary.json
    if len(rel.parts) < 5:
        return None
    gpu, model_slug, quant, config_dir = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    match = _CONFIG_DIR_RE.match(config_dir)
    if not match:
        return None
    return ParsedPath(
        gpu=gpu,
        model_slug=model_slug,
        quant=quant.lower(),
        batch=int(match.group("batch")),
        seq_in=int(match.group("seq_in")),
        seq_out=int(match.group("seq_out")),
    )


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_active_params_b(
    model_name: str | None,
    model_slug: str,
    overrides: Mapping[str, float] | None,
) -> float | None:
    candidates = []
    if model_name:
        candidates.append(model_name)
    candidates.append(model_slug)

    if overrides:
        for candidate in candidates:
            if candidate in overrides:
                return float(overrides[candidate])

    for candidate in candidates:
        match = _MODEL_ACTIVE_RE.search(candidate)
        if match:
            return float(match.group("active"))

    for candidate in candidates:
        match = _MODEL_SIZE_RE.search(candidate)
        if match:
            return float(match.group("size"))

    return None


def _extract_quant(summary: Mapping[str, Any], parsed_quant: str) -> str:
    profiler_config = summary.get("profiler_config")
    if isinstance(profiler_config, Mapping):
        client_params = profiler_config.get("client_params")
        if isinstance(client_params, Mapping):
            quant = client_params.get("quantization")
            dtype = client_params.get("dtype")
            if isinstance(quant, str) and quant.strip():
                return quant.strip().lower()
            if isinstance(dtype, str) and dtype.strip():
                return dtype.strip().lower()
    return parsed_quant


def _bytes_per_param(quant: str) -> float | None:
    quant_norm = quant.strip().lower()
    return _BYTES_PER_PARAM.get(quant_norm)


def aggregate_summaries(
    *,
    results_root: Path,
    model_size_overrides: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(results_root.rglob("summary.json")):
        parsed = _parse_config_path(summary_path, results_root)
        if parsed is None:
            continue

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, Mapping):
            continue

        model_name = summary.get("model")
        model_name = model_name if isinstance(model_name, str) else None
        quant = _extract_quant(summary, parsed.quant)
        phase_summary = summary.get("phase_summary")
        if not isinstance(phase_summary, Mapping):
            phase_summary = {}
        prefill = phase_summary.get("prefill")
        decode = phase_summary.get("decode")
        prefill = prefill if isinstance(prefill, Mapping) else {}
        decode = decode if isinstance(decode, Mapping) else {}

        active_params_b = _extract_active_params_b(
            model_name=model_name,
            model_slug=parsed.model_slug,
            overrides=model_size_overrides,
        )
        bpp = _bytes_per_param(quant)

        prefill_duration_ms = _first_float(prefill.get("mean_duration_ms"))
        decode_duration_ms = _first_float(decode.get("mean_duration_ms"))

        rows.append(
            {
                "summary_path": str(summary_path),
                "gpu": parsed.gpu,
                "model": model_name or parsed.model_slug,
                "model_slug": parsed.model_slug,
                "quant": quant,
                "batch_size": parsed.batch,
                "seq_in": parsed.seq_in,
                "seq_out": parsed.seq_out,
                "active_params_b": active_params_b,
                "bytes_per_param": bpp,
                "E_prefill_j": _first_float(prefill.get("total_energy_j")),
                "E_decode_j": _first_float(decode.get("total_energy_j")),
                "E_total_j": _first_float(
                    (summary.get("totals") or {}).get("total_energy_j")
                    if isinstance(summary.get("totals"), Mapping)
                    else None
                ),
                "P_prefill_w": _first_float(prefill.get("mean_power_w")),
                "P_decode_w": _first_float(decode.get("mean_power_w")),
                "T_prefill_s": (
                    prefill_duration_ms / 1000.0
                    if prefill_duration_ms is not None
                    else None
                ),
                "T_decode_s": (
                    decode_duration_ms / 1000.0
                    if decode_duration_ms is not None
                    else None
                ),
                "E_per_input_token_j": _first_float(
                    prefill.get("mean_energy_per_input_token_j")
                ),
                "E_per_output_token_j": _first_float(
                    decode.get("mean_energy_per_output_token_j")
                ),
            }
        )

    return pd.DataFrame(rows)


def _load_model_size_overrides(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Model-size map must be a JSON object")
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model size override for '{key}': {value}") from exc
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Step-3 results into scaling_law_data.parquet."
    )
    parser.add_argument("--results-root", required=True, help="Root results directory")
    parser.add_argument(
        "--output",
        default="scaling_law_data.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path for inspection",
    )
    parser.add_argument(
        "--model-size-map",
        default=None,
        help=(
            "Optional JSON map from model name/slug to active_params_b "
            "(used when size cannot be inferred from name)."
        ),
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows missing active_params_b or bytes_per_param",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).resolve()
    overrides = _load_model_size_overrides(
        Path(args.model_size_map).resolve() if args.model_size_map else None
    )
    frame = aggregate_summaries(
        results_root=results_root,
        model_size_overrides=overrides,
    )
    if args.drop_missing:
        frame = frame.dropna(subset=["active_params_b", "bytes_per_param"])

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    if args.output_csv:
        csv_path = Path(args.output_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)

    print(f"Wrote {len(frame)} rows to {output_path}")
    if len(frame) > 0:
        print(
            "Coverage: "
            f"gpus={frame['gpu'].nunique()} "
            f"models={frame['model'].nunique()} "
            f"quants={frame['quant'].nunique()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

