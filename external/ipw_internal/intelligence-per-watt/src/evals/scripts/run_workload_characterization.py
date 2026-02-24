#!/usr/bin/env python3
"""Run workload characterization across benchmark workloads — parallel edition.

Usage:
    # Single vLLM instance, sequential (backward compatible)
    python run_workload_characterization.py --workloads all --limit 40

    # Two vLLM instances, 16 concurrent workers
    python run_workload_characterization.py \
        --workloads all --limit 40 --workers 16 \
        --vllm-urls "http://localhost:8000,http://localhost:8001" \
        --model-name zai-org/GLM-4.7-Flash \
        --output-dir data/active_characterization
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("workload_characterization")

WORKLOAD_MAP = {
    "chat": "wildchat",
    "reasoning": "openthoughts",
    "rag": "hotpotqa",
    "agentic": "agentdata",
}

# Per-workload TraceCollector parameters (matching benchmark classes)
WORKLOAD_PARAMS: Dict[str, Dict[str, Any]] = {
    "chat": {"max_tokens": 8192, "temperature": 0.7},
    "reasoning": {"max_tokens": 16384, "temperature": 0.6},
    "rag": {"max_tokens": 4096, "temperature": 0.3},
    "agentic": {"max_tokens": 8192, "temperature": 0.3},
}


# ---------------------------------------------------------------------------
# Progress tracker (thread-safe)
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe progress tracking and streaming JSONL output."""

    def __init__(self, workload_limits: Dict[str, int], output_dir: Path):
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {w: 0 for w in workload_limits}
        self._limits = workload_limits
        self._total_done = 0
        self._total = sum(workload_limits.values())
        self._start_time = time.time()
        self._errors: Dict[str, int] = {w: 0 for w in workload_limits}

        # Pre-create JSONL files (truncate any existing)
        traces_dir = output_dir / "workload_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_paths: Dict[str, Path] = {}
        for wt in workload_limits:
            p = traces_dir / f"{wt}_traces.jsonl"
            p.write_text("")  # truncate
            self._jsonl_paths[wt] = p

    def append_trace(self, workload_type: str, trace_dict: Dict[str, Any]) -> None:
        """Thread-safe append trace to streaming JSONL."""
        line = json.dumps(trace_dict) + "\n"
        with self._lock:
            with open(self._jsonl_paths[workload_type], "a") as f:
                f.write(line)
            self._counts[workload_type] += 1
            self._total_done += 1

    def record_error(self, workload_type: str) -> None:
        with self._lock:
            self._errors[workload_type] += 1
            self._total_done += 1

    def log_completion(
        self,
        workload_type: str,
        sample_index: int,
        input_tokens: int,
        output_tokens: int,
        wall_clock_s: float,
        url_port: str,
    ) -> None:
        """Log per-query stats and periodic progress summary."""
        with self._lock:
            count = self._counts[workload_type]
            limit = self._limits[workload_type]
            total_done = self._total_done

        logger.info(
            f"[{workload_type} {count}/{limit}] sample={sample_index} | "
            f"{input_tokens} in + {output_tokens} out tokens | "
            f"{wall_clock_s:.1f}s | url={url_port}"
        )

        # Progress summary every 10 completions
        if total_done % 10 == 0 or total_done == self._total:
            with self._lock:
                parts = [f"{w}: {self._counts[w]}/{self._limits[w]}" for w in self._limits]
                elapsed = time.time() - self._start_time
            pct = 100.0 * total_done / self._total if self._total else 0
            logger.info(
                f"--- Progress: {total_done}/{self._total} ({pct:.1f}%) | "
                f"{' | '.join(parts)} | elapsed: {elapsed:.0f}s ---"
            )

    def jsonl_path(self, workload_type: str) -> Path:
        return self._jsonl_paths[workload_type]

    @property
    def error_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._errors)


# ---------------------------------------------------------------------------
# Sample preparation (extracts messages/params for TraceCollector)
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    """A single unit of work for the thread pool."""
    workload_type: str
    sample_index: int
    item_index: int  # global index for round-robin URL selection
    # Prepared query data
    query_id: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    conversation: List[Dict[str, str]] = field(default_factory=list)
    is_multi_turn: bool = False
    max_tokens: int = 8192
    temperature: float = 0.7


def _build_hotpotqa_prompt(sample: Any) -> str:
    """Build prompt with context (matches HotpotQABenchmark._build_prompt)."""
    if sample.context:
        return (
            "Answer the following question using the provided context. "
            "Give a short, direct answer.\n\n"
            f"Context:\n{sample.context}\n\n"
            f"Question: {sample.question}\n\n"
            "Answer:"
        )
    return (
        "Answer the following question. Give a short, direct answer.\n\n"
        f"Question: {sample.question}\n\n"
        "Answer:"
    )


def _build_agentdata_prompt(sample: Any) -> str:
    """Build prompt (matches AgentDataBenchmark._build_prompt)."""
    return (
        "You are a helpful AI agent. Complete the following task.\n\n"
        f"Task: {sample.task}\n\n"
        "Provide your response:"
    )


def load_all_samples(
    workloads: List[str], limit: int
) -> Dict[str, List[Any]]:
    """Load datasets sequentially (HuggingFace downloads not thread-safe)."""
    samples: Dict[str, List[Any]] = {}

    for wt in workloads:
        logger.info(f"Loading dataset for {wt} ({WORKLOAD_MAP[wt]})...")
        if wt == "chat":
            from evals.benchmarks.wildchat.dataset import load_wildchat_samples
            samples[wt] = list(load_wildchat_samples(limit=limit))
        elif wt == "reasoning":
            from evals.benchmarks.openthoughts.dataset import load_openthoughts_samples
            samples[wt] = list(load_openthoughts_samples(limit=limit))
        elif wt == "rag":
            from evals.benchmarks.hotpotqa.dataset import load_hotpotqa_samples
            samples[wt] = list(load_hotpotqa_samples(limit=limit))
        elif wt == "agentic":
            from evals.benchmarks.agentdata.dataset import load_agentdata_samples
            samples[wt] = list(load_agentdata_samples(limit=limit))
        else:
            logger.warning(f"Unknown workload type: {wt}, skipping")
            continue
        logger.info(f"  Loaded {len(samples[wt])} samples for {wt}")

    return samples


def build_work_items(
    samples: Dict[str, List[Any]],
) -> List[WorkItem]:
    """Convert loaded samples into WorkItems with prepared messages."""
    items: List[WorkItem] = []
    global_idx = 0

    for wt, sample_list in samples.items():
        params = WORKLOAD_PARAMS[wt]
        for si, sample in enumerate(sample_list):
            item = WorkItem(
                workload_type=wt,
                sample_index=sample.original_index,
                item_index=global_idx,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )

            if wt == "chat":
                item.query_id = str(sample.original_index)
                item.conversation = sample.conversation
                item.is_multi_turn = True
            elif wt == "reasoning":
                item.query_id = str(sample.original_index)
                prompt = sample.get_prompt()
                item.messages = [{"role": "user", "content": prompt}]
            elif wt == "rag":
                item.query_id = str(sample.original_index)
                prompt = _build_hotpotqa_prompt(sample)
                item.messages = [{"role": "user", "content": prompt}]
            elif wt == "agentic":
                item.query_id = str(sample.original_index)
                prompt = _build_agentdata_prompt(sample)
                item.messages = [{"role": "user", "content": prompt}]

            items.append(item)
            global_idx += 1

    return items


# ---------------------------------------------------------------------------
# Per-query worker
# ---------------------------------------------------------------------------

def process_single_query(
    item: WorkItem,
    vllm_url: str,
    model_name: str,
    progress: ProgressTracker,
) -> Optional[Dict[str, Any]]:
    """Execute a single query and stream results to JSONL."""
    from evals.telemetry.trace_collector import TraceCollector

    collector = TraceCollector(vllm_url=vllm_url, model_name=model_name)

    # Extract port for log readability
    url_port = ":" + vllm_url.rsplit(":", 1)[-1] if ":" in vllm_url else vllm_url

    try:
        if item.is_multi_turn:
            trace = collector.run_query_multi_turn_vllm(
                query_id=item.query_id,
                workload_type=item.workload_type,
                conversation=item.conversation,
                max_tokens=item.max_tokens,
                temperature=item.temperature,
            )
        else:
            trace = collector.run_query_direct_vllm(
                query_id=item.query_id,
                workload_type=item.workload_type,
                messages=item.messages,
                max_tokens=item.max_tokens,
                temperature=item.temperature,
            )

        trace_dict = trace.to_dict()
        progress.append_trace(item.workload_type, trace_dict)
        progress.log_completion(
            workload_type=item.workload_type,
            sample_index=item.sample_index,
            input_tokens=trace.total_input_tokens,
            output_tokens=trace.total_output_tokens,
            wall_clock_s=trace.total_wall_clock_s,
            url_port=url_port,
        )

        return {
            "workload_type": item.workload_type,
            "sample_index": item.sample_index,
            "trace": trace_dict,
            "completed": trace.completed,
        }

    except Exception as e:
        logger.error(
            f"[{item.workload_type}] sample={item.sample_index} FAILED: {e} | url={url_port}"
        )
        progress.record_error(item.workload_type)
        return {
            "workload_type": item.workload_type,
            "sample_index": item.sample_index,
            "error": str(e),
            "completed": False,
        }


# ---------------------------------------------------------------------------
# Post-processing: traces → profiles
# ---------------------------------------------------------------------------

def convert_traces_to_profiles(
    workloads: List[str],
    output_dir: Path,
    progress: ProgressTracker,
) -> Dict[str, Dict[str, Any]]:
    """Load streaming JSONL traces and convert to WorkloadProfiles."""
    from evals.telemetry.trace_collector import QueryTrace
    from evals.telemetry.trace_to_profile import TraceToProfile

    converter = TraceToProfile()
    profiles_dir = output_dir / "workload_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}

    for wt in workloads:
        jsonl_path = progress.jsonl_path(wt)
        if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
            logger.warning(f"No traces for {wt}, skipping profile conversion")
            results[wt] = {"error": "no traces collected"}
            continue

        traces = QueryTrace.load_jsonl(jsonl_path)
        logger.info(f"Converting {len(traces)} {wt} traces to profile...")

        benchmark_key = WORKLOAD_MAP[wt]
        profile = converter.convert(
            traces,
            workload_type=wt,
            source_dataset=f"{benchmark_key}_active",
        )

        profile_path = profiles_dir / f"{wt}_active_profile.json"
        profile.save(profile_path)
        logger.info(f"Saved active profile to {profile_path}")

        results[wt] = {
            "workload_type": wt,
            "num_traces": len(traces),
            "profile_path": str(profile_path),
            "traces_path": str(jsonl_path),
            "errors": progress.error_counts.get(wt, 0),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run parallel workload characterization"
    )
    parser.add_argument(
        "--workloads", default="all",
        help="Workloads to run (comma-separated or 'all'). Options: chat,reasoning,rag,agentic",
    )
    parser.add_argument(
        "--limit", type=int, default=40,
        help="Max queries per workload (default: 40)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Total concurrent workers (default: 16)",
    )
    parser.add_argument(
        "--vllm-urls", default="http://localhost:8000",
        help="Comma-separated vLLM endpoint URLs (default: http://localhost:8000)",
    )
    # Keep --vllm-url for backward compatibility
    parser.add_argument(
        "--vllm-url", default=None,
        help="(deprecated) Single vLLM URL. Use --vllm-urls instead.",
    )
    parser.add_argument(
        "--model-name", default="",
        help="Model name for vLLM requests",
    )
    parser.add_argument(
        "--output-dir", default="data/active_characterization",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Parse workloads
    if args.workloads == "all":
        workloads = list(WORKLOAD_MAP.keys())
    else:
        workloads = [w.strip() for w in args.workloads.split(",")]
        for w in workloads:
            if w not in WORKLOAD_MAP:
                parser.error(f"Unknown workload: {w}. Options: {list(WORKLOAD_MAP.keys())}")

    # Parse vLLM URLs (--vllm-url takes precedence if explicitly set)
    if args.vllm_url:
        urls = [args.vllm_url.strip()]
    else:
        urls = [u.strip() for u in args.vllm_urls.split(",") if u.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Parallel Workload Characterization ===")
    logger.info(f"Workloads: {workloads}")
    logger.info(f"Limit: {args.limit} per workload")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"vLLM URLs: {urls}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {output_dir}")

    # Step 1: Load all datasets sequentially
    logger.info("\n--- Loading datasets ---")
    all_samples = load_all_samples(workloads, args.limit)

    # Step 2: Build work items
    work_items = build_work_items(all_samples)
    total_items = len(work_items)
    logger.info(f"\nTotal work items: {total_items}")

    # Step 3: Create progress tracker
    workload_limits = {wt: len(all_samples[wt]) for wt in workloads if wt in all_samples}
    progress = ProgressTracker(workload_limits, output_dir)

    # Step 4: Execute in parallel
    logger.info(f"\n--- Starting parallel execution ({args.workers} workers) ---")
    start_time = time.time()
    results_list: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {
            executor.submit(
                process_single_query,
                item=item,
                vllm_url=urls[item.item_index % len(urls)],
                model_name=args.model_name,
                progress=progress,
            ): item
            for item in work_items
        }

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    results_list.append(result)
            except Exception as e:
                logger.error(
                    f"Unhandled exception for {item.workload_type} "
                    f"sample={item.sample_index}: {e}"
                )

    elapsed = time.time() - start_time
    logger.info(f"\n--- Execution complete: {elapsed:.1f}s for {total_items} queries ---")

    # Step 5: Convert traces to profiles
    logger.info("\n--- Converting traces to profiles ---")
    profile_results = convert_traces_to_profiles(workloads, output_dir, progress)

    # Step 6: Summary
    logger.info("\n=== Workload Characterization Summary ===")
    for wt, result in profile_results.items():
        if "error" in result:
            logger.info(f"  {wt}: FAILED - {result['error']}")
        else:
            logger.info(
                f"  {wt}: {result['num_traces']} traces "
                f"({result['errors']} errors) -> {result['profile_path']}"
            )

    logger.info(f"\nTotal time: {elapsed:.1f}s")
    logger.info(
        f"Throughput: {total_items / elapsed:.1f} queries/s"
        if elapsed > 0 else "Throughput: N/A"
    )

    # Save summary
    summary = {
        "config": {
            "workloads": workloads,
            "limit": args.limit,
            "workers": args.workers,
            "vllm_urls": urls,
            "model_name": args.model_name,
        },
        "timing": {
            "total_seconds": round(elapsed, 1),
            "queries_per_second": round(total_items / elapsed, 2) if elapsed > 0 else 0,
        },
        "results": profile_results,
    }
    summary_path = output_dir / "characterization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
