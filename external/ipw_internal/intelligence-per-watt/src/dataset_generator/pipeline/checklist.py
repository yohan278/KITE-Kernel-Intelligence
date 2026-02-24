"""Operator profiling checklist — tracks which operators have been profiled."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set


OPERATOR_CHECKLIST: Dict[str, List[str]] = {
    "token_ops": [
        "linear_qkv", "linear_o", "mlp_up", "mlp_gate", "mlp_down",
        "rmsnorm", "layernorm", "silu_activation", "gelu_activation",
        "embedding", "lm_head", "residual_add", "rotary_embedding",
        "softmax", "dropout", "cross_entropy_loss",
    ],
    "attention": [
        "attention_prefill", "attention_decode", "kv_cache_append",
        "kv_cache_evict", "sliding_window_attention", "mqa_gqa_expansion",
    ],
    "moe": [
        "moe_router", "moe_expert_mlp", "moe_combine", "expert_dispatch",
        "expert_combine", "load_balancing_loss", "capacity_factor_overhead",
        "shared_expert_mlp",
    ],
    "ssm": [
        "ssm_scan", "ssm_conv1d", "ssm_discretize", "ssm_gate",
        "ssm_residual_mix",
    ],
    "mtp": [
        "mtp_head_forward", "mtp_loss", "mtp_token_merge",
        "speculative_draft", "speculative_verify", "draft_accept_reject",
    ],
    "sampling": [
        "temperature_scaling", "top_p_filter", "top_k_filter",
        "repetition_penalty", "logit_processor_chain", "multinomial_sample",
        "beam_search_step",
    ],
    "communication": [
        "allreduce", "allgather", "reduce_scatter", "send_recv_p2p",
        "pipeline_bubble_idle", "tensor_parallel_split",
        "pipeline_stage_forward",
    ],
    "cpu_host": [
        "cpu_offload_transfer", "gpu_mem_alloc", "pcie_h2d_copy",
        "pcie_d2h_copy", "scheduler_overhead", "tokenizer_encode",
        "tokenizer_decode", "dynamic_batching_overhead",
    ],
    "agentic": [
        "tool_web_search", "tool_calculator", "tool_code_interpreter",
        "tool_file_read", "tool_file_write", "tool_bm25_retrieval",
        "tool_faiss_retrieval", "tool_api_call", "tool_bash_exec",
    ],
}


def get_total_operators() -> int:
    """Return the total number of operators across all profiler categories."""
    return sum(len(ops) for ops in OPERATOR_CHECKLIST.values())


def _scan_csv_for_operators(csv_path: Path) -> Set[str]:
    """Extract unique operator names from a CSV file."""
    operators: Set[str] = set()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("operator_name") or row.get("tool_name") or row.get("operation")
                if name:
                    operators.add(name)
    except (OSError, KeyError):
        pass
    return operators


def get_checklist_status(profiling_dir: Path) -> Dict[str, Dict[str, bool]]:
    """Scan profiling CSV files and return which operators have been profiled.

    Returns:
        Dict mapping profiler category -> {operator_name -> profiled_bool}
    """
    profiling_dir = Path(profiling_dir)

    # Collect all operator names found in any CSV under the profiling dir
    found_operators: Set[str] = set()
    if profiling_dir.is_dir():
        for csv_file in profiling_dir.rglob("*.csv"):
            found_operators.update(_scan_csv_for_operators(csv_file))

    status: Dict[str, Dict[str, bool]] = {}
    for category, operators in OPERATOR_CHECKLIST.items():
        status[category] = {}
        for op in operators:
            status[category][op] = op in found_operators

    return status


def print_checklist(status: Dict[str, Dict[str, bool]]) -> str:
    """Format the checklist status as markdown-style output with checkmarks.

    Returns:
        Formatted string for display.
    """
    lines: List[str] = []
    total = 0
    done = 0

    for category, operators in status.items():
        cat_done = sum(1 for v in operators.values() if v)
        cat_total = len(operators)
        total += cat_total
        done += cat_done

        lines.append(f"## {category} ({cat_done}/{cat_total})")
        for op_name, profiled in operators.items():
            mark = "[x]" if profiled else "[ ]"
            lines.append(f"  - {mark} {op_name}")
        lines.append("")

    lines.insert(0, f"# Operator Checklist ({done}/{total} profiled)\n")
    return "\n".join(lines)
