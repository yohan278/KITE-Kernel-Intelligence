"""Shared utilities for evaluation benchmarks."""

from __future__ import annotations

from typing import List, Optional


def print_task_header(
    *,
    index: int,
    total: int,
    task_id: str,
    question: str,
    metadata: str = "",
    verbose: bool = False,
) -> None:
    """Print a unified task header for eval benchmarks.

    Default mode: one-line summary with task ID, metadata, and truncated question.
    Verbose mode (-v): full boxed header with the complete question text.
    """
    meta_suffix = f"  |  {metadata}" if metadata else ""

    if verbose:
        print(f"\n{'═' * 76}")
        print(f"  TASK {index + 1}/{total}  |  {task_id}{meta_suffix}")
        print(f"{'═' * 76}")
        for line in question.split('\n'):
            print(f"  {line}")
        print()
    else:
        # Truncate question for compact display
        q = question.replace('\n', ' ')
        if len(q) > 200:
            q = q[:200] + "..."
        meta_paren = f" ({metadata})" if metadata else ""
        print(f"\n[{index + 1}/{total}] {task_id}{meta_paren}")
        print(f"  {q}")


def print_result_box(
    *,
    status: str,
    latency_seconds: float,
    num_turns: int,
    tools_used: List[str],
    tools_successful: int,
    tools_failed: int,
    predicted: Optional[str] = None,
    expected: Optional[str] = None,
    model_response: Optional[str] = None,
    raw_responses: Optional[List[str]] = None,
    extra_lines: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """Print a unified result box for eval benchmarks.

    Default mode: basic summary (status, latency, turns, tool counts, predicted/expected).
    Verbose mode (-v): adds tool name breakdown, extra benchmark lines,
    and model response when orchestrator was not used.
    """
    print(f"  ┌─ Result {'─' * 63}")

    # Status line
    turn_s = "turn" if num_turns == 1 else "turns"
    tool_count = len(tools_used)
    tool_s = "call" if tool_count == 1 else "calls"
    print(f"  │ {status}  |  {latency_seconds:.1f}s  |  "
          f"{num_turns} {turn_s}  |  "
          f"{tool_count} tool {tool_s} ({tools_successful} ok, {tools_failed} failed)")

    # Tool name breakdown (verbose only)
    if verbose and tools_used:
        tool_counts: dict[str, int] = {}
        for t in tools_used:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        tools_str = ", ".join(
            f"{t} x{c}" if c > 1 else t for t, c in tool_counts.items()
        )
        print(f"  │ Tools:     {tools_str}")

    # Predicted / Expected
    if predicted is not None or expected is not None:
        print(f"  │")
        if predicted is not None:
            print(f"  │ Predicted: {predicted}")
        if expected is not None:
            print(f"  │ Expected:  {expected}")

    # Extra benchmark-specific lines (verbose only)
    if verbose and extra_lines:
        print(f"  │")
        for line in extra_lines:
            print(f"  │ {line}")

    # Model response (verbose only, non-orchestrator mode)
    if verbose and not raw_responses and model_response:
        print(f"  │")
        print(f"  │ Model Response:")
        for line in model_response.split('\n'):
            print(f"  │   {line}")

    print(f"  └{'─' * 72}")
    print()


def extract_final_answer(response: str) -> str:
    """Extract the final answer from model response.

    Only looks for the FINAL_ANSWER: tag that the orchestrator is trained to produce.
    If not found, returns the full response for the LM judge to evaluate.

    Args:
        response: Raw model response text

    Returns:
        Extracted answer string, or full response if no FINAL_ANSWER tag found
    """
    response = response.strip()
    if not response:
        return ""

    # Check for orchestrator FINAL_ANSWER pattern (the trained output format)
    if "FINAL_ANSWER:" in response:
        answer = response.split("FINAL_ANSWER:")[-1].strip()
        return answer

    # No FINAL_ANSWER tag found — return full response and let the LM judge handle it.
    return response
