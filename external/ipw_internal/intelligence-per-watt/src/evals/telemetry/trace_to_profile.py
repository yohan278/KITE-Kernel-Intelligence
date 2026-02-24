"""Convert QueryTrace lists to WorkloadProfile."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from evals.telemetry.trace_collector import QueryTrace
from inference_simulator.types.fitted_distribution import FittedDistribution
from inference_simulator.types.workload_profile import WorkloadProfile


class TraceToProfile:
    """Converts a list of QueryTraces into a WorkloadProfile."""

    def convert(
        self,
        traces: List[QueryTrace],
        workload_type: str = "",
        source_dataset: str = "active_inference",
    ) -> WorkloadProfile:
        """Convert traces to a WorkloadProfile.

        Mapping:
        - num_turns across queries -> turns_or_steps_dist
        - turn.input_tokens -> input_tokens_dist
        - turn.output_tokens -> answer_tokens_dist
        - tokens grouped by turn_index -> input_tokens_by_position, output_tokens_by_position
        - fraction of turns with tools -> tool_call_probability
        - turn.tool_result_tokens -> tool_call_tokens_dist
        - normalized tool name counts -> tool_type_distribution
        """
        if not traces:
            return WorkloadProfile(
                workload_type=workload_type or "unknown",
                source_dataset=source_dataset,
                n_samples=0,
            )

        wt = workload_type or (traces[0].workload_type if traces else "unknown")

        # Collect data
        turn_counts: List[float] = []
        all_input_tokens: List[float] = []
        all_output_tokens: List[float] = []
        tool_result_tokens_list: List[float] = []
        tool_counts: Counter[str] = Counter()
        total_turns_with_tools = 0
        total_turns = 0

        # Per-position token lists
        input_by_pos: Dict[int, List[float]] = defaultdict(list)
        output_by_pos: Dict[int, List[float]] = defaultdict(list)

        # Inter-turn timing
        inter_turn_times: List[float] = []

        max_context = 0

        for trace in traces:
            if not trace.turns:
                continue

            turn_counts.append(float(trace.num_turns))

            cumulative_tokens = 0
            prev_end_time: Optional[float] = None

            for turn in trace.turns:
                all_input_tokens.append(float(turn.input_tokens))
                all_output_tokens.append(float(turn.output_tokens))

                input_by_pos[turn.turn_index].append(float(turn.input_tokens))
                output_by_pos[turn.turn_index].append(float(turn.output_tokens))

                total_turns += 1
                if turn.tools_called:
                    total_turns_with_tools += 1
                    for tool in turn.tools_called:
                        tool_counts[tool] += 1

                if turn.tool_result_tokens > 0:
                    tool_result_tokens_list.append(float(turn.tool_result_tokens))

                cumulative_tokens += turn.input_tokens + turn.output_tokens
                max_context = max(max_context, cumulative_tokens)

                # Inter-turn timing
                if prev_end_time is not None and turn.wall_clock_s > 0:
                    inter_turn_times.append(turn.wall_clock_s)
                prev_end_time = turn.wall_clock_s

        # Fit distributions
        turns_dist = FittedDistribution.fit(turn_counts) if turn_counts else None
        input_dist = FittedDistribution.fit(all_input_tokens) if all_input_tokens else None
        output_dist = FittedDistribution.fit(all_output_tokens) if all_output_tokens else None

        # Per-position distributions
        input_by_pos_fitted: Dict[int, FittedDistribution] = {}
        for pos, values in sorted(input_by_pos.items()):
            if len(values) >= 2:
                input_by_pos_fitted[pos] = FittedDistribution.fit(values)

        output_by_pos_fitted: Dict[int, FittedDistribution] = {}
        for pos, values in sorted(output_by_pos.items()):
            if len(values) >= 2:
                output_by_pos_fitted[pos] = FittedDistribution.fit(values)

        # Tool stats
        tool_call_prob = total_turns_with_tools / total_turns if total_turns > 0 else 0.0
        tool_dist = FittedDistribution.fit(tool_result_tokens_list) if tool_result_tokens_list else None

        # Normalize tool type distribution
        total_tool_calls = sum(tool_counts.values())
        tool_type_dist = {
            name: count / total_tool_calls
            for name, count in tool_counts.items()
        } if total_tool_calls > 0 else {}

        # Inter-turn timing
        inter_turn_dist = FittedDistribution.fit(inter_turn_times) if inter_turn_times else None

        return WorkloadProfile(
            workload_type=wt,
            source_dataset=source_dataset,
            n_samples=len(traces),
            turns_or_steps_dist=turns_dist,
            input_tokens_dist=input_dist,
            answer_tokens_dist=output_dist,
            input_tokens_by_position=input_by_pos_fitted,
            output_tokens_by_position=output_by_pos_fitted,
            tool_call_probability=tool_call_prob,
            tool_call_tokens_dist=tool_dist,
            tool_type_distribution=tool_type_dist,
            inter_turn_seconds_dist=inter_turn_dist,
            max_context_observed=max_context,
        )
