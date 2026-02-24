"""Workload generator using Poisson arrival process and truncated normal token distributions."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from inference_simulator.request.request import Request
from inference_simulator.types.execution import LLMStep, ToolCall
from inference_simulator.types.workload_profile import WorkloadProfile
from inference_simulator.types.workload_spec import WorkloadSpec, WorkloadType


class WorkloadGenerator:
    """Generates synthetic inference workloads.

    Uses a Poisson process (or gamma-distributed inter-arrivals when
    ``burstiness != 1.0``) for request arrivals and truncated normal
    distributions for input/output token lengths.
    """

    @staticmethod
    def _generate_arrivals_ns(
        rng: np.random.Generator,
        qps: float,
        duration_s: float,
        burstiness: float = 1.0,
    ) -> List[int]:
        """Generate arrival times using a gamma-distributed inter-arrival process.

        When burstiness=1.0, the gamma distribution reduces to exponential
        (Poisson process), preserving current behavior.

        Args:
            rng: NumPy random generator.
            qps: Queries per second (arrival rate).
            duration_s: Duration in seconds.
            burstiness: Shape parameter for gamma distribution.
                1.0 = Poisson, <1.0 = bursty, >1.0 = uniform.

        Returns:
            List of arrival times in nanoseconds, sorted.
        """
        mean_inter_arrival_s = 1.0 / qps
        duration_ns = int(duration_s * 1_000_000_000)

        arrivals_ns: List[int] = []
        current_ns = 0
        while current_ns < duration_ns:
            inter_arrival_s = rng.gamma(
                shape=burstiness,
                scale=mean_inter_arrival_s / burstiness,
            )
            current_ns += int(inter_arrival_s * 1_000_000_000)
            if current_ns < duration_ns:
                arrivals_ns.append(current_ns)

        return arrivals_ns

    def generate(
        self,
        workload_spec: WorkloadSpec,
        duration_s: float,
        seed: Optional[int] = None,
        max_seq_len: int = 131072,
    ) -> List[Request]:
        """Generate a list of requests over a time duration.

        Args:
            workload_spec: Workload parameters (QPS, token distributions).
            duration_s: Duration in seconds to generate requests for.
            seed: Random seed for reproducibility.
            max_seq_len: Maximum sequence length (for clamping).

        Returns:
            List of Request objects sorted by arrival time.
        """
        rng = np.random.default_rng(seed)

        if workload_spec.qps <= 0 or duration_s <= 0:
            return []

        arrivals_ns = self._generate_arrivals_ns(
            rng, workload_spec.qps, duration_s, workload_spec.burstiness,
        )

        num_requests = len(arrivals_ns)
        if num_requests == 0:
            return []

        # Generate token counts from truncated normal distributions
        input_tokens = self._sample_truncated_normal(
            rng,
            mean=workload_spec.avg_input_tokens,
            std=workload_spec.input_token_std,
            min_val=1,
            max_val=max_seq_len,
            size=num_requests,
        )

        output_tokens = self._sample_truncated_normal(
            rng,
            mean=workload_spec.avg_output_tokens,
            std=workload_spec.output_token_std,
            min_val=1,
            max_val=max_seq_len,
            size=num_requests,
        )

        requests = []
        for i in range(num_requests):
            requests.append(
                Request(
                    request_id=i,
                    arrival_time_ns=arrivals_ns[i],
                    input_tokens=int(input_tokens[i]),
                    max_output_tokens=int(output_tokens[i]),
                )
            )

        return requests

    def generate_multi_step(
        self,
        workload_spec: WorkloadSpec,
        duration_s: float,
        seed: Optional[int] = None,
        max_seq_len: int = 131072,
    ) -> List[Request]:
        """Generate multi-step requests based on workload type.

        Args:
            workload_spec: Workload parameters including workload_type.
            duration_s: Duration in seconds.
            seed: Random seed for reproducibility.
            max_seq_len: Maximum sequence length.

        Returns:
            List of Request objects with steps populated for multi-step types.
        """
        wt = workload_spec.workload_type

        if wt is None:
            return self.generate(workload_spec, duration_s, seed, max_seq_len)

        rng = np.random.default_rng(seed)

        if workload_spec.qps <= 0 or duration_s <= 0:
            return []

        arrivals_ns = self._generate_arrivals_ns(
            rng, workload_spec.qps, duration_s, workload_spec.burstiness,
        )

        num_requests = len(arrivals_ns)
        if num_requests == 0:
            return []

        requests: List[Request] = []

        for i in range(num_requests):
            input_tok = max(1, int(rng.normal(
                workload_spec.avg_input_tokens, workload_spec.input_token_std
            )))
            input_tok = min(input_tok, max_seq_len)

            output_tok = max(1, int(rng.normal(
                workload_spec.avg_output_tokens, workload_spec.output_token_std
            )))
            output_tok = min(output_tok, workload_spec.max_output_tokens)

            if wt == WorkloadType.CHAT:
                steps = self._gen_chat_steps(input_tok, output_tok)
            elif wt == WorkloadType.REASONING:
                steps = self._gen_reasoning_steps(input_tok, output_tok, rng, max_seq_len)
            elif wt == WorkloadType.AGENTIC:
                steps = self._gen_agentic_steps(
                    input_tok, output_tok, workload_spec, rng, max_seq_len,
                )
            elif wt == WorkloadType.RAG:
                steps = self._gen_rag_steps(output_tok, workload_spec, rng)
            elif wt == WorkloadType.CODING:
                steps = self._gen_coding_steps(
                    input_tok, output_tok, workload_spec, rng, max_seq_len,
                )
            else:
                steps = [LLMStep(
                    input_tokens=input_tok,
                    output_tokens=output_tok,
                    cumulative_context=0,
                )]

            # First step's tokens define the request's initial tokens
            first_step = steps[0]
            req = Request(
                request_id=i,
                arrival_time_ns=arrivals_ns[i],
                input_tokens=first_step.input_tokens,
                max_output_tokens=first_step.output_tokens,
                steps=steps,
                workload_type=wt.value,
            )
            requests.append(req)

        return requests

    @staticmethod
    def _gen_chat_steps(input_tok: int, output_tok: int) -> List[LLMStep]:
        """CHAT: single step, input/output tokens."""
        return [LLMStep(
            input_tokens=input_tok,
            output_tokens=output_tok,
            cumulative_context=0,
        )]

    @staticmethod
    def _gen_reasoning_steps(
        input_tok: int,
        output_tok: int,
        rng: np.random.Generator,
        max_seq_len: int,
    ) -> List[LLMStep]:
        """REASONING: single step with large output range."""
        # Use the caller-provided output_tok (already capped by max_output_tokens)
        # instead of hardcoded N(16384, 8192) which ignores the workload spec.
        large_output = min(output_tok, max_seq_len)
        return [LLMStep(
            input_tokens=input_tok,
            output_tokens=large_output,
            cumulative_context=0,
        )]

    @staticmethod
    def _gen_agentic_steps(
        input_tok: int,
        output_tok: int,
        spec: WorkloadSpec,
        rng: np.random.Generator,
        max_seq_len: int,
    ) -> List[LLMStep]:
        """AGENTIC: N steps with tool calls between them.

        Step 0: input_tokens / output_tokens
        Steps 1..N-1: context_extension_per_tool / output_tokens, tool_call between
        Last step: no tool_call
        """
        lo, hi = spec.num_tool_calls_range
        num_tool_calls = rng.integers(max(lo, 0), max(hi, lo) + 1)
        num_steps = num_tool_calls + 1

        steps: List[LLMStep] = []
        cumulative = 0

        for s in range(num_steps):
            if s == 0:
                step_input = input_tok
            else:
                step_input = min(spec.context_extension_per_tool, max_seq_len)

            cumulative += step_input

            tool = None
            if s < num_steps - 1:
                tool = ToolCall("web_search", "default")

            steps.append(LLMStep(
                input_tokens=step_input,
                output_tokens=output_tok,
                cumulative_context=cumulative,
                tool_call=tool,
            ))
            cumulative += output_tok

        return steps

    @staticmethod
    def _gen_rag_steps(
        output_tok: int,
        spec: WorkloadSpec,
        rng: np.random.Generator,
    ) -> List[LLMStep]:
        """RAG: 2 steps -- query then grounded generation.

        Step 0: 64-token query -> output, tool_call=faiss_retrieval
        Step 1: (num_docs * tokens_per_doc + 64) input -> output, no tool_call
        """
        lo, hi = spec.num_retrieved_docs_range
        num_docs = rng.integers(max(lo, 1), max(hi, lo) + 1)
        retrieval_input = int(num_docs) * spec.tokens_per_doc + 64

        query_tokens = 64
        steps = [
            LLMStep(
                input_tokens=query_tokens,
                output_tokens=output_tok,
                cumulative_context=0,
                tool_call=ToolCall("faiss_retrieval", "default"),
            ),
            LLMStep(
                input_tokens=retrieval_input,
                output_tokens=output_tok,
                cumulative_context=query_tokens + output_tok,
            ),
        ]
        return steps

    @staticmethod
    def _gen_coding_steps(
        input_tok: int,
        output_tok: int,
        spec: WorkloadSpec,
        rng: np.random.Generator,
        max_seq_len: int,
    ) -> List[LLMStep]:
        """CODING: similar to agentic but with code_interpreter tools."""
        lo, hi = spec.num_tool_calls_range
        num_tool_calls = rng.integers(max(lo, 0), max(hi, lo) + 1)
        num_steps = num_tool_calls + 1

        steps: List[LLMStep] = []
        cumulative = 0

        for s in range(num_steps):
            if s == 0:
                step_input = input_tok
            else:
                step_input = min(spec.context_extension_per_tool, max_seq_len)

            cumulative += step_input

            tool = None
            if s < num_steps - 1:
                tool = ToolCall("code_interpreter", "default")

            steps.append(LLMStep(
                input_tokens=step_input,
                output_tokens=output_tok,
                cumulative_context=cumulative,
                tool_call=tool,
            ))
            cumulative += output_tok

        return steps

    def generate_from_profile(
        self,
        profile: WorkloadProfile,
        qps: float,
        duration_s: float,
        seed: Optional[int] = None,
        max_seq_len: int = 131072,
    ) -> List[Request]:
        """Generate requests by sampling from fitted WorkloadProfile distributions.

        Uses the same Poisson arrival process as generate(), but draws
        token counts and step structure from the empirical distributions
        stored in the WorkloadProfile.

        Args:
            profile: Empirically-fitted workload profile.
            qps: Queries per second (arrival rate).
            duration_s: Duration in seconds to generate requests for.
            seed: Random seed for reproducibility.
            max_seq_len: Maximum sequence length (for clamping).

        Returns:
            List of Request objects sorted by arrival time.
        """
        rng = np.random.default_rng(seed)

        if qps <= 0 or duration_s <= 0:
            return []

        # 1. Arrivals (Poisson by default; burstiness=1.0)
        arrivals_ns = self._generate_arrivals_ns(rng, qps, duration_s)

        if not arrivals_ns:
            return []

        requests: List[Request] = []
        for i, arrival_ns in enumerate(arrivals_ns):
            # 2. Sample number of steps/turns
            if profile.turns_or_steps_dist is not None:
                num_steps = max(1, int(profile.turns_or_steps_dist.sample(rng, 1)[0]))
            else:
                num_steps = 1

            steps: List[LLMStep] = []
            cumulative = 0
            for s in range(num_steps):
                # 3. Sample input tokens (position-conditioned or global)
                if s in profile.input_tokens_by_position:
                    inp = max(1, int(profile.input_tokens_by_position[s].sample(rng, 1)[0]))
                elif profile.input_tokens_dist is not None:
                    inp = max(1, int(profile.input_tokens_dist.sample(rng, 1)[0]))
                else:
                    inp = 500
                inp = min(inp, max_seq_len)

                # 4. Sample output tokens (position-conditioned or global)
                if s in profile.output_tokens_by_position:
                    out = max(1, int(profile.output_tokens_by_position[s].sample(rng, 1)[0]))
                elif profile.answer_tokens_dist is not None:
                    out = max(1, int(profile.answer_tokens_dist.sample(rng, 1)[0]))
                else:
                    out = 200
                out = min(out, max_seq_len)

                # For reasoning workloads: add thinking tokens to first step
                if profile.thinking_tokens_dist is not None and s == 0:
                    thinking = max(0, int(profile.thinking_tokens_dist.sample(rng, 1)[0]))
                    out = min(out + thinking, max_seq_len)

                cumulative += inp

                # Sample tool call (not on the last step)
                tool = None
                if s < num_steps - 1 and rng.random() < profile.tool_call_probability:
                    tool_type = "web_search"
                    if profile.tool_type_distribution:
                        types = list(profile.tool_type_distribution.keys())
                        probs = list(profile.tool_type_distribution.values())
                        total = sum(probs)
                        if total > 0:
                            probs = [p / total for p in probs]
                            tool_type = rng.choice(types, p=probs)
                    tool = ToolCall(tool_type, "default")

                steps.append(LLMStep(
                    input_tokens=inp,
                    output_tokens=out,
                    cumulative_context=cumulative,
                    tool_call=tool,
                ))
                cumulative += out

            first_step = steps[0]
            req = Request(
                request_id=i,
                arrival_time_ns=arrival_ns,
                input_tokens=first_step.input_tokens,
                max_output_tokens=first_step.output_tokens,
                steps=steps,
                workload_type=profile.workload_type,
            )
            requests.append(req)

        return requests

    @staticmethod
    def profile_to_workload_spec(
        profile: WorkloadProfile, qps: float = 1.0,
    ) -> WorkloadSpec:
        """Convert a WorkloadProfile to a WorkloadSpec using distribution means.

        Useful when the caller needs a WorkloadSpec (e.g., for the simulator)
        but has a WorkloadProfile from characterization.

        Args:
            profile: Empirically-fitted workload profile.
            qps: Queries per second.

        Returns:
            WorkloadSpec with averages derived from profile distributions.
        """
        wt_map = {
            "chat": WorkloadType.CHAT,
            "reasoning": WorkloadType.REASONING,
            "agentic": WorkloadType.AGENTIC,
            "rag": WorkloadType.RAG,
            "coding": WorkloadType.CODING,
        }

        avg_input = int(profile.input_tokens_dist.mean) if profile.input_tokens_dist else 500
        avg_output = int(profile.answer_tokens_dist.mean) if profile.answer_tokens_dist else 200
        input_std = profile.input_tokens_dist.std if profile.input_tokens_dist else 200.0
        output_std = profile.answer_tokens_dist.std if profile.answer_tokens_dist else 100.0

        return WorkloadSpec(
            qps=qps,
            avg_input_tokens=avg_input,
            avg_output_tokens=avg_output,
            input_token_std=input_std,
            output_token_std=output_std,
            workload_type=wt_map.get(profile.workload_type),
        )

    @staticmethod
    def _sample_truncated_normal(
        rng: np.random.Generator,
        mean: float,
        std: float,
        min_val: int,
        max_val: int,
        size: int,
    ) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        if std <= 0:
            return np.full(size, max(min_val, min(int(mean), max_val)))

        samples = rng.normal(mean, std, size=size)
        samples = np.clip(samples, min_val, max_val)
        return np.round(samples).astype(int)
