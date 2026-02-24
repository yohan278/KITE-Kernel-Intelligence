"""Trace collection for eval benchmark runs."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TurnTrace:
    """Per-turn telemetry data."""
    turn_index: int
    input_tokens: int = 0
    output_tokens: int = 0
    tool_result_tokens: int = 0
    tools_called: List[str] = field(default_factory=list)
    tool_latencies_s: Dict[str, float] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "tools_called": list(self.tools_called),
            "tool_latencies_s": dict(self.tool_latencies_s),
            "wall_clock_s": self.wall_clock_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TurnTrace:
        return cls(
            turn_index=d["turn_index"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            tool_result_tokens=d.get("tool_result_tokens", 0),
            tools_called=d.get("tools_called", []),
            tool_latencies_s=d.get("tool_latencies_s", {}),
            wall_clock_s=d.get("wall_clock_s", 0.0),
            error=d.get("error"),
        )


@dataclass
class QueryTrace:
    """Per-query aggregate telemetry."""
    query_id: str
    workload_type: str
    query_text: str = ""
    response_text: str = ""
    turns: List[TurnTrace] = field(default_factory=list)
    total_wall_clock_s: float = 0.0
    completed: bool = False

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def tool_call_count(self) -> int:
        return sum(len(t.tools_called) for t in self.turns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "workload_type": self.workload_type,
            "query_text": self.query_text,
            "response_text": self.response_text,
            "turns": [t.to_dict() for t in self.turns],
            "total_wall_clock_s": self.total_wall_clock_s,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QueryTrace:
        return cls(
            query_id=d["query_id"],
            workload_type=d["workload_type"],
            query_text=d.get("query_text", ""),
            response_text=d.get("response_text", ""),
            turns=[TurnTrace.from_dict(t) for t in d.get("turns", [])],
            total_wall_clock_s=d.get("total_wall_clock_s", 0.0),
            completed=d.get("completed", False),
        )

    def save_jsonl(self, path: Path) -> None:
        """Append this trace as a JSONL line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> List[QueryTrace]:
        """Load traces from a JSONL file."""
        traces = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(cls.from_dict(json.loads(line)))
        return traces


class TraceCollector:
    """Wraps agent/model execution to collect telemetry traces."""

    def __init__(self, vllm_url: str = "http://localhost:8000", model_name: str = ""):
        self.vllm_url = vllm_url.rstrip("/")
        self.model_name = model_name

    def run_query_direct_vllm(
        self,
        query_id: str,
        workload_type: str,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> QueryTrace:
        """Run a single query against vLLM's OpenAI-compatible API.

        For no-tool benchmarks (WildChat, OpenThoughts). Sends messages to
        vLLM /v1/chat/completions, extracts token counts from usage response.

        Args:
            query_id: Unique identifier for this query.
            workload_type: Workload category (chat, reasoning, etc.).
            messages: List of {"role": ..., "content": ...} messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            QueryTrace with a single TurnTrace containing token usage.
        """
        query_text = messages[-1]["content"] if messages else ""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            with httpx.Client(timeout=240.0) as client:
                response = client.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            wall_clock = time.time() - start_time
            usage = data.get("usage", {})
            content = data["choices"][0]["message"]["content"]

            turn = TurnTrace(
                turn_index=0,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                wall_clock_s=wall_clock,
            )

            return QueryTrace(
                query_id=query_id,
                workload_type=workload_type,
                query_text=query_text,
                response_text=content,
                turns=[turn],
                total_wall_clock_s=wall_clock,
                completed=True,
            )

        except Exception as e:
            wall_clock = time.time() - start_time
            logger.error(f"vLLM query failed for {query_id}: {e}")
            turn = TurnTrace(
                turn_index=0,
                wall_clock_s=wall_clock,
                error=str(e),
            )
            return QueryTrace(
                query_id=query_id,
                workload_type=workload_type,
                query_text=query_text,
                turns=[turn],
                total_wall_clock_s=wall_clock,
                completed=False,
            )

    def run_query_multi_turn_vllm(
        self,
        query_id: str,
        workload_type: str,
        conversation: List[Dict[str, str]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> QueryTrace:
        """Run a multi-turn conversation against vLLM.

        Sends user turns one at a time, accumulating assistant responses.
        Used for WildChat multi-turn replay.

        Args:
            query_id: Unique identifier.
            workload_type: Workload category.
            conversation: Full conversation as alternating user/assistant messages.
            max_tokens: Max tokens per turn.
            temperature: Sampling temperature.

        Returns:
            QueryTrace with one TurnTrace per user turn.
        """
        query_text = conversation[0]["content"] if conversation else ""
        start_time = time.time()
        turns: List[TurnTrace] = []
        messages_so_far: List[Dict[str, str]] = []
        response_text = ""
        completed = True
        turn_idx = 0

        # Extract user turns from conversation
        for msg in conversation:
            if msg["role"] == "user":
                messages_so_far.append(msg)
                turn_start = time.time()

                try:
                    with httpx.Client(timeout=240.0) as client:
                        response = client.post(
                            f"{self.vllm_url}/v1/chat/completions",
                            json={
                                "model": self.model_name,
                                "messages": messages_so_far,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            },
                        )
                        response.raise_for_status()
                        data = response.json()

                    usage = data.get("usage", {})
                    content = data["choices"][0]["message"]["content"]
                    messages_so_far.append({"role": "assistant", "content": content})
                    response_text = content

                    turns.append(TurnTrace(
                        turn_index=turn_idx,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        wall_clock_s=time.time() - turn_start,
                    ))
                except Exception as e:
                    turns.append(TurnTrace(
                        turn_index=turn_idx,
                        wall_clock_s=time.time() - turn_start,
                        error=str(e),
                    ))
                    completed = False
                    break

                turn_idx += 1
            elif msg["role"] == "assistant":
                # Skip original assistant messages in replay
                continue
            else:
                messages_so_far.append(msg)

        return QueryTrace(
            query_id=query_id,
            workload_type=workload_type,
            query_text=query_text,
            response_text=response_text,
            turns=turns,
            total_wall_clock_s=time.time() - start_time,
            completed=completed,
        )

    def run_query_react(
        self,
        query_id: str,
        workload_type: str,
        query: str,
        tools: List[Dict[str, Any]],
        *,
        max_turns: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> QueryTrace:
        """Run a React-style agent loop against vLLM with tools.

        Used for HotpotQA with retrieval tool. The agent sends messages
        to vLLM, checks for tool calls, executes tools, and continues.

        Args:
            query_id: Unique identifier.
            workload_type: Workload category.
            query: The query text.
            tools: Tool definitions (not used in direct vLLM calls, but captured).
            max_turns: Maximum agent turns.
            max_tokens: Max tokens per turn.
            temperature: Sampling temperature.

        Returns:
            QueryTrace with TurnTraces for each agent turn.
        """
        # Stub implementation - actual React loop would be more complex
        # For now, just do a single vLLM call and record it
        return self.run_query_direct_vllm(
            query_id=query_id,
            workload_type=workload_type,
            messages=[{"role": "user", "content": query}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def run_query_openhands(
        self,
        query_id: str,
        workload_type: str,
        task_description: str,
        **kwargs: Any,
    ) -> QueryTrace:
        """Run a query using OpenHands-style agent.

        Used for AgentData and SWE-bench benchmarks.
        This is a stub - actual implementation would use OpenHands SDK.

        Args:
            query_id: Unique identifier.
            workload_type: Workload category.
            task_description: Task description for the agent.

        Returns:
            QueryTrace capturing the agent execution.
        """
        start_time = time.time()
        # Stub: return empty trace for now
        return QueryTrace(
            query_id=query_id,
            workload_type=workload_type,
            query_text=task_description,
            total_wall_clock_s=time.time() - start_time,
            completed=False,
        )
