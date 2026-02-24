"""Multi-turn executor for orchestrator inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .policy import InferencePolicy, Action


@dataclass
class ExecutorTurn:
    """Single turn in orchestrator execution."""

    turn: int
    """Turn number (0-indexed)"""

    thought: str
    """Orchestrator's reasoning"""

    tool_name: str
    """Tool/model used"""

    tool_prompt: str
    """Prompt sent to tool"""

    observation: str
    """Tool response"""

    latency_ms: float
    """Tool execution latency"""

    cost_usd: float
    """Tool cost (for cloud APIs)"""

    energy_joules: float
    """Energy consumed"""

    power_watts: float
    """Power draw"""

    tokens: int
    """Total tokens (if applicable)"""


@dataclass
class ExecutorResult:
    """Result from multi-turn orchestrator execution."""

    final_answer: str
    """Final answer produced"""

    turns: List[ExecutorTurn] = field(default_factory=list)
    """Sequence of turns taken"""

    success: bool = True
    """Whether execution completed successfully"""

    error: Optional[str] = None
    """Error message if execution failed"""

    # Aggregate metrics
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_energy_joules: float = 0.0
    total_tokens: int = 0

    ttft_ms: Optional[float] = None
    """Time to first token (first tool call)"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_answer": self.final_answer,
            "num_turns": len(self.turns),
            "turns": [
                {
                    "turn": t.turn,
                    "thought": t.thought,
                    "tool": t.tool_name,
                    "prompt": t.tool_prompt[:100],  # Truncate
                    "observation": t.observation[:100],  # Truncate
                    "latency_ms": t.latency_ms,
                    "cost_usd": t.cost_usd,
                    "energy_joules": t.energy_joules,
                    "tokens": t.tokens,
                }
                for t in self.turns
            ],
            "success": self.success,
            "error": self.error,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "total_energy_joules": self.total_energy_joules,
            "total_tokens": self.total_tokens,
            "ttft_ms": self.ttft_ms,
        }


class OrchestratorExecutor:
    """Multi-turn executor for orchestrator inference.

    Runs the orchestration loop:
    1. Policy predicts action (thought + tool + prompt)
    2. Execute tool with real telemetry
    3. Update state with observation
    4. Repeat until final answer or max turns

    Example:
        executor = OrchestratorExecutor(
            policy=policy,
            mcp_tools={"calculator": calc_server, ...},
            max_turns=10
        )

        result = executor.run("What is 15 + 27?")
        print(result.final_answer)  # "42"
        print(result.total_cost_usd)  # 0.0001
        print(len(result.turns))  # 1
    """

    def __init__(
        self,
        policy: InferencePolicy,
        mcp_tools: Dict[str, Any],
        max_turns: int = 10,
    ):
        """Initialize executor.

        Args:
            policy: Trained policy model
            mcp_tools: Dictionary of MCP tool servers
            max_turns: Maximum number of turns
        """
        self.policy = policy
        self.mcp_tools = mcp_tools
        self.max_turns = max_turns

    def run(self, question: str) -> ExecutorResult:
        """Run multi-turn orchestration.

        Args:
            question: Initial question/task

        Returns:
            ExecutorResult with final answer and trajectory
        """
        # Initialize state
        state = {
            "question": question,
            "history": [],
        }

        turns = []
        final_answer = ""
        ttft_ms = None

        try:
            for turn_num in range(self.max_turns):
                # 1. Policy predicts action
                policy_output = self.policy.predict_action(
                    state,
                    available_tools=list(self.mcp_tools.keys()),
                )

                action = policy_output.action

                # 2. Execute tool with telemetry
                start = time.time()
                tool_result = self._execute_tool(action.tool_name, action.tool_prompt)
                end = time.time()

                latency_ms = (end - start) * 1000

                # Track TTFT (first tool execution)
                if ttft_ms is None:
                    ttft_ms = latency_ms

                # 3. Create turn record
                turn = ExecutorTurn(
                    turn=turn_num,
                    thought=action.thought,
                    tool_name=action.tool_name,
                    tool_prompt=action.tool_prompt,
                    observation=tool_result.get("content", ""),
                    latency_ms=latency_ms,
                    cost_usd=tool_result.get("cost_usd", 0.0),
                    energy_joules=tool_result.get("energy_joules", 0.0),
                    power_watts=tool_result.get("power_watts", 0.0),
                    tokens=tool_result.get("tokens", 0),
                )

                turns.append(turn)

                # 4. Update state
                state["history"].append({
                    "thought": action.thought,
                    "tool": action.tool_name,
                    "prompt": action.tool_prompt,
                    "observation": tool_result.get("content", ""),
                })

                # 5. Check if final answer
                if action.is_final_answer:
                    final_answer = tool_result.get("content", "")
                    break

            # If no final answer, use last observation
            if not final_answer and turns:
                final_answer = turns[-1].observation

            # Compute aggregate metrics
            total_latency_ms = sum(t.latency_ms for t in turns)
            total_cost_usd = sum(t.cost_usd for t in turns)
            total_energy_joules = sum(t.energy_joules for t in turns)
            total_tokens = sum(t.tokens for t in turns)

            return ExecutorResult(
                final_answer=final_answer,
                turns=turns,
                success=True,
                total_latency_ms=total_latency_ms,
                total_cost_usd=total_cost_usd,
                total_energy_joules=total_energy_joules,
                total_tokens=total_tokens,
                ttft_ms=ttft_ms,
            )

        except Exception as e:
            # Execution failed
            return ExecutorResult(
                final_answer="",
                turns=turns,
                success=False,
                error=str(e),
                ttft_ms=ttft_ms,
            )

    def _execute_tool(self, tool_name: str, prompt: str) -> Dict[str, Any]:
        """Execute MCP tool and extract telemetry.

        Args:
            tool_name: Name of tool to execute
            prompt: Prompt to send to tool

        Returns:
            Dictionary with content and telemetry
        """
        # Get tool server
        if tool_name not in self.mcp_tools:
            # Tool not found - return error
            return {
                "content": f"Error: Tool '{tool_name}' not available",
                "cost_usd": 0.0,
                "energy_joules": 0.0,
                "power_watts": 0.0,
                "tokens": 0,
            }

        tool_server = self.mcp_tools[tool_name]

        # Execute tool
        try:
            result = tool_server.execute(prompt)

            # Extract telemetry from result
            # MCP servers return MCPToolResult with telemetry_samples
            telemetry_samples = getattr(result, "telemetry_samples", [])

            # Aggregate telemetry
            if telemetry_samples:
                total_energy = sum(s.energy_joules for s in telemetry_samples)
                avg_power = (
                    sum(s.power_watts for s in telemetry_samples) / len(telemetry_samples)
                    if telemetry_samples else 0.0
                )
            else:
                total_energy = 0.0
                avg_power = 0.0

            return {
                "content": result.content,
                "cost_usd": result.cost_usd or 0.0,
                "energy_joules": total_energy,
                "power_watts": avg_power,
                "tokens": result.usage.get("total_tokens", 0) if result.usage else 0,
            }

        except Exception as e:
            # Tool execution failed
            return {
                "content": f"Error executing {tool_name}: {e}",
                "cost_usd": 0.0,
                "energy_joules": 0.0,
                "power_watts": 0.0,
                "tokens": 0,
            }
