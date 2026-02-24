"""Orchestrator agent implementation - trained policy-based orchestrator.

This agent uses a trained policy model to orchestrate multiple tools (LLMs, calculators,
code interpreters) to solve complex tasks efficiently.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agents.base import BaseAgent

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder


class Orchestrator(BaseAgent):
    """Trained orchestrator agent that routes tasks to optimal tools.

    Uses a learned policy to select the most efficient tool for each step
    of a multi-turn task, optimizing for energy/cost/latency.
    """

    DEFAULT_INSTRUCTIONS = (
        "You are an efficient orchestrator that routes tasks to the most "
        "appropriate tool based on task requirements and efficiency constraints."
    )

    def __init__(
        self,
        checkpoint_path: str | None = None,
        max_turns: int = 10,
        ollama_base_url: str = "http://localhost:11434",
        openai_api_key: str | None = None,
        available_tools: List[str] | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Orchestrator agent.

        Args:
            checkpoint_path: Path to trained policy checkpoint. If None, uses
                a simple heuristic policy for zero-shot evaluation.
            max_turns: Maximum number of turns per conversation.
            ollama_base_url: Base URL for Ollama API (local models).
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var).
            available_tools: List of tool names to load. If None, loads all available.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional configuration (max_tokens, temperature, top_p).
        """
        super().__init__(event_recorder=event_recorder)

        self.checkpoint_path = checkpoint_path
        self.max_turns = max_turns
        self.ollama_base_url = ollama_base_url
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.available_tools = available_tools
        self.config = kwargs

        # Lazy load policy and executor
        self._policy = None
        self._executor = None
        self._mcp_tools: Dict[str, Any] = {}
        self._last_result = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of policy and executor."""
        if self._policy is not None:
            return

        # Import training/inference modules (heavy dependencies)
        from orchestrator.inference.policy import InferencePolicy
        from orchestrator.inference.executor import OrchestratorExecutor

        # Load policy
        if self.checkpoint_path:
            self._policy = InferencePolicy.from_checkpoint(
                self.checkpoint_path,
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
            )
        else:
            # TODO: Implement heuristic policy for zero-shot evaluation
            self._policy = None

        # Load MCP tools
        self._mcp_tools = self._load_mcp_tools()

        # Create executor
        if self._policy:
            self._executor = OrchestratorExecutor(
                policy=self._policy,
                mcp_tools=self._mcp_tools,
                max_turns=self.max_turns,
            )

    def _load_mcp_tools(self) -> Dict[str, Any]:
        """Load MCP tool servers with telemetry and event recording.

        Returns:
            Dictionary mapping tool names to MCP server instances.
        """
        from agents.mcp import (
            CalculatorServer,
            WebSearchServer,
            CodeInterpreterServer,
            ThinkServer,
            OllamaMCPServer,
            OpenAIMCPServer,
            AnthropicMCPServer,
            OpenRouterMCPServer,
            VLLMMCPServer,
        )

        # Initialize telemetry collector
        telemetry_collector = None
        try:
            from ipw.telemetry import EnergyMonitorCollector
            telemetry_collector = EnergyMonitorCollector(
                interval=1.0,
                start_immediately=True
            )
        except ImportError:
            pass
        except Exception:
            pass

        # Get event recorder for per-action tracking
        event_recorder = self.event_recorder

        tools = {}

        # Map tool names to their setup logic
        tool_configs = {
            # Free tools (no API cost)
            "calculator": lambda: CalculatorServer(telemetry_collector),
            "think": lambda: ThinkServer(telemetry_collector),
            "web_search": lambda: WebSearchServer(telemetry_collector=telemetry_collector),
            "code_interpreter": lambda: CodeInterpreterServer(telemetry_collector=telemetry_collector),

            # Local models via Ollama (with event recording)
            "llama3_2_1b": lambda: OllamaMCPServer(
                model_name="llama3.2:1b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector,
                event_recorder=event_recorder,
            ),
            "llama3_2_3b": lambda: OllamaMCPServer(
                model_name="llama3.2:3b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector,
                event_recorder=event_recorder,
            ),

            # Cloud APIs (requires API keys, with event recording)
            "gpt4o_mini": lambda: OpenAIMCPServer(
                api_key=self.openai_api_key,
                model="gpt-4o-mini",
                telemetry_collector=telemetry_collector,
                event_recorder=event_recorder,
            ),
            "gpt4o": lambda: OpenAIMCPServer(
                api_key=self.openai_api_key,
                model="gpt-4o",
                telemetry_collector=telemetry_collector,
                event_recorder=event_recorder,
            ),
        }

        # Add vLLM tools dynamically based on available_tools
        if self.available_tools:
            for tool_name in self.available_tools:
                if tool_name.startswith("vllm:"):
                    model_alias = tool_name.split(":", 1)[1]
                    tool_configs[tool_name] = lambda m=model_alias: VLLMMCPServer(
                        model_name=m,
                        telemetry_collector=telemetry_collector,
                        event_recorder=event_recorder,
                    )
                elif tool_name.startswith("ollama:"):
                    model_name = tool_name.split(":", 1)[1]
                    tool_configs[tool_name] = lambda m=model_name: OllamaMCPServer(
                        model_name=m,
                        base_url=self.ollama_base_url,
                        telemetry_collector=telemetry_collector,
                        event_recorder=event_recorder,
                    )

        # Determine which tools to load
        tools_to_load = self.available_tools if self.available_tools else list(tool_configs.keys())

        # Load each tool
        for tool_name in tools_to_load:
            if tool_name not in tool_configs:
                continue
            try:
                tools[tool_name] = tool_configs[tool_name]()
            except Exception:
                pass

        if not tools:
            tools["calculator"] = CalculatorServer(telemetry_collector)

        return tools

    def get_available_tools(self) -> List[str]:
        """Get the list of available tool names.

        This is useful for auto-detecting submodels in the CLI.

        Returns:
            List of tool names (e.g., ['vllm:qwen3-32b', 'ollama:llama3.2:1b'])
        """
        return self.available_tools or []

    def run(self, input: str, **kwargs: Any) -> Any:
        """Run the Orchestrator agent.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments.

        Returns:
            The final answer from the orchestrator.
        """
        self._ensure_initialized()

        if self._executor is None:
            raise RuntimeError(
                "No executor available. Provide checkpoint_path to load trained policy, "
                "or ensure the policy loaded successfully."
            )

        self._record_event("lm_inference_start", agent="orchestrator")
        try:
            result = self._executor.run(input)
            self._last_result = result
            return result.final_answer
        finally:
            self._record_event("lm_inference_end", agent="orchestrator")

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the trajectory from the last run.

        Returns:
            List of turn dictionaries with tool, thought, and observation.
        """
        if self._last_result is None:
            return []
        return [
            {
                "turn": t.turn,
                "thought": t.thought,
                "tool": t.tool_name,
                "observation": t.observation[:100] if t.observation else "",
            }
            for t in self._last_result.turns
        ]

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last run.

        Returns:
            Dictionary with trajectory, cost, and tool usage statistics.
        """
        if self._last_result is None:
            return {}

        # Compute tool usage statistics
        tool_usage: Dict[str, Dict[str, Any]] = {}
        for turn in self._last_result.turns:
            tool_name = turn.tool_name
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {
                    "count": 0,
                    "total_cost_usd": 0.0,
                    "total_energy_joules": 0.0,
                    "total_latency_ms": 0.0,
                }
            tool_usage[tool_name]["count"] += 1
            tool_usage[tool_name]["total_cost_usd"] += turn.cost_usd
            tool_usage[tool_name]["total_energy_joules"] += turn.energy_joules
            tool_usage[tool_name]["total_latency_ms"] += turn.latency_ms

        return {
            "trajectory": self.get_trajectory(),
            "total_cost_usd": self._last_result.total_cost_usd,
            "num_turns": len(self._last_result.turns),
            "tool_usage": tool_usage,
        }
