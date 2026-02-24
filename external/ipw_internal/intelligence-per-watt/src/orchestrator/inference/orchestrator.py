"""OrchestratorClient - InferenceClient implementation for trained orchestrator.

Integrates with IPW profiling infrastructure to measure energy/cost/latency
during multi-turn orchestration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Import orchestrator components
from orchestrator.inference.policy import InferencePolicy
from orchestrator.inference.executor import OrchestratorExecutor

# Import IPW components (required)
try:
    # Try installed ipw package first
    from ipw.core.registry import ClientRegistry
    from ipw.core.types import ChatUsage, Response
    from ipw.clients.base import InferenceClient
    HAS_IPW = True
except ImportError:
    try:
        # Fallback to local path (for development)
        ipw_src = Path(__file__).parent.parent.parent.parent / "ipw" / "src"
        sys.path.insert(0, str(ipw_src))
        from core.registry import ClientRegistry
        from core.types import ChatUsage, Response
        from clients.base import InferenceClient
        HAS_IPW = True
    except ImportError as e:
        raise ImportError(
            f"IPW package not found: {e}. "
            "Install with: pip install -e ./ipw"
        )


class OrchestratorClient(InferenceClient):
    """Energy-aware orchestrator client for IPW profiling."""

    client_id = "orchestrator"
    client_name = "Energy-Aware Orchestrator"

    def __init__(
        self,
        base_url: str | None = None,
        checkpoint_path: str | None = None,
        max_turns: int = 10,
        ollama_base_url: str = "http://localhost:11434",
        openai_api_key: str | None = None,
        available_tools: list[str] | None = None,
        **config: Any
    ):
        """Initialize orchestrator client.

        Args:
            base_url: Base URL (unused, for compatibility)
            checkpoint_path: Path to trained checkpoint (optional for heuristic policy)
            max_turns: Maximum turns per conversation
            ollama_base_url: Ollama API base URL
            openai_api_key: OpenAI API key (if None, uses env var)
            available_tools: List of tool names to load (if None, loads all available)
            **config: Additional config (max_tokens, temperature, top_p, etc.)
        """
        super().__init__(base_url or "orchestrator://local", **config)

        self.checkpoint_path = checkpoint_path or "checkpoints/qwen_1.7b/final"
        self.max_turns = max_turns
        self.ollama_base_url = ollama_base_url
        self.openai_api_key = openai_api_key
        self.available_tools = available_tools

        # Load policy model (if checkpoint provided)
        if checkpoint_path:
            print(f"Loading orchestrator policy from {checkpoint_path}...")
            self.policy = InferencePolicy.from_checkpoint(
                checkpoint_path,
                max_tokens=config.get("max_tokens", 512),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.9),
            )
        else:
            # TODO: Implement heuristic policy for zero-shot evaluation
            print("Warning: No checkpoint provided. Using simple heuristic policy.")
            self.policy = None

        # Initialize MCP tools
        self.mcp_tools = self._load_mcp_tools()

        # Create executor (only if policy loaded)
        if self.policy:
            self.executor = OrchestratorExecutor(
                policy=self.policy,
                mcp_tools=self.mcp_tools,
                max_turns=max_turns,
            )
        else:
            self.executor = None

        # Store last execution metadata
        self._last_result = None

        print(f"✓ Orchestrator client initialized with {len(self.mcp_tools)} tools")

    def _load_mcp_tools(self) -> dict[str, Any]:
        """Load MCP tool servers with telemetry.

        Returns:
            Dictionary mapping tool names to MCP server instances
        """
        from agents.mcp.tool_server import CalculatorServer, WebSearchServer, CodeInterpreterServer, ThinkServer
        from agents.mcp.ollama_server import OllamaMCPServer
        from agents.mcp.openai_server import OpenAIMCPServer
        from agents.mcp.anthropic_server import AnthropicMCPServer
        from agents.mcp.openrouter_server import OpenRouterMCPServer
        from agents.mcp.vllm_server import VLLMMCPServer

        # Initialize telemetry collector
        telemetry_collector = None
        try:
            from ipw.telemetry import EnergyMonitorCollector
            telemetry_collector = EnergyMonitorCollector(
                interval=1.0,  # 1 second sampling
                start_immediately=True
            )
        except ImportError:
            print("Warning: ipw.telemetry not available. Running without energy monitoring.")
        except Exception as e:
            print(f"Warning: Could not start telemetry collector: {e}")
            print("Running without energy monitoring")

        tools = {}

        # Map tool names to their setup logic
        tool_configs = {
            # Free tools (no API cost)
            "calculator": lambda: CalculatorServer(telemetry_collector),
            "think": lambda: ThinkServer(telemetry_collector),
            "web_search": lambda: WebSearchServer(telemetry_collector=telemetry_collector),
            "code_interpreter": lambda: CodeInterpreterServer(telemetry_collector=telemetry_collector),

            # Local models via Ollama
            "llama3_2_1b": lambda: OllamaMCPServer(
                model="llama3.2:1b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector
            ),
            "llama3_2_3b": lambda: OllamaMCPServer(
                model="llama3.2:3b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector
            ),
            "qwen2_5_0_5b": lambda: OllamaMCPServer(
                model="qwen2.5:0.5b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector
            ),
            "qwen2_5_1_5b": lambda: OllamaMCPServer(
                model="qwen2.5:1.5b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector
            ),

            # Large models via vLLM (requires vLLM server)
            "qwen3_8b": lambda: VLLMMCPServer(
                model_name="qwen3-8b",
                telemetry_collector=telemetry_collector
            ),
            "qwen3_32b": lambda: VLLMMCPServer(
                model_name="qwen3-32b",
                telemetry_collector=telemetry_collector
            ),
            "llama_70b": lambda: VLLMMCPServer(
                model_name="llama-70b",
                telemetry_collector=telemetry_collector
            ),

            # Specialist models via vLLM
            "qwen_math_7b": lambda: VLLMMCPServer(
                model_name="qwen-math-7b",
                telemetry_collector=telemetry_collector
            ),
            "glm_4_7": lambda: VLLMMCPServer(
                model_name="glm-4.7",
                telemetry_collector=telemetry_collector
            ),
            "qwen_coder_7b": lambda: VLLMMCPServer(
                model_name="qwen-coder-7b",
                telemetry_collector=telemetry_collector
            ),
            "qwen3_coder_plus": lambda: VLLMMCPServer(
                model_name="qwen3-coder-plus",
                telemetry_collector=telemetry_collector
            ),

            # Cloud APIs (requires API keys)
            "gpt5_mini": lambda: OpenAIMCPServer(
                api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                model="gpt-5-mini-2025-08-07",
                telemetry_collector=telemetry_collector
            ),
            "gpt4o": lambda: OpenAIMCPServer(
                api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                model="gpt-4o",
                telemetry_collector=telemetry_collector
            ),

            # === Tools with EXACT names matching training data ===

            # Anthropic tools (training data format: anthropic:model-name)
            "anthropic:claude-haiku-4-5-20251001": lambda: AnthropicMCPServer(
                model_name="claude-haiku-4-5-20251001",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
            "anthropic:claude-sonnet-4-5-20250929": lambda: AnthropicMCPServer(
                model_name="claude-sonnet-4-5-20250929",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
            "anthropic:claude-opus-4-5-20251101": lambda: AnthropicMCPServer(
                model_name="claude-opus-4-5-20251101",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                telemetry_collector=telemetry_collector
            ),

            # OpenAI tools (training data format: openai:model-name)
            "openai:gpt-5-mini-2025-08-07": lambda: OpenAIMCPServer(
                model_name="gpt-5-mini-2025-08-07",
                api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
            "openai:gpt-5.2-2025-12-11": lambda: OpenAIMCPServer(
                model_name="gpt-5.2-2025-12-11",
                api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
            "openai:gpt-5-nano-2025-08-07": lambda: OpenAIMCPServer(
                model_name="gpt-5-nano-2025-08-07",
                api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                telemetry_collector=telemetry_collector
            ),

            # Ollama tools (training data format: ollama:model-name)
            "ollama:qwen3:1.5b": lambda: OllamaMCPServer(
                model_name="qwen3:1.5b",
                base_url=self.ollama_base_url,
                telemetry_collector=telemetry_collector
            ),

            # OpenRouter tools (training data format: openrouter:provider/model)
            "openrouter:qwen/qwen3-32b": lambda: OpenRouterMCPServer(
                model_name="qwen/qwen3-32b",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
            "openrouter:z-ai/glm-4.7": lambda: OpenRouterMCPServer(
                model_name="z-ai/glm-4.7",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
            "openrouter:qwen/qwen3-coder-plus": lambda: OpenRouterMCPServer(
                model_name="qwen/qwen3-coder-plus",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                telemetry_collector=telemetry_collector
            ),
        }

        # Determine which tools to load
        tools_to_load = self.available_tools if self.available_tools else list(tool_configs.keys())

        # Load each tool
        for tool_name in tools_to_load:
            if tool_name not in tool_configs:
                print(f"Warning: Unknown tool '{tool_name}', skipping")
                continue

            try:
                tools[tool_name] = tool_configs[tool_name]()
            except Exception as e:
                print(f"Warning: Could not load tool '{tool_name}': {e}")

        if not tools:
            print("Warning: No tools loaded! Adding calculator as fallback.")
            tools["calculator"] = CalculatorServer(telemetry_collector)

        print(f"Loaded {len(tools)} MCP tools: {list(tools.keys())}")

        return tools

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Run multi-turn orchestration."""
        if self.executor is None:
            raise RuntimeError(
                "No executor available. Provide checkpoint_path to __init__ to load trained policy."
            )

        # Run orchestration
        result = self.executor.run(prompt)
        self._last_result = result

        # Aggregate tokens
        total_tokens = sum(t.tokens for t in result.turns)

        return Response(
            content=result.final_answer,
            usage=ChatUsage(
                prompt_tokens=total_tokens // 2,
                completion_tokens=total_tokens // 2,
                total_tokens=total_tokens,
            ),
            time_to_first_token_ms=result.ttft_ms or 0.0,
        )

    def list_models(self) -> Sequence[str]:
        """Return list of models."""
        return ["orchestrator-qwen-1.7b"]

    def health(self) -> bool:
        """Check health."""
        return self.policy is not None

    def get_last_trajectory(self) -> list[dict[str, Any]]:
        """Get last trajectory."""
        if self._last_result is None:
            return []
        return [
            {
                "turn": t.turn,
                "thought": t.thought,
                "tool": t.tool_name,
                "observation": t.observation[:100],
            }
            for t in self._last_result.turns
        ]

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata."""
        if self._last_result is None:
            return {}

        # Compute tool usage statistics
        tool_usage = {}
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
            "trajectory": self.get_last_trajectory(),
            "total_cost_usd": self._last_result.total_cost_usd,
            "num_turns": len(self._last_result.turns),
            "tool_usage": tool_usage,
        }

    def run(self, prompt: str) -> Response:
        """Convenience method for running orchestration.

        This is an alias for stream_chat_completion() to provide a simpler
        interface for evaluation scripts.

        Args:
            prompt: User query

        Returns:
            Response with final answer
        """
        return self.stream_chat_completion(
            model=self.list_models()[0],
            prompt=prompt
        )

# Register with IPW ClientRegistry (skip if already registered)
try:
    ClientRegistry.register_value("orchestrator", OrchestratorClient)
except ValueError:
    pass  # Already registered
