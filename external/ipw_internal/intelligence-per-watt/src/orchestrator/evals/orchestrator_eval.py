"""Orchestrator evaluation module.

This module provides evaluation infrastructure that matches the training setup:
- Same system prompt with tool descriptions (from prompt_registry)
- Same tool execution loop
- Same multi-turn conversation format
- Same available tools
- Same generation parameters (temperature=0.7, max_tokens=8192)

This ensures the model sees identical conditions during evaluation as during training.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import MCP servers from the training codebase
import sys
from pathlib import Path
# Add src/ to path so "agents.mcp.*" imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompt_registry import AVAILABLE_TOOLS, build_system_prompt


@dataclass
class OrchestratorResult:
    """Result from orchestrator execution."""

    final_answer: Optional[str]
    """Final answer if reached, None if max turns exceeded"""

    conversation: List[Dict[str, str]]
    """Full conversation history"""

    tools_used: List[str]
    """List of tools that were called"""

    num_turns: int
    """Number of turns taken"""

    total_latency: float
    """Total time for all turns"""

    success: bool
    """Whether a final answer was produced"""

    raw_responses: List[str] = field(default_factory=list)
    """Raw model responses for debugging"""


class OrchestratorEvaluator:
    """Evaluator that runs the orchestrator with full tool execution.

    This matches the training setup EXACTLY:
    - Same system prompt with tool descriptions AND EXAMPLES
    - Same tool execution loop
    - Same response parsing (THOUGHT/TOOL/INPUT/FINAL_ANSWER)
    - Same available tools
    """

    def __init__(
        self,
        model_fn: Callable[[str, List[Dict]], str],
        available_tools: Optional[List[str]] = None,
        max_turns: int = 10,
        verbose: bool = False,
    ):
        """Initialize orchestrator evaluator.

        Args:
            model_fn: Function that takes (system_prompt, messages) and returns response
            available_tools: List of tool names to make available. If None, uses defaults.
            max_turns: Maximum turns before stopping
            verbose: Print debug output
        """
        self.model_fn = model_fn
        self.max_turns = max_turns
        self.verbose = verbose

        # Use training-matched default tools if not specified
        if available_tools is None:
            available_tools = list(AVAILABLE_TOOLS)
        self.available_tools = available_tools

        # Initialize tools
        self.tools = self._init_tools()

        # Build system prompt with tool descriptions
        self.system_prompt = self._build_system_prompt()

    def _init_tools(self) -> Dict[str, Any]:
        """Initialize tool servers."""
        tools = {}
        self.tool_errors: Dict[str, str] = {}

        for tool_name in self.available_tools:
            try:
                if tool_name == "calculator":
                    from agents.mcp.tool_server import CalculatorServer
                    tools[tool_name] = CalculatorServer()

                elif tool_name == "think":
                    from agents.mcp.tool_server import ThinkServer
                    tools[tool_name] = ThinkServer()

                elif tool_name == "code_interpreter":
                    from agents.mcp.tool_server import CodeInterpreterServer
                    tools[tool_name] = CodeInterpreterServer()

                elif tool_name == "web_search":
                    from agents.mcp.tool_server import WebSearchServer
                    tools[tool_name] = WebSearchServer()

                elif tool_name.startswith("ollama:"):
                    from agents.mcp.ollama_server import OllamaMCPServer
                    model = tool_name.split(":", 1)[1]
                    tools[tool_name] = OllamaMCPServer(model_name=model)

                elif tool_name.startswith("openai:"):
                    from agents.mcp.openai_server import OpenAIMCPServer
                    model = tool_name.split(":", 1)[1]
                    tools[tool_name] = OpenAIMCPServer(model_name=model)

                elif tool_name.startswith("anthropic:"):
                    from agents.mcp.anthropic_server import AnthropicMCPServer
                    model = tool_name.split(":", 1)[1]
                    tools[tool_name] = AnthropicMCPServer(model_name=model)

                elif tool_name.startswith("openrouter:"):
                    from agents.mcp.openrouter_server import OpenRouterMCPServer
                    model = tool_name.split(":", 1)[1]
                    tools[tool_name] = OpenRouterMCPServer(model_name=model)

                elif tool_name.startswith("vllm:"):
                    from agents.mcp.vllm_server import VLLMMCPServer
                    model = tool_name.split(":", 1)[1]
                    tools[tool_name] = VLLMMCPServer(model_name=model)

                else:
                    if self.verbose:
                        print(f"Warning: Unknown tool type '{tool_name}'")

            except ImportError as e:
                self.tool_errors[tool_name] = f"No module named '{e.name}'"
                if self.verbose:
                    print(f"Warning: Could not import tool '{tool_name}': {e}")
            except Exception as e:
                self.tool_errors[tool_name] = str(e)
                if self.verbose:
                    print(f"Warning: Could not initialize tool '{tool_name}': {e}")

        return tools

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions (from prompt_registry)."""
        return build_system_prompt(self.available_tools)

    def _parse_response(self, response: str) -> Tuple[str, str, str, bool]:
        """Parse model response into thought, tool, input, is_final.

        This EXACTLY matches the parsing in trajectory_generator.py
        """
        # Strip <think>...</think> blocks (some models output thinking)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r".*</think>", "", response, flags=re.DOTALL)
        response = response.strip()

        thought = ""
        tool_name = ""
        tool_input = ""
        is_final = False

        # Extract THOUGHT
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=\n(?:TOOL|FINAL_ANSWER)|$)",
            response, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for FINAL_ANSWER - capture everything after it (matches training parser)
        # Use .* (not .+) so empty answers like "FINAL_ANSWER:" are recognised
        final_match = re.search(
            r"FINAL_ANSWER:\s*(.*)",
            response, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            is_final = True
            tool_name = "final_answer"
            tool_input = final_match.group(1).strip()
            return thought, tool_name, tool_input, is_final

        # Extract TOOL
        tool_match = re.search(r"TOOL:\s*(\S+)", response, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1).strip()

        # Extract INPUT
        input_match = re.search(
            r"INPUT:\s*(.+?)(?=\n(?:THOUGHT|TOOL)|$)",
            response, re.DOTALL | re.IGNORECASE
        )
        if input_match:
            tool_input = input_match.group(1).strip()

        # Fallback: if no tool selected, use think
        if not tool_name and not is_final and response:
            tool_name = "think"
            tool_input = response
            thought = "Model answered directly, capturing as thinking"

        return thought, tool_name, tool_input, is_final

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the response."""
        if tool_name == "final_answer":
            return tool_input

        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not available. Available tools: {list(self.tools.keys())}"

        try:
            tool = self.tools[tool_name]
            result = tool.execute(tool_input)
            return result.content
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    def evaluate(self, task: str) -> OrchestratorResult:
        """Run orchestrator on a task with full tool execution.

        Args:
            task: The task/question to solve

        Returns:
            OrchestratorResult with final answer and conversation history
        """
        conversation = []
        tools_used = []
        raw_responses = []
        total_latency = 0.0

        # Build initial message
        messages = [{"role": "user", "content": f"Task: {task}"}]

        for turn in range(self.max_turns):
            turn_label = f" Turn {turn + 1} "

            # --- Print turn header and "thinking" indicator BEFORE model call ---
            if self.verbose:
                print(f"  ┌─{turn_label}{'─' * (70 - len(turn_label))}", flush=True)
                print(f"  │ [thinking...]", flush=True)
            else:
                print(f"  {turn_label} thinking...", end="\r", flush=True)

            start_time = time.time()

            # Get model response
            try:
                response = self.model_fn(self.system_prompt, messages)
            except Exception as e:
                if self.verbose:
                    print(f"  │ Error calling model: {e}", flush=True)
                    print(f"  └{'─' * 72}", flush=True)
                else:
                    print(f"  {turn_label} ✗ model error: {e:<55}", flush=True)
                break

            latency = time.time() - start_time
            total_latency += latency
            raw_responses.append(response)

            # Parse response
            thought, tool_name, tool_input, is_final = self._parse_response(response)

            # --- Print THOUGHT / TOOL / INPUT immediately after parsing ---
            if self.verbose:
                if thought:
                    for j, line in enumerate(thought.split('\n')):
                        prefix = "  │ THOUGHT: " if j == 0 else "  │          "
                        print(f"{prefix}{line}", flush=True)

                if is_final:
                    for j, line in enumerate(tool_input.split('\n')):
                        prefix = "  │ ANSWER:  " if j == 0 else "  │          "
                        print(f"{prefix}{line}", flush=True)
                    print(f"  └{'─' * 72}", flush=True)
                else:
                    print(f"  │ TOOL:    {tool_name}", flush=True)
                    if tool_input:
                        for j, line in enumerate(tool_input.split('\n')):
                            prefix = "  │ INPUT:   " if j == 0 else "  │          "
                            print(f"{prefix}{line}", flush=True)
            else:
                if is_final:
                    print(f"  {turn_label} FINAL_ANSWER{' ' * 55}", flush=True)
                else:
                    input_preview = tool_input.replace('\n', ' ')[:45]
                    print(f"  {turn_label} {tool_name}: {input_preview}", end="\r", flush=True)

            # Add assistant message to conversation
            if is_final:
                assistant_content = f"THOUGHT: {thought}\nFINAL_ANSWER: {tool_input}"
            else:
                assistant_content = f"THOUGHT: {thought}\nTOOL: {tool_name}\nINPUT: {tool_input}"

            messages.append({"role": "assistant", "content": assistant_content})
            conversation.append({"role": "assistant", "content": assistant_content})

            if is_final:
                return OrchestratorResult(
                    final_answer=tool_input,
                    conversation=conversation,
                    tools_used=tools_used,
                    num_turns=turn + 1,
                    total_latency=total_latency,
                    success=True,
                    raw_responses=raw_responses,
                )

            # Execute tool
            if tool_name and tool_name != "final_answer":
                # --- Print "executing" indicator BEFORE tool call ---
                if self.verbose:
                    print(f"  │ [executing {tool_name}...]", flush=True)

                tool_start = time.time()
                tools_used.append(tool_name)
                tool_response = self._execute_tool(tool_name, tool_input)
                tool_latency = time.time() - tool_start

                # --- Print STATUS / OUTPUT immediately after tool returns ---
                is_error = tool_response.startswith("Error")
                if self.verbose:
                    print(f"  │ STATUS:  {'✗ FAILED' if is_error else '✓ OK'}", flush=True)
                    for j, line in enumerate(tool_response.split('\n')):
                        prefix = "  │ OUTPUT:  " if j == 0 else "  │          "
                        print(f"{prefix}{line}", flush=True)
                    print(f"  └{'─' * 72}", flush=True)
                else:
                    status = "✗" if is_error else "✓"
                    input_preview = tool_input.replace('\n', ' ')[:45]
                    print(f"  {turn_label} {tool_name}: {input_preview:<45} {status} ({tool_latency:.1f}s)", flush=True)

                # Add tool response to conversation (matches training format)
                messages.append({"role": "user", "content": f"OBSERVATION: {tool_response}\n\nWhat is your next step?"})
                conversation.append({"role": "tool", "content": tool_response})
            else:
                # No tool selected, break
                if self.verbose:
                    print(f"  │ (no tool selected)", flush=True)
                    print(f"  └{'─' * 72}", flush=True)
                break

        # Max turns reached without final answer
        return OrchestratorResult(
            final_answer=None,
            conversation=conversation,
            tools_used=tools_used,
            num_turns=self.max_turns,
            total_latency=total_latency,
            success=False,
            raw_responses=raw_responses,
        )


def create_orchestrator_model_fn(
    model_fn: Callable[[str, List[Dict]], str],
    available_tools: Optional[List[str]] = None,
    max_turns: int = 10,
    verbose: bool = False,
) -> Callable[[str, List[Dict]], str]:
    """Create a model function wrapper that runs orchestrator evaluation.

    This wraps a base model_fn to provide full orchestrator functionality
    (tool execution, multi-turn, etc.) while presenting the same interface.

    The returned function has a `last_result` attribute containing the full
    OrchestratorResult from the most recent call (for metadata access).

    Args:
        model_fn: Base model function (system_prompt, messages) -> response
        available_tools: Tools to make available (defaults to training tools)
        max_turns: Max orchestrator turns
        verbose: Print debug info

    Returns:
        Wrapped model function that runs full orchestrator evaluation
    """
    evaluator = OrchestratorEvaluator(
        model_fn=model_fn,
        available_tools=available_tools,
        max_turns=max_turns,
        verbose=verbose,
    )

    def orchestrator_fn(_system_prompt: str, messages: List[Dict]) -> str:
        """Run orchestrator and return final answer or last response.

        _system_prompt is unused — the orchestrator builds its own from
        prompt_registry.  The parameter exists to match the model_fn interface.
        """
        # Extract task from messages
        task = messages[-1]["content"] if messages else ""

        # Run orchestrator
        result = evaluator.evaluate(task)

        # Store full result as attribute for metadata access
        orchestrator_fn.last_result = result

        if result.final_answer:
            return f"THOUGHT: Task completed.\nFINAL_ANSWER: {result.final_answer}"
        elif result.raw_responses:
            return result.raw_responses[-1]
        else:
            return "THOUGHT: Could not complete task.\nFINAL_ANSWER: Unable to determine answer."

    orchestrator_fn.last_result = None
    return orchestrator_fn
