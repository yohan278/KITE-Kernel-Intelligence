"""Text-based tool calling agent for grid evaluation.

Uses the orchestrator's proven approach: plain text generation + THOUGHT/TOOL/INPUT
system prompt + regex parsing + MCP tool routing. This avoids dependence on vLLM's
native tool calling (--enable-auto-tool-choice --tool-call-parser) which gives 0%
accuracy on GAIA due to model-specific parser mismatches.

Adapted from src/orchestrator/data/trajectory_generator.py.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a TextToolAgent run, including tool calling metrics."""

    content: str
    """Final answer text."""

    tool_calls_attempted: int = 0
    """Number of TOOL: lines parsed from model output."""

    tool_calls_succeeded: int = 0
    """Number of successful MCP tool executions."""

    tool_names_used: List[str] = field(default_factory=list)
    """Which tools were called (may contain duplicates)."""

    num_turns: int = 0
    """Number of THOUGHT/TOOL/INPUT turns taken."""

    steps: List[Dict[str, Any]] = field(default_factory=list)
    """Per-turn step details for debugging."""


# ---------------------------------------------------------------------------
# System prompt (adapted from orchestrator's TEACHER_SYSTEM_PROMPT_BASE)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are an intelligent assistant that solves tasks by using the most appropriate tools.

=== AVAILABLE TOOLS ===
{tools_description}

=== TOOL SELECTION GUIDE ===
{tool_selection_guide}

=== RESPONSE FORMAT ===
You MUST respond in this EXACT format:

THOUGHT: <analyze the task and explain which tool is best and why>
TOOL: <exact tool name from the list>
INPUT: <input for the tool>

After getting tool results, either use another tool or give final answer:
THOUGHT: <analyze the result>
FINAL_ANSWER: <your final answer>

=== RULES ===
1. Match the tool to the task type (see guide above)
2. For LLM tools, write clear prompts that will get good responses
3. Prefer specialized tools when available
4. For simple factual questions, use fast/cheap models when available
5. Always end with FINAL_ANSWER when you have enough information
"""

# Tool descriptions (subset from orchestrator, focused on grid eval tools)
TOOL_DESCRIPTIONS = {
    "calculator": """CALCULATOR - Instant math computation
  - BEST FOR: Arithmetic, algebra, trigonometry, scientific calculations
  - STRENGTHS: Instant (<1ms), perfect accuracy, zero cost
  - Input: math expression (e.g., '15 * 7 + 23', 'sqrt(144)')""",

    "think": """THINK - Internal reasoning scratchpad
  - BEST FOR: Logic puzzles, step-by-step reasoning, planning approach
  - STRENGTHS: Organizes thoughts, shows work, no external calls
  - Input: your detailed reasoning process""",

    "code_interpreter": """CODE_INTERPRETER - Python execution sandbox
  - BEST FOR: Data processing, algorithms, simulations, file operations
  - STRENGTHS: Full Python + numpy/pandas, handles loops/conditionals
  - Input: Python code to execute""",

    "web_search": """WEB_SEARCH - Real-time internet search
  - BEST FOR: Current events, recent news, fact-checking, looking up information
  - STRENGTHS: Access to up-to-date information beyond training data
  - Input: search query string""",

    "file_read": """FILE_READ - Read file contents
  - BEST FOR: Reading data files, configuration, logs
  - Input: file path""",

    "file_write": """FILE_WRITE - Write or append to files
  - BEST FOR: Saving results, writing reports
  - Input: file path and content""",
}

# Cloud LLM tool descriptions (added dynamically if API keys present)
CLOUD_LLM_DESCRIPTIONS = {
    "openai:gpt-4o": """GPT-4O - Capable OpenAI model
  - BEST FOR: Complex reasoning, analysis, creative tasks
  - Input: detailed prompt""",

    "openai:gpt-5-mini-2025-08-07": """GPT-5-MINI - Fast, capable OpenAI model
  - BEST FOR: Simple Q&A, classification, quick responses
  - Input: prompt for the model""",

    "anthropic:claude-sonnet-4-20250514": """CLAUDE SONNET - Balanced Anthropic model
  - BEST FOR: Analysis, writing, reasoning, code review
  - Input: detailed prompt with context""",

    "anthropic:claude-3-5-haiku-20241022": """CLAUDE HAIKU - Fast Anthropic model
  - BEST FOR: Quick analysis, simple reasoning, classification
  - Input: prompt for the model""",

    "openrouter:google/gemini-2.5-flash": """GEMINI 2.5 FLASH - Fast Google model
  - BEST FOR: General tasks, quick responses
  - Input: prompt for the model""",

    "openrouter:google/gemini-2.5-pro": """GEMINI 2.5 PRO - Capable Google model
  - BEST FOR: Complex analysis, long context tasks
  - Input: detailed prompt""",
}


def _build_selection_guide(available_tools: List[str]) -> str:
    """Build tool selection guide based on available tools."""
    lines = ["Choose tools based on task type:\n"]

    # MATH section
    math_tools = []
    if "calculator" in available_tools:
        math_tools.append("- Simple arithmetic/algebra -> calculator (instant, accurate)")
    if "code_interpreter" in available_tools:
        math_tools.append("- Numerical algorithms -> code_interpreter (programmable)")
    if math_tools:
        lines.append("MATH PROBLEMS:")
        lines.extend(math_tools)
        lines.append("")

    # CODING section
    code_tools = []
    if "code_interpreter" in available_tools:
        code_tools.append("- Write/run code -> code_interpreter")
    if code_tools:
        lines.append("CODING TASKS:")
        lines.extend(code_tools)
        lines.append("")

    # REASONING section
    reasoning_tools = []
    if "think" in available_tools:
        reasoning_tools.append("- Step-by-step analysis -> think (organize thoughts first)")
    # Cloud LLMs for complex reasoning
    cloud = [t for t in available_tools if ":" in t]
    if cloud:
        reasoning_tools.append(f"- Complex reasoning -> {cloud[0]}")
    if reasoning_tools:
        lines.append("REASONING/LOGIC:")
        lines.extend(reasoning_tools)
        lines.append("")

    # FACTUAL section
    factual_tools = []
    if "web_search" in available_tools:
        factual_tools.append("- Current events, recent facts -> web_search")
    if factual_tools:
        lines.append("FACTUAL/KNOWLEDGE:")
        lines.extend(factual_tools)
        lines.append("")

    return "\n".join(lines)


class TextToolAgent:
    """Agent using text-based THOUGHT/TOOL/INPUT format with MCP tool routing.

    Instead of relying on vLLM's native tool calling (which requires model-specific
    parsers and gives 0% accuracy on many models), this agent:
    1. Sends a system prompt instructing the model to output THOUGHT/TOOL/INPUT
    2. Parses the plain text response with regex
    3. Routes tool calls to MCP servers
    4. Appends observations and loops until FINAL_ANSWER

    This approach achieves 85% tool call success rate in the orchestrator.
    """

    def __init__(
        self,
        model_id: str,
        vllm_base_url: str,
        mcp_tools: Dict[str, Any],
        event_recorder: Optional[Any] = None,
        max_turns: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_context_tokens: Optional[int] = None,
    ) -> None:
        """Initialize the text tool agent.

        Args:
            model_id: HuggingFace model ID served by vLLM.
            vllm_base_url: vLLM OpenAI-compatible API URL (e.g. http://localhost:8000/v1).
            mcp_tools: Dict mapping tool name to initialized MCP server instance.
            event_recorder: Optional telemetry event recorder.
            max_turns: Maximum number of tool-use turns before forcing final answer.
            temperature: Sampling temperature for model calls.
            max_tokens: Maximum tokens per model response.
            max_context_tokens: Optional max tokens for context window management.
                If set, messages are truncated to fit within this budget.
        """
        self.model_id = model_id
        self.vllm_base_url = vllm_base_url
        self.mcp_tools = mcp_tools
        self.event_recorder = event_recorder
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens

        # Build system prompt with available tool descriptions
        self.system_prompt = self._build_system_prompt()

        # Initialize OpenAI client pointing at vLLM
        import openai

        # Strip /v1 suffix if present — openai client adds it
        base_url = vllm_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key="dummy",  # vLLM doesn't require a real key
        )

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Generate system prompt with available tool descriptions."""
        tools_desc = self._build_tools_description()
        selection_guide = _build_selection_guide(list(self.mcp_tools.keys()))
        return SYSTEM_PROMPT_TEMPLATE.format(
            tools_description=tools_desc,
            tool_selection_guide=selection_guide,
        )

    def _build_tools_description(self) -> str:
        """Build tool descriptions string from available MCP tools."""
        lines = []
        for tool_name in self.mcp_tools:
            # Check our description dicts
            if tool_name in TOOL_DESCRIPTIONS:
                desc = TOOL_DESCRIPTIONS[tool_name]
            elif tool_name in CLOUD_LLM_DESCRIPTIONS:
                desc = CLOUD_LLM_DESCRIPTIONS[tool_name]
            else:
                # Fallback: use MCP server spec if available
                server = self.mcp_tools[tool_name]
                spec = getattr(server, "_spec", None)
                if spec and hasattr(spec, "description"):
                    desc = spec.description
                else:
                    desc = f"Tool: {tool_name}"
            lines.append(f"- {tool_name}: {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------

    def _truncate_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Truncate middle messages to fit within context token budget.

        Preserves:
        - messages[0]: system prompt (tool instructions)
        - messages[1]: original user query
        - As many recent messages as fit in remaining budget (newest first)

        Inserts a separator where content was removed.
        """
        if not self.max_context_tokens or len(messages) <= 4:
            return messages

        # Estimate total tokens (1 token ≈ 4 chars)
        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self.max_context_tokens:
            return messages

        # Always keep first 2 messages (system prompt + user query)
        header = messages[:2]
        header_chars = sum(len(m["content"]) for m in header)
        separator_chars = 100
        remaining_budget = (self.max_context_tokens * 4) - header_chars - separator_chars

        # Fill from end — most recent messages are most valuable
        tail: List[Dict[str, str]] = []
        tail_chars = 0
        for msg in reversed(messages[2:]):
            msg_chars = len(msg["content"])
            if tail_chars + msg_chars > remaining_budget:
                break
            tail.insert(0, msg)
            tail_chars += msg_chars

        removed_count = len(messages) - 2 - len(tail)
        if removed_count > 0:
            separator = {
                "role": "user",
                "content": f"[{removed_count} earlier conversation turns omitted due to context length limit.]",
            }
            logger.info(
                f"Context truncation: removed {removed_count} middle messages, kept {len(tail)} recent"
            )
            return header + [separator] + tail

        return messages

    # ------------------------------------------------------------------
    # Response parsing (from orchestrator's _parse_teacher_response)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> Tuple[str, str, str, bool]:
        """Parse model response into (thought, tool_name, tool_input, is_final).

        Handles:
        - <think>...</think> blocks (GLM, Qwen3 thinking mode)
        - THOUGHT:/TOOL:/INPUT: format
        - FINAL_ANSWER: format
        - Fallback to "think" tool for unformatted responses
        """
        # Extract content from <think>...</think> blocks before stripping.
        # Some models (Qwen3) put TOOL/INPUT directives inside think blocks.
        think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
        think_content = think_match.group(1) if think_match else ""
        # Strip <think>...</think> blocks from main response
        stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Handle unclosed think tags
        stripped = re.sub(r".*</think>", "", stripped, flags=re.DOTALL)
        # Strip [Thinking]...[/Thinking] blocks (GLM alternate format)
        stripped = re.sub(r"\[Thinking\].*?\[/Thinking\]", "", stripped, flags=re.DOTALL)
        # Handle bare [Thinking] tag (no closing tag) — just remove the marker
        stripped = re.sub(r"\[Thinking\]\s*\n?", "", stripped)
        stripped = stripped.strip()
        # If stripped content has TOOL/FINAL_ANSWER, use it; otherwise
        # check inside think block for those directives.
        if re.search(r"(?:TOOL|FINAL_ANSWER):", stripped, re.IGNORECASE):
            response = stripped
        elif think_content and re.search(r"(?:TOOL|FINAL_ANSWER):", think_content, re.IGNORECASE):
            response = think_content.strip()
        else:
            response = stripped

        thought = ""
        tool_name = ""
        tool_input = ""
        is_final = False

        # Extract THOUGHT
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=\n(?:TOOL|FINAL_ANSWER)|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract TOOL first — if both TOOL and FINAL_ANSWER are present,
        # the model is hallucinating tool results. Execute the tool call.
        tool_match = re.search(r"TOOL:\s*(\S+)", response, re.IGNORECASE)
        has_tool = tool_match is not None

        # Check for FINAL_ANSWER
        final_match = re.search(
            r"FINAL_ANSWER:\s*(.+)", response, re.DOTALL | re.IGNORECASE
        )
        if final_match and not has_tool:
            # Only treat as final if there's no tool call to execute first
            is_final = True
            tool_name = "final_answer"
            tool_input = final_match.group(1).strip()
            return thought, tool_name, tool_input, is_final

        # Extract TOOL
        if tool_match:
            tool_name = tool_match.group(1).strip()

        # Extract INPUT (stop at next THOUGHT, TOOL, or FINAL_ANSWER)
        input_match = re.search(
            r"INPUT:\s*(.+?)(?=\n(?:THOUGHT|TOOL|FINAL_ANSWER)|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            tool_input = input_match.group(1).strip()

        # Fallback: treat unformatted response as "think" tool
        if not tool_name and not is_final and response:
            tool_name = "think"
            tool_input = response
            thought = "Model answered directly, using think tool to capture reasoning"

        return thought, tool_name, tool_input, is_final

    # ------------------------------------------------------------------
    # Tool execution (from orchestrator's _execute_tool)
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_input: str) -> Tuple[str, bool]:
        """Execute a tool via MCP server.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input string for the tool.

        Returns:
            (response_text, success)
        """
        if tool_name == "final_answer":
            return tool_input, True

        if tool_name not in self.mcp_tools:
            return f"Error: Tool '{tool_name}' not available. Available tools: {', '.join(self.mcp_tools.keys())}", False

        try:
            tool = self.mcp_tools[tool_name]
            result = tool.execute(tool_input)
            content = result.content if hasattr(result, "content") else str(result)
            return content, True
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' execution failed: {e}")
            return f"Error executing tool '{tool_name}': {e}", False

    # ------------------------------------------------------------------
    # Model calling (plain text, no native tool_calls)
    # ------------------------------------------------------------------

    def _call_model(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        """Call vLLM via OpenAI-compatible API (plain text, no tool_calls).

        Args:
            messages: Chat messages in OpenAI format.

        Returns:
            Tuple of (response_text, usage_dict) where usage_dict contains
            prompt_tokens, completion_tokens, total_tokens.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            usage: Dict[str, int] = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
            message = response.choices[0].message
            response_text = message.content or ""
            # GPT-OSS models use vLLM's harmony channel separation:
            # reasoning goes to message.reasoning, final answer to message.content.
            # If content is empty, fall back to reasoning.
            if not response_text:
                reasoning = getattr(message, "reasoning", None) or getattr(
                    message, "reasoning_content", None
                )
                if reasoning:
                    response_text = reasoning
            return response_text, usage
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return f"Error: {e}", {}

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        max_error_retries: int = 3,
        **kwargs: Any,
    ) -> RunResult:
        """Run the agent on a query with multi-turn tool use.

        Args:
            query: The task/question to solve.
            max_error_retries: Maximum consecutive model errors before giving up.

        Returns:
            RunResult with final answer and tool calling metrics.
        """
        if self.event_recorder:
            self.event_recorder.record("agent_start", query=query[:200])

        # Build user message with tool reminder to reinforce tool-use format.
        # Some models (Qwen3, GLM) weight user messages more heavily than system prompts.
        tool_names = ", ".join(self.mcp_tools.keys())
        user_content = (
            f"{query}\n\n"
            f"Remember: You have these tools available: [{tool_names}]. "
            f"Use THOUGHT/TOOL/INPUT format to call them. "
            f"Do NOT answer directly — use tools to gather information first, "
            f"then provide FINAL_ANSWER."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        tool_calls_attempted = 0
        tool_calls_succeeded = 0
        tool_names_used: List[str] = []
        steps: List[Dict[str, Any]] = []
        final_answer = ""
        error_retries = 0

        for turn in range(self.max_turns):
            # Call model with paired start/end events
            if self.event_recorder:
                self.event_recorder.record(
                    "lm_inference_start", turn=turn, model_id=self.model_id
                )

            # Truncate context if approaching limit
            truncated_messages = self._truncate_context(messages)
            raw_response, usage = self._call_model(truncated_messages)

            if self.event_recorder:
                self.event_recorder.record(
                    "lm_inference_end",
                    turn=turn,
                    model_id=self.model_id,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )

            if not raw_response or raw_response.startswith("Error:"):
                error_retries += 1
                logger.warning(
                    f"Model returned error/empty at turn {turn} "
                    f"(retry {error_retries}/{max_error_retries}): "
                    f"{raw_response[:200] if raw_response else '<empty>'}"
                )
                if error_retries >= max_error_retries:
                    break
                continue  # Retry the same turn

            # Reset error counter on successful response
            error_retries = 0

            # Debug: log first turn raw response to diagnose tool-use compliance
            if turn == 0:
                logger.info(f"  Turn 0 raw response (first 500 chars): {raw_response[:500]}")

            # Parse response
            thought, tool_name, tool_input, is_final = self._parse_response(raw_response)

            step = {
                "turn": turn,
                "thought": thought,
                "tool_name": tool_name,
                "tool_input": tool_input[:500],  # Truncate for logging
                "is_final": is_final,
            }

            if is_final:
                final_answer = tool_input
                step["final_answer"] = final_answer
                steps.append(step)

                if self.event_recorder:
                    self.event_recorder.record("final_answer", turn=turn)
                break

            # Execute tool
            if tool_name:
                tool_calls_attempted += 1
                tool_names_used.append(tool_name)

                if self.event_recorder:
                    self.event_recorder.record(
                        "tool_call_start", tool_name=tool_name, turn=turn
                    )

                tool_response, success = self._execute_tool(tool_name, tool_input)
                if success:
                    tool_calls_succeeded += 1

                if self.event_recorder:
                    self.event_recorder.record(
                        "tool_call_end",
                        tool_name=tool_name,
                        turn=turn,
                        success=success,
                    )

                step["tool_response"] = tool_response[:1000]  # Truncate
                step["tool_success"] = success

                # Append assistant response and tool observation
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"OBSERVATION: {tool_response}",
                    }
                )
            else:
                # No tool extracted, append as assistant turn
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please use a tool or provide FINAL_ANSWER.",
                    }
                )

            steps.append(step)

        # If no FINAL_ANSWER was given, run a synthesis turn
        if not final_answer:
            # SYNTHESIS TURN: one extra LLM call to force answer synthesis
            synthesis_prompt = (
                "You have used all your tool-use turns. Based on all the information "
                "you have gathered above, you MUST provide your final answer now.\n\n"
                f"The original question was: {query}\n\n"
                "Respond with ONLY:\n"
                "FINAL_ANSWER: <your concise answer to the original question>"
            )
            messages.append({"role": "user", "content": synthesis_prompt})

            if self.event_recorder:
                self.event_recorder.record(
                    "lm_inference_start", turn=len(steps), model_id=self.model_id
                )

            truncated_messages = self._truncate_context(messages)
            raw_response, usage = self._call_model(truncated_messages)

            if self.event_recorder:
                self.event_recorder.record(
                    "lm_inference_end",
                    turn=len(steps),
                    model_id=self.model_id,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )

            # Parse for FINAL_ANSWER
            thought, tool_name, tool_input, is_final = self._parse_response(raw_response)
            if is_final:
                final_answer = tool_input
            else:
                # Use the synthesis response directly — at least it's LM-generated
                final_answer = raw_response.strip()

            steps.append({
                "turn": len(steps),
                "thought": thought or "Synthesis turn",
                "tool_name": "synthesis",
                "tool_input": final_answer[:500],
                "is_final": True,
                "final_answer": final_answer,
            })

            if self.event_recorder:
                self.event_recorder.record("final_answer", turn=len(steps) - 1)

        # Clean any residual thinking markers from final answer
        if final_answer:
            final_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL)
            final_answer = re.sub(r"\[Thinking\].*?\[/Thinking\]", "", final_answer, flags=re.DOTALL)
            final_answer = re.sub(r"\[Thinking\]\s*\n?", "", final_answer)
            final_answer = final_answer.strip()

        if self.event_recorder:
            self.event_recorder.record(
                "agent_end",
                tool_calls_attempted=tool_calls_attempted,
                tool_calls_succeeded=tool_calls_succeeded,
                num_turns=len(steps),
            )

        return RunResult(
            content=final_answer,
            tool_calls_attempted=tool_calls_attempted,
            tool_calls_succeeded=tool_calls_succeeded,
            tool_names_used=tool_names_used,
            num_turns=len(steps),
            steps=steps,
        )


__all__ = ["TextToolAgent", "RunResult"]
