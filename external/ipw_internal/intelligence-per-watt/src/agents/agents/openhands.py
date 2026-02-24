"""OpenHands agent implementation with per-tool energy tracking."""

from __future__ import annotations

import logging
import os
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from openhands.sdk import LLM, Agent, Conversation, Tool, Event, LLMConvertibleEvent
from openhands.sdk.context.condenser.llm_summarizing_condenser import LLMSummarizingCondenser
from openhands.sdk.event.llm_convertible.action import ActionEvent
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition, ToolExecutor
from openhands.tools.preset.default import get_default_agent, get_default_tools

from agents.base import BaseAgent

if TYPE_CHECKING:
    from agents.mcp.base import BaseMCPServer
    from ipw.telemetry.events import EventRecorder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight OpenHands tool wrappers for IPW MCP servers
# ---------------------------------------------------------------------------


class IPWMCPAction(Action):
    """Input action for an MCP-backed tool."""

    query: str


class IPWMCPObservation(Observation):
    """Output observation for an MCP-backed tool."""

    pass  # Uses Observation.content (list of TextContent)


class MCPToolExecutor(ToolExecutor):
    """Executor that delegates to a BaseMCPServer.execute()."""

    def __init__(self, mcp_server: "BaseMCPServer") -> None:
        self._server = mcp_server

    def __call__(
        self, action: IPWMCPAction, conversation: Any = None
    ) -> IPWMCPObservation:
        result = self._server.execute(action.query)
        content = result.content if hasattr(result, "content") else str(result)
        return IPWMCPObservation.from_text(text=content)


class _MCPToolDef(ToolDefinition[IPWMCPAction, IPWMCPObservation]):
    """Concrete ToolDefinition subclass for MCP-backed tools."""

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["_MCPToolDef"]:
        # Not used — instances are created directly in the factory below
        return []


def _register_mcp_tools(mcp_tools: Dict[str, "BaseMCPServer"]) -> List[Tool]:
    """Register MCP servers as OpenHands tools and return Tool specs.

    Each MCP server is wrapped as a ToolDefinition and registered in the
    OpenHands tool registry so the Agent can resolve it by name.

    Args:
        mcp_tools: Mapping of tool name to BaseMCPServer instance.

    Returns:
        List of Tool specs that can be passed to Agent(tools=...).
    """
    tool_specs: List[Tool] = []

    for name, server in mcp_tools.items():
        # Derive a safe OpenHands-compatible tool name
        oh_name = f"mcp_{name}"

        # Build description from the MCP server spec if available
        spec = getattr(server, "_spec", None)
        description = (
            spec.description
            if spec and hasattr(spec, "description")
            else f"Execute the {name} tool"
        )

        # Create the executor bound to this server instance
        executor = MCPToolExecutor(server)

        # Build a factory that returns the ToolDefinition sequence.
        # OpenHands calls: factory(conv_state=conv_state, **params)
        def _make_factory(
            _oh_name: str, _desc: str, _executor: MCPToolExecutor
        ):
            def factory(conv_state: Any = None, **kwargs: Any) -> Sequence[ToolDefinition]:
                tool_def = _MCPToolDef(
                    description=_desc,
                    action_type=IPWMCPAction,
                    observation_type=IPWMCPObservation,
                    executor=_executor,
                    annotations=ToolAnnotations(title=_oh_name),
                )
                # ToolDefinition.name is a ClassVar — override per-instance
                object.__setattr__(tool_def, "name", _oh_name)
                return [tool_def]

            return factory

        register_tool(oh_name, _make_factory(oh_name, description, executor))
        tool_specs.append(Tool(name=oh_name))
        logger.info(f"Registered OpenHands MCP tool: {oh_name}")

    return tool_specs


class OpenHands(BaseAgent):
    """OpenHands agent using the OpenHands SDK with energy telemetry."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: LLM,
        tools: List[Tool] | None = None,
        mcp_tools: Optional[Dict[str, "BaseMCPServer"]] = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenHands agent.

        Args:
            model: The model instance to use.
            tools: List of callable tools/functions for the agent to use.
            mcp_tools: Optional dict mapping tool name to BaseMCPServer instance.
                Each server is registered as an OpenHands tool so the agent can
                call it alongside the default terminal/file_editor tools.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to the Agent constructor.
        """
        super().__init__(event_recorder=event_recorder)

        self.model = model
        self.tools = tools
        self._pending_tool: Optional[str] = None  # Track tool in progress

        # Per-run tracking (reset at the start of each run())
        self._tool_names_used: List[str] = []
        self._num_turns: int = 0

        # Context condenser: summarizes old conversation events when
        # approaching the context window limit, preventing overflow.
        condenser = LLMSummarizingCondenser(
            llm=model,
            max_tokens=24000,  # Trigger condensation at ~75% of 32K limit
            keep_first=2,      # Always keep system prompt + initial query
        )

        if tools:
            # Caller provided explicit tool list
            self.agent = Agent(llm=model, tools=tools, condenser=condenser)
        elif mcp_tools:
            # Register MCP servers as OpenHands tools and merge with defaults
            extra_tool_specs = _register_mcp_tools(mcp_tools)
            self.agent = get_default_agent(llm=model, cli_mode=True)
            # Agent is frozen (Pydantic), so reconstruct with extra tools
            all_tools = list(self.agent.tools) + extra_tool_specs
            self.agent = Agent(
                llm=model,
                tools=all_tools,
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
            )
        else:
            self.agent = get_default_agent(llm=model, cli_mode=True)
            # Reconstruct with condenser attached
            self.agent = Agent(
                llm=model,
                tools=list(self.agent.tools),
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
            )

        self.conversation = Conversation(
            agent=self.agent,
            callbacks=[self._instrumented_callback],
            workspace=os.getcwd(),
            max_iteration_per_run=15,  # Safety cap (default is 500)
        )
        self.current_result = ""

    def _instrumented_callback(self, event: Event) -> None:
        """Instrumented callback that emits telemetry events for tool calls.

        OpenHands SDK fires ActionEvent when a tool call starts and
        ObservationEvent when execution completes.  Both carry a
        ``tool_name`` attribute with the tool's registered name.
        """
        if isinstance(event, ActionEvent):
            tool_name = event.tool_name
            self._pending_tool = tool_name
            self._tool_names_used.append(tool_name)
            self._num_turns += 1
            self._record_event("tool_call_start", tool=tool_name)
        elif isinstance(event, ObservationEvent):
            tool_name = self._pending_tool or event.tool_name
            self._record_event("tool_call_end", tool=tool_name)
            self._pending_tool = None

        # Preserve original behavior for result extraction
        if isinstance(event, LLMConvertibleEvent):
            self.current_result = event.to_llm_message()

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Extract plain text from an OpenHands Message or fallback to str().

        Message.content is Sequence[TextContent | ImageContent].
        We join all TextContent.text values.  Extended-thinking
        ``<think>...</think>`` blocks are stripped so the scorer sees
        only the final answer.
        """
        if hasattr(message, "content") and isinstance(message.content, (list, tuple)):
            from openhands.sdk.llm.message import TextContent

            parts = [
                item.text for item in message.content if isinstance(item, TextContent)
            ]
            if parts:
                text = "\n".join(parts)
            else:
                text = str(message)
        elif isinstance(message, str):
            text = message
        else:
            text = str(message)

        # Strip <think>...</think> blocks (extended thinking output)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Handle unclosed think tags (content before closing </think>)
        text = re.sub(r".*</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def run(self, input: str, **kwargs: Any) -> Any:
        """Run the OpenHands agent with telemetry.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments passed to agent.run().

        Returns:
            RunResult with content, tool_names_used, and num_turns.
        """
        from grid_eval.tool_agent import RunResult

        # Reset per-run tracking
        self._tool_names_used = []
        self._num_turns = 0

        self._record_event("lm_inference_start", model=str(self.model))
        try:
            self.conversation.send_message(input)
            self.conversation.run()

            result = self._extract_text(self.current_result)
            self.current_result = ""
            return RunResult(
                content=result,
                tool_calls_attempted=len(self._tool_names_used),
                tool_calls_succeeded=len(self._tool_names_used),
                tool_names_used=list(self._tool_names_used),
                num_turns=self._num_turns,
            )
        finally:
            self._record_event("lm_inference_end", model=str(self.model))
