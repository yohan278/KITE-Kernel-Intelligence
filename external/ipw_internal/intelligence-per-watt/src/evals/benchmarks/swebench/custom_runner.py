"""
Custom agent runner for SWE-bench evaluation.

Runs custom orchestrators (any BaseOrchestrater subclass) on a SWE-bench instance.
The agent runs on the HOST, tools execute commands in the Docker container.

Prompt formatting: Wraps problem_statement in <issue> tags, matching SWE-agent's format.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Type

from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude

from agents import BaseAgent, React

from .swe_env_wrapper import SWEBenchEnv
from .container_tools import create_tools, get_tool_descriptions

if TYPE_CHECKING:
    from .dataset import SWEBenchSample

logger = logging.getLogger(__name__)


# System prompt for SWE-bench agents
SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a GitHub issue.

You are working in a git repository at /testbed. The repository has been checked out
to a specific commit where the issue exists. Your goal is to make the minimal code
changes necessary to fix the issue described.

IMPORTANT GUIDELINES:
1. First, understand the issue by reading the problem statement carefully
2. Explore the codebase to understand the relevant code
3. Make focused, minimal changes to fix the issue
4. Test your changes if possible
5. When done, call submit_patch() to finalize your changes

{tool_descriptions}

Remember:
- You are working in the /testbed directory
- Use git_diff() to review your changes before submitting
- Keep changes minimal and focused on the issue
- Don't introduce unnecessary modifications
"""


def create_model(provider: str, model_name: str, base_url: str | None = None):
    """
    Create a model instance for the orchestrator.

    Args:
        provider: Model provider ("openai" or "anthropic")
        model_name: Model identifier (e.g., "gpt-4o", "Qwen/Qwen3-8B")
        base_url: Optional base URL for OpenAI-compatible APIs (e.g. vLLM)

    Returns:
        Model instance compatible with agno
    """
    if provider == "openai":
        kwargs = {"id": model_name}
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "dummy")
        return OpenAIChat(**kwargs)
    elif provider == "anthropic":
        return Claude(id=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def run_custom_on_sample(
    sample: "SWEBenchSample",
    model_name: str = "gpt-4o",
    provider: str = "openai",
    max_iterations: int = 10,
    timeout: int = 1800,
    orchestrator_cls: Type[BaseAgent] = React,
    use_swe_agent_tools: bool = True,
    base_url: str | None = None,
) -> tuple[str, str]:
    """
    Run a custom orchestrator on a SWE-bench sample.

    The agent runs on the HOST machine, but all file operations
    and commands execute inside the Docker container via SWEBenchEnv.

    Args:
        sample: SWEBenchSample to process
        model_name: Model identifier
        provider: Model provider
        max_iterations: Maximum agent iterations
        timeout: Overall timeout in seconds
        orchestrator_cls: Orchestrator class
        use_swe_agent_tools: If True (default), uses SWE-agent tools (bash,
               str_replace_editor, submit). If False, no tools are passed.
        base_url: Optional base URL for OpenAI-compatible APIs (e.g. vLLM)

    Returns:
        Tuple of (agent_output, patch)
    """
    logger.info(f"Running {orchestrator_cls.__name__} on {sample.instance_id}")
    logger.info(f"Model: {provider}/{model_name}")
    
    env = None
    try:
        # 1. Create and start the SWE-bench environment
        env = SWEBenchEnv.from_sample(sample)
        env.start()
        logger.info(f"Container started for {sample.instance_id}")
        
        # 2. Create tools that execute in the container
        if use_swe_agent_tools:
            agent_tools = create_tools(env)
            tool_descriptions = get_tool_descriptions()
            logger.debug(f"Using {len(agent_tools)} SWE-agent tools")
        else:
            agent_tools = []
            tool_descriptions = ""
            logger.debug("No tools provided")
        
        # 3. Create the model
        model = create_model(provider, model_name, base_url=base_url)
        
        # 4. Create system prompt with tool descriptions
        system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions
        )
        
        # 5. Create orchestrator (React, CoT, etc.)
        agent = orchestrator_cls(
            model=model,
            tools=agent_tools,
            instructions=system_prompt,
        )
        
        # 6. Format the problem statement
        problem_statement = _format_problem_statement(sample)
        
        # 7. Run the agent
        logger.info("Starting agent execution...")
        response = agent.run(problem_statement)
        
        # 8. Extract full conversation with tool calls
        agent_output = _serialize_messages(response)
        
        # 9. Get the final patch
        patch = env.get_patch()
        logger.info(f"Agent completed. Patch length: {len(patch)} chars")
        
        return agent_output, patch
        
    except Exception as e:
        logger.error(f"Error running orchestrator: {e}", exc_info=True)
        return f"Error: {e}", ""
        
    finally:
        if env is not None:
            env.close()
            logger.info("Container closed")


def _serialize_messages(response) -> str:
    """
    Serialize the full conversation including tool calls to JSON.
    
    Args:
        response: RunOutput from agent.run()
        
    Returns:
        JSON string with full conversation history
    """
    if not hasattr(response, "messages") or not response.messages:
        # Fallback to content if no messages
        return response.content if hasattr(response, "content") else str(response)
    
    messages = []
    for msg in response.messages:
        # Try to use to_dict() if available (agno Message objects have this)
        if hasattr(msg, "to_dict"):
            msg_dict = msg.to_dict()
        else:
            msg_dict = {
                "role": getattr(msg, "role", "unknown"),
                "content": getattr(msg, "content", None),
            }
            
            # Include tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            
            # Include tool call ID for tool responses
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            
            # Include tool name and args for tool messages
            if hasattr(msg, "tool_name") and msg.tool_name:
                msg_dict["tool_name"] = msg.tool_name
            if hasattr(msg, "tool_args") and msg.tool_args:
                msg_dict["tool_args"] = msg.tool_args
        
        messages.append(msg_dict)
    
    return json.dumps(messages, indent=2, default=str)


def _format_problem_statement(sample: "SWEBenchSample") -> str:
    """
    Format the problem statement for the agent.
    
    Uses SWE-bench standard format with <issue> tags.
    
    Args:
        sample: SWEBenchSample
        
    Returns:
        Formatted problem statement
    """
    parts = [f"<issue>\n{sample.problem_statement}\n</issue>"]
    
    if sample.hints_text:
        parts.append(f"\n<hints>\n{sample.hints_text}\n</hints>")
    
    parts.append(f"\nRepository: {sample.repo}")
    parts.append(f"Base commit: {sample.base_commit}")
    
    return "\n".join(parts)

