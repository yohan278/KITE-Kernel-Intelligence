"""MCP (Model Context Protocol) server implementations.

This module provides unified interfaces for:
- Local models (via Ollama, vLLM)
- Cloud APIs (OpenAI, Anthropic, OpenRouter)
- Tools (calculator, web search, code interpreter)
- Retrieval (BM25, dense, grep, hybrid)

All servers automatically capture telemetry (energy, power, cost, latency).
"""

from .base import BaseMCPServer, MCPToolResult
from .openai_server import OpenAIMCPServer
from .anthropic_server import AnthropicMCPServer
from .openrouter_server import OpenRouterMCPServer
from .vllm_server import VLLMMCPServer
from .tool_server import (
    CalculatorServer,
    WebSearchServer,
    CodeInterpreterServer,
    ThinkServer,
    FileReadServer,
    FileWriteServer,
)
from .tool_registry import ToolRegistry, ToolSpec, ToolCategory, ADPDomainServer, get_registry

# Retrieval module - import lazily to avoid mandatory dependencies
from . import retrieval


def __getattr__(name):
    if name == "OllamaMCPServer":
        from .ollama_server import OllamaMCPServer
        return OllamaMCPServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    "BaseMCPServer",
    "MCPToolResult",
    # Model servers
    "OllamaMCPServer",
    "OpenAIMCPServer",
    "AnthropicMCPServer",
    "OpenRouterMCPServer",
    "VLLMMCPServer",
    # Tool servers
    "CalculatorServer",
    "WebSearchServer",
    "CodeInterpreterServer",
    "ThinkServer",
    "FileReadServer",
    "FileWriteServer",
    # Tool registry
    "ToolRegistry",
    "ToolSpec",
    "ToolCategory",
    "ADPDomainServer",
    "get_registry",
    # Retrieval module
    "retrieval",
]
