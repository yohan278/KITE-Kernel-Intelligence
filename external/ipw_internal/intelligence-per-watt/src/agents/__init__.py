"""Agents package - unified agent implementations with MCP and telemetry support.

This package provides:
- agents/: Agent implementations (React, OpenHands, Terminus, Orchestrator)
- mcp/: MCP tool and model servers
- training/: Training pipelines for agents
"""

from agents.base import BaseAgent, BaseOrchestrater

__all__ = [
    # Base class
    "BaseAgent",
    "BaseOrchestrater",  # Backwards compatibility alias
    # Agent implementations (lazy loaded)
    "React",
    "OpenHands",
    "Terminus",
    "Orchestrator",
    # Submodules (lazy loaded)
    "mcp",
    "training",
]

# Lazy imports for all agents and submodules with heavy dependencies
def __getattr__(name: str):
    if name == "React":
        from agents.agents.react import React
        return React
    if name == "OpenHands":
        from agents.agents.openhands import OpenHands
        return OpenHands
    if name == "Terminus":
        from agents.agents.terminus import Terminus
        return Terminus
    if name == "Orchestrator":
        from agents.agents.orchestrator import Orchestrator
        return Orchestrator
    if name == "mcp":
        from agents import mcp
        return mcp
    if name == "training":
        from agents import training
        return training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
