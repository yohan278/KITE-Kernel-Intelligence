"""Agent implementations subdirectory.

Contains specific agent implementations:
- React: ReAct agent using Agno framework
- OpenHands: OpenHands SDK-based agent
- Terminus: Terminal-based agent for Docker containers
- Orchestrator: Trained policy-based orchestrator
"""

__all__ = [
    "React",
    "OpenHands",
    "Terminus",
    "Orchestrator",
]

# Lazy imports for all agents (they have heavy/optional dependencies)
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
