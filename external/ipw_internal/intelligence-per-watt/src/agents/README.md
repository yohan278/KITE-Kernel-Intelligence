# Agents

Agent implementations and MCP (Model Context Protocol) server management.

## Directory Structure

```
src/
├── base.py           # BaseAgent class with telemetry hooks
├── agents/           # Agent implementations
│   ├── react.py          # ReAct agent (Agno-based)
│   ├── openhands.py      # OpenHands agent wrapper
│   └── terminus.py       # Terminus agent wrapper
├── mcp/              # MCP tool servers
│   ├── base.py           # BaseMCPServer with automatic telemetry
│   ├── tool_server.py    # Calculator, WebSearch, Think tools
│   ├── ollama_server.py  # Local Ollama models
│   ├── vllm_server.py    # vLLM models
│   └── openai_server.py  # OpenAI API
├── training/         # Training code (GRPO, SFT)
├── serve.py          # FastAPI server for hosting agents
└── tools/            # Additional tool implementations
```

## Usage

```python
from agents import React, BaseAgent
from agents.mcp import OllamaMCPServer, CalculatorServer

# Create agent with MCP tools
model = ...  # Agno model
agent = React(model=model, tools=[calculator])
result = agent.run("What is 2+2?")
```

## Telemetry Integration

Agents support `EventRecorder` for per-action energy tracking:
- Tool calls emit `tool_call_start` / `tool_call_end` events
- LM inference emits `lm_inference_start` / `lm_inference_end` events
