# MCP Servers

Model Context Protocol server implementations for tools and LM inference.

## Server Types

| Server | Purpose | Endpoint |
|--------|---------|----------|
| `CalculatorServer` | Math expressions | Local |
| `OllamaMCPServer` | Local Ollama models | `localhost:11434` |
| `VLLMMCPServer` | vLLM OpenAI-compat | `localhost:8000` |
| `OpenAIMCPServer` | OpenAI API | `api.openai.com` |

## Telemetry

All servers inherit from `BaseMCPServer` which provides automatic telemetry:

```python
@dataclass
class MCPToolResult:
    content: str
    usage: dict = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_seconds: Optional[float] = None
    telemetry_samples: List[Any] = field(default_factory=list)
```

## Usage

```python
from agents.mcp import OllamaMCPServer, CalculatorServer

# LM server
ollama = OllamaMCPServer(model="llama3.2:1b")
result = ollama.execute("Explain quantum computing")

# Tool server
calc = CalculatorServer()
result = calc.execute("sqrt(144) + 5")
```

## Adding New Servers

1. Inherit from `BaseMCPServer`
2. Implement `_execute_impl(prompt: str) -> MCPToolResult`
3. Server auto-captures latency and telemetry samples
