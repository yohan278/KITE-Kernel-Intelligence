# Orchestrator

An intelligent orchestrator that learns to delegate tasks between local models, cloud APIs, and tools while optimizing for accuracy, cost, latency, and energy consumption.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for local models)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:1.5b
```

### Set API Keys

```bash
# Create .env file or export directly
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
export TAVILY_API_KEY="tvly-..."
```

### Generate Trajectories

```bash
# Start vLLM server for teacher model
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000

# Generate 100K trajectories
python -m src.cli.generate_trajectories \
  --config configs/trajectory_generation.yaml \
  --limit 100000
```

### Train Model

```bash
# SFT training on Qwen3-8B
python -m src.cli.train_sft \
  --model Qwen/Qwen3-8B \
  --train-only \
  --epochs 3
```

### Run Evaluation

```bash
# HLE evaluation
python evals/run_eval.py \
  --benchmark hle \
  --model Qwen/Qwen3-8B
```

## Project Structure

```
src/orchestrator/
├── data/                   # Data pipeline
│   ├── trajectory_generator.py
│   ├── episode_builder.py
│   └── sft_trace_generator.py
├── training/               # Training infrastructure
│   ├── sft_trainer.py
│   └── policy.py
├── cli/                    # CLI commands
│   ├── generate_trajectories.py
│   └── train_sft.py
├── configs/                # Configuration files
│   ├── trajectory_generation.yaml
│   └── sft_training.yaml
└── tests/                  # Unit tests

# Related packages in src/:
# - src/agents/mcp/       # MCP server implementations (Ollama, OpenAI, vLLM, etc.)
# - src/evals/            # Evaluation benchmarks (Tau-bench, etc.)
```

## Available Tools

### Utility Tools
| Tool | Description |
|------|-------------|
| `calculator` | Safe math expression evaluation |
| `think` | Internal reasoning/scratchpad |
| `code_interpreter` | Python code execution sandbox |
| `web_search` | Tavily API for web search |

### Local Models (Ollama)
| Tool | Size |
|------|------|
| `ollama:qwen3:1.5b` | 1.5B |
| `ollama:qwen2.5:0.5b` | 0.5B |
| `ollama:llama3.2:1b` | 1B |

### Cloud APIs
| Tool | Provider |
|------|----------|
| `openai:gpt-5-mini` | OpenAI |
| `openai:gpt-5` | OpenAI |
| `anthropic:claude-sonnet-4-5` | Anthropic |
| `anthropic:claude-opus-4-5` | Anthropic |

### Large Models (OpenRouter/vLLM)
| Tool | Size |
|------|------|
| `openrouter:qwen/qwen3-32b` | 32B |
| `openrouter:z-ai/glm-4.7` | Math specialist |
| `openrouter:qwen/qwen3-coder-plus` | Code specialist |
| `vllm:Qwen/Qwen3-32B` | 32B |

## Configuration

### Trajectory Generation (`configs/trajectory_generation.yaml`)

```yaml
teacher:
  model: "vllm:Qwen/Qwen3-32B@8000"
  temperature: 0.7

tools:
  - calculator
  - think
  - ollama:qwen3:1.5b
  - openrouter:qwen/qwen3-32b

generation:
  traces_per_query: 1
  max_attempts: 3
```

### SFT Training (`configs/sft_training.yaml`)

```yaml
model:
  partial_pretrain: Qwen/Qwen3-8B
  enable_gradient_checkpointing: true

data:
  max_length: 32768
  train_batch_size: 64

trainer:
  total_epochs: 3
  n_gpus_per_node: 8
```

## GPU Requirements

| Task | GPUs | Memory |
|------|------|--------|
| Trajectory Generation (Qwen3-32B) | 4x A100 | 160GB |
| SFT Training (Qwen3-8B) | 8x A100 | 320GB |
| Inference (Qwen3-8B) | 1x A100 | 40GB |

## References

- [ToolOrchestra Paper](https://arxiv.org/abs/2511.21689)
- [ToolScale Dataset](https://huggingface.co/datasets/nvidia/ToolScale)
- [Qwen3 Models](https://huggingface.co/Qwen)
