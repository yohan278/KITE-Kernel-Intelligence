# tau2-bench

**Model evaluation benchmark for customer service tasks.**

tau2-bench is installed via pip from the official Sierra Research repository. This directory only contains the task data files.

## What it evaluates

tau2-bench tests LLM capability on multi-turn customer service conversations across domains:
- **mock** - Simple task management (good for testing)
- **airline** - Flight booking, cancellations, policies
- **retail** - Order management, returns, product inquiries  
- **telecom** - Account management, billing, tech support

## Important: Model comparison only

tau2-bench controls the agent loop internally. It is **NOT suitable for evaluating orchestration strategies** (React, Terminus, etc.). Use SWE-bench or GAIA for that.

This benchmark answers: "How well does model X handle customer service tasks?"

## Usage

```bash
# Run mock domain with GPT-4.1
python evals/scripts/run_tau2.py --domain mock --model gpt-4.1

# Run retail domain with Claude
python evals/scripts/run_tau2.py --domain retail --model claude-3-5-sonnet-20241022

# Custom endpoint
python evals/scripts/run_tau2.py --model my-model --api-base http://localhost:8000/v1
```

## Data directory

The `data/` folder is auto-downloaded on first run from:
https://github.com/sierra-research/tau2-bench

To manually download:
```bash
python evals/scripts/run_tau2.py --help  # triggers download
```

## Package source

tau2 is installed from git (pinned to a specific commit for reproducibility):
```toml
# In evals/pyproject.toml
tau2 = { git = "https://github.com/sierra-research/tau2-bench.git", rev = "..." }
```

