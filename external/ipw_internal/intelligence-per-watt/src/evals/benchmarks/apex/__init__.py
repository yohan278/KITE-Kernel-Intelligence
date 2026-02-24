# benchmarks/apex/__init__.py
"""
APEX (AI Productivity Index Extended) Benchmark.

Evaluates AI models on economically valuable tasks across professional domains
using the Mercor APEX-v1-extended dataset.

Domains covered:
- Investment Banking
- Management Consulting
- Law
- Medicine

Usage:
    from evals.benchmarks.apex import APEXBenchmark
    
    benchmark = APEXBenchmark(limit=10, domains=["Law"])
    results = benchmark.run_benchmark(orchestrator)
    
Or via registry:
    from evals.registry import get_benchmark
    
    benchmark = get_benchmark("apex")(options={"limit": 10})
"""

from evals.benchmarks.apex.main import (
    APEXBenchmark,
    APEXResult,
    APEXAgentsBenchmark,
    APEXAgentsResult,
)
from evals.benchmarks.apex.dataset import (
    APEXSample,
    get_apex_domains,
    load_apex_samples,
)
from evals.benchmarks.apex.agents import (
    APEXAgentsSample,
    get_apex_agents_job_categories,
    load_apex_agents_samples,
    get_apex_agents_worlds,
    APEX_AGENTS_SYSTEM_PROMPT,
)
from evals.benchmarks.apex.grading import (
    GradingModelConfig,
    GradingResult,
    GradingTask,
)

__all__ = [
    # Main benchmarks
    "APEXBenchmark",
    "APEXResult",
    "APEXAgentsBenchmark",
    "APEXAgentsResult",
    # Dataset (APEX-v1)
    "APEXSample",
    "get_apex_domains",
    "load_apex_samples",
    # Dataset (APEX-Agents)
    "APEXAgentsSample",
    "get_apex_agents_job_categories",
    "load_apex_agents_samples",
    "get_apex_agents_worlds",
    "APEX_AGENTS_SYSTEM_PROMPT",
    # Grading
    "GradingModelConfig",
    "GradingResult",
    "GradingTask",
]

