# BrowseComp Benchmark for Intelligence Per Watt Evals.
# Paper: https://arxiv.org/pdf/2504.12516
# 1,266 questions requiring web browsing to find hard-to-find information.

from .dataset import decrypt, decrypt_row, load_browsecomp_samples
from .main import BrowseCompBenchmark
from .prompts import BROWSING_INSTRUCTION, format_judge_prompt, format_query, parse_model_response
from .scorer import BrowseCompScorer, compute_metrics
from .browsing import BrowseCompTools, CacheMissError, create_browsecomp_tools
from .types import BrowseCompResult, BrowseCompSample

__all__ = [
    "BrowseCompBenchmark",
    "BrowseCompResult",
    "BrowseCompSample",
    "BrowseCompScorer",
    "BrowseCompTools",
    "BROWSING_INSTRUCTION",
    "CacheMissError",
    "compute_metrics",
    "create_browsecomp_tools",
    "decrypt",
    "decrypt_row",
    "format_judge_prompt",
    "format_query",
    "load_browsecomp_samples",
    "parse_model_response",
]
