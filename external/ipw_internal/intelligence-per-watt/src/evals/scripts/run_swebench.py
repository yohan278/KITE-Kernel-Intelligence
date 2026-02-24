#!/usr/bin/env python3
"""
Run the SWE-bench benchmark.

Evaluates AI agents on real-world GitHub issues from SWE-bench.

Usage:
    python run_swebench.py [--dataset DATASET] [--limit N] [--model MODEL]

Environment Variables:
    OPENAI_API_KEY: Required for OpenAI models
    ANTHROPIC_API_KEY: Required for Anthropic models

Examples:
    # Run on 5 samples from verified_mini with default settings
    python run_swebench.py --dataset verified_mini --limit 5
    
    # Run with Claude on specific instances
    python run_swebench.py --model claude-sonnet-4-20250514 --provider anthropic \
        --instance-ids django__django-11790 django__django-12345
    
    # Run without evaluation (just generate patches)
    python run_swebench.py --limit 10 --no-eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file (in evals/ directory)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.swebench import SWEBenchBenchmark, SWEBenchConfig

# Default output directory (relative to the swebench module)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "src" / "benchmarks" / "swebench" / "outputs"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the evaluation run."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    
    return logging.getLogger("swebench_eval")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SWE-bench benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["verified", "verified_mini"],
        default="verified_mini",
        help="Dataset variant (default: verified_mini). "
             "verified=500 tasks, verified_mini=50 tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated list of repos to filter (e.g., django/django,sphinx-doc/sphinx)",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific instance IDs to evaluate",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    
    # Agent configuration
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["react", "openhands"],
        default="react",
        help="Agent/orchestrator type (default: react)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for the agent (default: gpt-4o)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="Model provider (default: openai)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations per task (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum retries on exceptions for OpenHands (default: 0 = no retries)",
    )
    
    # Container configuration
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=1800,
        help="Timeout for agent execution in seconds (default: 1800 = 30 min)",
    )
    
    # Execution configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for running agents (default: 1 = sequential). "
             "Higher values speed up runs but use more memory/API calls.",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip swebench evaluation harness (just generate patches)",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=4,
        help="Max parallel workers for swebench evaluation harness (default: 4)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save results (default: src/benchmarks/swebench/outputs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for this evaluation (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Parse list arguments
    repos = None
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",")]
        logger.info(f"Filtering to repos: {repos}")
    
    # Create config
    config = SWEBenchConfig(
        # Dataset
        dataset=args.dataset,
        shuffle=args.shuffle,
        seed=args.seed,
        limit=args.limit,
        repos=repos,
        instance_ids=args.instance_ids,
        # Agent
        agent_type=args.agent_type,
        model=args.model,
        provider=args.provider,
        max_iterations=args.max_iterations,
        max_retries=args.max_retries,
        # Container
        agent_timeout=args.agent_timeout,
        # Execution
        num_workers=args.num_workers,
        # Evaluation
        run_evaluation=not args.no_eval,
        eval_workers=args.eval_workers,
        # Output
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
    )
    
    # Create benchmark
    logger.info("=" * 60)
    logger.info("SWE-bench Evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Agent: {config.agent_type}")
    logger.info(f"Model: {config.provider}/{config.model}")
    logger.info(f"Max iterations: {config.max_iterations}")
    logger.info(f"Limit: {config.limit or 'all'}")
    logger.info(f"Workers: {config.num_workers} (parallel)" if config.num_workers > 1 else "Workers: 1 (sequential)")
    logger.info(f"Run evaluation: {config.run_evaluation}")
    logger.info("=" * 60)
    
    benchmark = SWEBenchBenchmark(config=config, logger=logger)
    
    start_time = datetime.now()
    
    try:
        # Run benchmark
        metrics = benchmark.run_benchmark()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log results
        logger.info("=" * 60)
        logger.info("SWE-bench Evaluation Complete")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.2f}s")
        logger.info(f"Total Samples: {metrics.get('total_samples', 0):.0f}")
        logger.info(f"Patches Generated: {metrics.get('patches_generated', 0):.0f}")
        logger.info(f"Patch Rate: {metrics.get('patch_rate', 0)*100:.1f}%")
        logger.info(f"Errors: {metrics.get('errors', 0):.0f}")
        
        if config.run_evaluation:
            logger.info(f"Resolved: {metrics.get('resolved', 0):.0f}")
            logger.info(f"Resolve Rate: {metrics.get('resolve_rate', 0)*100:.1f}%")
        
        logger.info(f"Avg Response Time: {metrics.get('avg_response_time_seconds', 0):.1f}s")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {config.output_dir}")
        
        return metrics
        
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

