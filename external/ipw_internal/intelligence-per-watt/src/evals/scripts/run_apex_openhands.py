#!/usr/bin/env python3
"""
Evaluate OpenHands on the APEX benchmark.

Usage:
    python run_apex.py [--model MODEL] [--limit N] [--domains DOMAIN1,DOMAIN2] [--grading-model MODEL]

Environment Variables:
    OPENAI_API_KEY: Required for OpenAI models
    ANTHROPIC_API_KEY: Required for Anthropic models
    GOOGLE_API_KEY: Required for Gemini grading model

Example:
    # Run with default settings (all domains, no limit)
    python run_apex.py
    
    # Run on 10 samples from Law domain
    python run_apex.py --model gpt-4o --limit 10 --domains Law
    
    # Run with specific grading model
    python run_apex.py --model claude-sonnet-4-20250514 --grading-model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from openhands.sdk import LLM

from evals.benchmarks.apex.main import APEXBenchmark
from agents import OpenHands


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
    
    return logging.getLogger("apex_eval")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OpenHands on APEX benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to evaluate (default: gpt-4o). Supports OpenAI, Anthropic, etc.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model (uses env vars if not provided)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains to evaluate (default: all). "
             "Options: Investment Banking, Management Consulting, Law, Medicine",
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
    
    # Grading configuration
    parser.add_argument(
        "--grading-model",
        type=str,
        default="gemini-2.5-flash",
        help="Model for grading responses (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--grading-api-key",
        type=str,
        default=None,
        help="API key for grading model (uses env vars if not provided)",
    )
    parser.add_argument(
        "--max-concurrent-grading",
        type=int,
        default=5,
        help="Max concurrent grading calls (default: 5)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: ./results)",
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
    
    # Parse domains if provided
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
        logger.info(f"Filtering to domains: {domains}")
    
    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the LLM for OpenHands
    logger.info(f"Initializing OpenHands with model: {args.model}")
    
    llm = LLM(model=args.model, api_key=args.api_key)
    orchestrator = OpenHands(model=llm)
    
    # Initialize the APEX benchmark
    logger.info("Initializing APEX benchmark")
    benchmark = APEXBenchmark(
        limit=args.limit,
        domains=domains,
        shuffle=args.shuffle,
        seed=args.seed,
        grading_model=args.grading_model,
        grading_api_key=args.grading_api_key,
        max_concurrent_grading=args.max_concurrent_grading,
        logger=logger,
    )
    
    # Run the benchmark
    logger.info("=" * 60)
    logger.info("Starting APEX Evaluation with OpenHands")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Grading Model: {args.grading_model}")
    logger.info(f"Limit: {args.limit or 'all'}")
    logger.info(f"Domains: {domains or 'all'}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Run the full benchmark pipeline
        metrics = benchmark.run_benchmark(orchestrator)
        
        # Get detailed results
        detailed_results = benchmark.get_results()
        domain_breakdown = benchmark.get_domain_breakdown()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log results
        logger.info("=" * 60)
        logger.info("APEX Evaluation Complete")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.2f}s")
        logger.info(f"Total Samples: {metrics.get('total_samples', 0):.0f}")
        logger.info(f"Graded Samples: {metrics.get('graded_samples', 0):.0f}")
        logger.info(f"Average Score: {metrics.get('average_score', 0):.2f}%")
        logger.info(f"Pass Rate: {metrics.get('pass_rate', 0):.2f}%")
        logger.info("-" * 60)
        
        # Log domain breakdown
        logger.info("Domain Breakdown:")
        for domain, data in domain_breakdown.items():
            logger.info(
                f"  {domain}: {data.get('average_score', 0):.1f}% avg, "
                f"{data.get('pass_rate', 0):.1f}% pass rate "
                f"({data.get('passed', 0)}/{data.get('total', 0)})"
            )
        
        logger.info("-" * 60)
        logger.info(f"Grading Tokens: {metrics.get('total_grading_tokens', 0):.0f}")
        logger.info(f"Grading Cost: ${metrics.get('total_grading_cost', 0):.4f}")
        logger.info("=" * 60)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace("/", "_").replace(":", "_")
        results_file = output_dir / f"apex_{model_name}_{timestamp}.json"
        
        output_data = {
            "config": {
                "model": args.model,
                "grading_model": args.grading_model,
                "limit": args.limit,
                "domains": domains,
                "shuffle": args.shuffle,
                "seed": args.seed,
            },
            "metrics": metrics,
            "domain_breakdown": domain_breakdown,
            "duration_seconds": duration,
            "timestamp": timestamp,
            "detailed_results": {
                task_id: {
                    "task_id": result.task_id,
                    "domain": result.domain,
                    "score": result.score,
                    "passed": result.passed,
                    "response_time_seconds": result.response_time_seconds,
                    "error": result.error,
                }
                for task_id, result in detailed_results.items()
            },
        }
        
        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        return metrics
        
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
