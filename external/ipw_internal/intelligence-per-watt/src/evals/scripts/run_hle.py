#!/usr/bin/env python3
"""
Example script for running the HLE (Humanity's Last Exam) benchmark with React orchestrator.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="your-api-key"

    # Run with default settings (gpt-5-mini-2025-08-07, 3 samples)
    uv run python scripts/run_hle.py

    # Run with custom model and sample count
    uv run python scripts/run_hle.py --model gpt-4o --limit 10
"""
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run HLE benchmark with React orchestrator and OpenAI."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run HLE benchmark with React orchestrator")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="OpenAI model to use (default: gpt-5-mini-2025-08-07)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: uses OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of samples to evaluate (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category",
    )
    parser.add_argument(
        "--grading",
        type=str,
        default="contains",
        choices=["exact_match", "contains", "fuzzy"],
        help="Grading method (default: contains)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("HLE (Humanity's Last Exam) Benchmark with React Orchestrator")
    logger.info("=" * 70)

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("ERROR: OpenAI API key not found!")
        logger.error("Please set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Import dependencies
    from agno.models.openai import OpenAIChat
    from agents import React
    from evals.benchmarks.hle import HLEBenchmark

    # Configure benchmark
    logger.info(f"\n1. Configuring benchmark...")
    logger.info(f"   Split: {args.split}")
    logger.info(f"   Limit: {args.limit}")
    logger.info(f"   Category: {args.category or 'all'}")
    logger.info(f"   Grading: {args.grading}")

    benchmark = HLEBenchmark(
        split=args.split,
        limit=args.limit,
        category_filter=args.category,
        grading_method=args.grading,
    )

    # Create OpenAI model
    logger.info(f"\n2. Creating OpenAI model ({args.model})...")
    model = OpenAIChat(
        id=args.model,
        api_key=api_key,
    )

    # Create React orchestrator
    logger.info("3. Creating React orchestrator...")
    orchestrator = React(
        model=model,
        instructions=(
            "You are a helpful AI assistant solving challenging exam questions. "
            "Think step by step and provide clear, accurate answers. "
            "When answering, be concise and focus on the key points."
        ),
    )

    # Run benchmark
    logger.info("\n4. Running benchmark...\n")
    try:
        metrics = benchmark.run_benchmark(orchestrator)

        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 70)

        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Total Samples:    {int(metrics.get('num_samples', 0))}")
        logger.info(f"  Correct:          {int(metrics.get('num_correct', 0))}")
        logger.info(f"  Accuracy:         {metrics.get('accuracy', 0):.2%}")
        logger.info(f"  Avg Latency:      {metrics.get('avg_latency_seconds', 0):.2f}s")

        # Efficiency metrics
        logger.info(f"\nEfficiency Metrics:")
        logger.info(f"  Avg Cost:         ${metrics.get('avg_cost_usd', 0):.4f}")
        logger.info(f"  Avg Energy:       {metrics.get('avg_energy_joules', 0):.2f}J")
        logger.info(f"  IPJ:              {metrics.get('ipj', 0):.6f}")

        logger.info("\n" + "=" * 70)
        logger.info("Benchmark completed successfully!")
        logger.info("=" * 70)
        logger.info(f"\nModel used: {args.model}")
        logger.info(f"Total samples: {int(metrics.get('num_samples', 0))}")
        logger.info(f"Final accuracy: {metrics.get('accuracy', 0):.2%}")

        # Save results if requested
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"\nResults saved to: {output_path}")

    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
