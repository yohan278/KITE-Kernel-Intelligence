#!/usr/bin/env python3
"""
Example script for running the GAIA benchmark with React orchestrator.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="your-api-key"
    
    # Run with default settings (gpt-5-mini-2025-08-07, 3 samples)
    uv run python scripts/run_gaia.py
    
    # Run with custom model and sample count
    uv run python scripts/run_gaia.py --model gpt-4o --limit 10
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
    """Run GAIA benchmark with React orchestrator and OpenAI."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with React orchestrator")
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
        "--subset",
        type=str,
        default="2023_level1",
        choices=["2023_all", "2023_level1", "2023_level2", "2023_level3"],
        help="Dataset subset to use (default: 2023_level1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to use (default: validation)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("GAIA Benchmark with React Orchestrator")
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
    from evals.benchmarks.gaia import GAIABenchmark, create_gaia_tools
    
    # Configure benchmark
    logger.info(f"\n1. Configuring benchmark...")
    logger.info(f"   Subset: {args.subset}")
    logger.info(f"   Split: {args.split}")
    logger.info(f"   Limit: {args.limit}")
    
    benchmark = GAIABenchmark(
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        shuffle=False,
    )
    
    # Create OpenAI model
    logger.info(f"\n2. Creating OpenAI model ({args.model})...")
    model = OpenAIChat(
        id=args.model,
        api_key=api_key,
    )
    
    # Create file handling tools for GAIA
    logger.info("3. Creating file handling tools...")
    tools = create_gaia_tools()
    logger.info(f"   Created {len(tools)} file handling tools")
    
    # Create React orchestrator with file tools
    logger.info("4. Creating React orchestrator...")
    orchestrator = React(
        model=model,
        tools=tools,
        instructions=(
            "You are a helpful AI assistant that can read and analyze various file types. "
            "When a file path is provided in the question, use the appropriate file reading tool to access it. "
            "Answer questions accurately and concisely. Follow the instructions in the prompt carefully."
        ),
    )
    
    # Run benchmark
    logger.info("\n5. Running benchmark...\n")
    try:
        metrics = benchmark.run_benchmark(orchestrator)
        
        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Total Samples:    {int(metrics['total_samples'])}")
        logger.info(f"  Valid Samples:    {int(metrics['valid_samples'])}")
        logger.info(f"  Accuracy:         {metrics['accuracy']}%")
        logger.info(f"  Correct:          {int(metrics['correct_count'])}")
        logger.info(f"  Incorrect:        {int(metrics['incorrect_count'])}")
        logger.info(f"  Avg Response Time: {metrics['avg_response_time_seconds']:.2f}s")
        
        # Per-level metrics
        logger.info(f"\nPer-Level Breakdown:")
        for key in sorted(metrics.keys()):
            if key.startswith('level_') and key.endswith('_accuracy'):
                level = key.split('_')[1]
                accuracy = metrics[key]
                correct = int(metrics.get(f'level_{level}_correct', 0))
                total = int(metrics.get(f'level_{level}_total', 0))
                logger.info(f"  Level {level}: {accuracy}% ({correct}/{total})")
        
        # Get detailed results
        logger.info(f"\nDetailed Sample Results:")
        results = benchmark.get_results()
        for task_id, result in results.items():
            status = "✓" if result.is_correct else "✗"
            logger.info(f"  {status} {task_id[:8]}... (Level {result.level})")
            logger.info(f"     Q: {result.question[:70]}...")
            logger.info(f"     Expected: {result.ground_truth}")
            logger.info(f"     Got:      {result.model_answer}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark completed successfully!")
        logger.info("=" * 70)
        logger.info(f"\nModel used: {args.model}")
        logger.info(f"Total samples: {int(metrics['total_samples'])}")
        logger.info(f"Final accuracy: {metrics['accuracy']}%")
        
    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
