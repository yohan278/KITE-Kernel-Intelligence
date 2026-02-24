#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model(provider: str, model_id: str, api_key: str = None):
    """Create a model instance based on the provider."""
    if provider == "openai":
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=model_id, api_key=api_key or os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        from agno.models.anthropic import Claude
        return Claude(id=model_id, api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "gemini":
        from agno.models.google import Gemini
        return Gemini(id=model_id, api_key=api_key or os.getenv("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    defaults = {
        "openai": "gpt-5-mini-2025-08-07",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash-latest",
    }
    return defaults.get(provider, "gpt-5-mini-2025-08-07")


def main():
    parser = argparse.ArgumentParser(description="Run FinanceBench benchmark")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"], help="Model provider (default: openai)")
    parser.add_argument("--model", default=None, help="Model ID (default: provider-specific)")
    parser.add_argument("--api-key", default=None, help="API key (default: from env var)")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples (default: 3)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--with-context", action="store_true", help="Include evidence context in prompts")
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic", "gemini"], help="Judge provider (default: openai)")
    parser.add_argument("--judge-model", default="gpt-5-mini-2025-08-07", help="Judge model (default: gpt-5-mini-2025-08-07)")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    
    # Set default model based on provider if not specified
    if args.model is None:
        args.model = get_default_model(args.provider)
    
    logger.info("=" * 70)
    logger.info("FinanceBench Benchmark")
    logger.info("=" * 70)
    
    # Validate API keys
    env_key_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY"}
    required_key = env_key_map[args.provider]
    if not args.api_key and not os.getenv(required_key):
        logger.error(f"{required_key} not set")
        sys.exit(1)
    
    from agno.agent import Agent
    from evals.benchmarks.financebench import FinanceBenchBenchmark
    
    class SimpleOrchestrator:
        """Simple orchestrator wrapper using Agno."""
        def __init__(self, model, instructions=None, **kwargs):
            self.agent = Agent(model=model, instructions=instructions or "You are a helpful assistant.", **kwargs)
        
        def run(self, prompt: str):
            response = self.agent.run(prompt)
            return response.content if hasattr(response, 'content') else str(response)
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    logger.info(f"\n1. Config: limit={args.limit}, shuffle={args.shuffle}, with_context={args.with_context}")
    logger.info(f"2. Creating model ({args.provider}/{args.model})...")
    model = get_model(args.provider, args.model, args.api_key)
    
    logger.info("3. Creating orchestrator...")
    instructions = (
        "You are a financial analyst assistant. Answer questions about financial documents "
        "accurately and concisely. Provide answers in format: Explanation, Exact Answer, Confidence."
    )
    orchestrator = SimpleOrchestrator(model=model, instructions=instructions)
    
    logger.info(f"4. Creating judge ({args.judge_provider}/{args.judge_model})...")
    judge_model = get_model(args.judge_provider, args.judge_model, args.judge_api_key)
    judge_orchestrator = SimpleOrchestrator(model=judge_model, instructions="You are a strict evaluator determining if answers match.")
    
    logger.info("5. Creating benchmark...")
    benchmark = FinanceBenchBenchmark(
        shuffle=args.shuffle, seed=args.seed, limit=args.limit, cache_dir=cache_dir,
        with_context=args.with_context, judge_orchestrator=judge_orchestrator
    )
    
    logger.info("\n6. Running benchmark...\n")
    try:
        metrics = benchmark.run_benchmark(orchestrator)
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 70)
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Total Samples:       {int(metrics.get('total_count', 0))}")
        logger.info(f"  Valid Samples:       {int(metrics.get('valid_count', 0))}")
        logger.info(f"  Error Samples:       {int(metrics.get('error_count', 0))}")
        logger.info(f"  Accuracy:            {metrics.get('accuracy', 0):.1f}%")
        logger.info(f"  Correct:             {int(metrics.get('correct_count', 0))}")
        logger.info(f"  ECE (10-bin):        {metrics.get('ece_10bin', 0):.1f}%")
        logger.info(f"  Avg Response Time:   {metrics.get('avg_response_time_seconds', 0):.2f}s")
        
        if "accuracy_by_question_type" in metrics:
            logger.info(f"\nAccuracy by Question Type:")
            for qtype, acc in metrics["accuracy_by_question_type"].items():
                logger.info(f"  {qtype}: {acc:.1f}%")
        
        logger.info(f"\nDetailed Results:")
        for uid, r in benchmark.get_results().items():
            logger.info(f"  {'✓' if r.is_correct else '✗'} {uid} ({r.company})")
            logger.info(f"     Q: {r.question[:70]}...")
            logger.info(f"     Expected: {r.ground_truth[:100]}...")
            logger.info(f"     Got: {r.extracted_answer[:100]}...")
            logger.info(f"     Confidence: {r.confidence:.0f}%")
        
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark completed successfully!")
        logger.info("=" * 70)
        logger.info(f"\nModel: {args.provider}/{args.model}, Judge: {args.judge_provider}/{args.judge_model}")
        logger.info(f"Context: {'enabled' if args.with_context else 'disabled'}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.1f}%, ECE: {metrics.get('ece_10bin', 0):.1f}%")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
