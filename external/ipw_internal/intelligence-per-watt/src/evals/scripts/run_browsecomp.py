#!/usr/bin/env python3
"""
BrowseComp benchmark runner with React orchestrator.

Usage:
    export OPENAI_API_KEY="your-key"
    export TAVILY_API_KEY="your-tavily-key"  # For web search
    
    uv run python scripts/run_browsecomp.py                         # Default: gpt-5-mini-2025-08-07, 3 samples
    uv run python scripts/run_browsecomp.py --no-browsing           # Baseline (no web access)
    uv run python scripts/run_browsecomp.py --model gpt-5-mini --limit 20 --concurrency 2
    uv run python scripts/run_browsecomp.py --max-steps 50 --timeout 300  # Limit agent steps and time
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run BrowseComp benchmark")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="Model for answering (default: gpt-5-mini-2025-08-07)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (default: OPENAI_API_KEY env)")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples (default: 3)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--browse-mode", choices=["live", "cached"], default="live", help="Browsing mode (default: live)")
    parser.add_argument("--with-browsing", dest="with_browsing", action="store_true", default=True)
    parser.add_argument("--no-browsing", dest="with_browsing", action="store_false", help="Disable browsing (baseline)")
    parser.add_argument("--search-provider", choices=["tavily", "serper"], default="tavily")
    parser.add_argument("--max-search-results", type=int, default=5)
    parser.add_argument("--judge-model", default="gpt-5-mini-2025-08-07", help="Judge model (default: gpt-5-mini-2025-08-07)")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--concurrency", type=int, default=2, help="Number of parallel questions (default: 2)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps per question (default: 50)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per question in seconds (default: 300)")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("BrowseComp Benchmark with React Orchestrator")
    logger.info("=" * 70)
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    
    if args.with_browsing and args.browse_mode == "live":
        key_env = "TAVILY_API_KEY" if args.search_provider == "tavily" else "SERPER_API_KEY"
        if not os.getenv(key_env):
            logger.warning(f"{key_env} not set. Web search may fail.")
    
    from agno.models.openai import OpenAIChat
    from agno.agent import Agent
    from evals.benchmarks.browsecomp import BrowseCompBenchmark, create_browsecomp_tools
    
    class React:
        """Minimal React agent wrapper using Agno."""
        def __init__(self, model, tools, instructions=None, max_steps=None, **kwargs):
            # Note: max_steps stored for potential future use, but agno Agent uses timeout instead
            self.max_steps = max_steps
            self.agent = Agent(
                model=model, 
                tools=tools, 
                instructions=instructions or "You are a helpful assistant.",
                **kwargs
            )
        
        def run(self, prompt: str):
            response = self.agent.run(prompt)
            return response.content if hasattr(response, 'content') else str(response)
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    logger.info(f"\n1. Config: limit={args.limit}, shuffle={args.shuffle}, browse_mode={args.browse_mode}, with_browsing={args.with_browsing}")
    logger.info(f"   concurrency={args.concurrency}, max_steps={args.max_steps}, timeout={args.timeout}s")
    logger.info(f"2. Creating model ({args.model})...")
    
    # Instructions for the agent
    instructions_with_browsing = (
        "You are a helpful AI that can search the web. Use browsing tools to find and verify information. "
        "Provide answers in format: Explanation, Exact Answer, Confidence."
    )
    instructions_no_browsing = (
        "You are a helpful AI. Answer based on your knowledge. Format: Explanation, Exact Answer, Confidence."
    )
    instructions = instructions_with_browsing if args.with_browsing else instructions_no_browsing
    
    # Factory function to create fresh orchestrators (needed for parallel execution)
    def create_orchestrator():
        model = OpenAIChat(id=args.model, api_key=api_key)
        tools = []
        if args.with_browsing:
            tools = create_browsecomp_tools(
                mode=args.browse_mode, 
                cache_dir=cache_dir, 
                search_provider=args.search_provider, 
                max_search_results=args.max_search_results
            )
        return React(model=model, tools=tools, instructions=instructions, max_steps=args.max_steps)
    
    if args.with_browsing:
        logger.info(f"3. Browsing enabled (mode={args.browse_mode}, provider={args.search_provider})")
    else:
        logger.info("3. Browsing disabled (baseline)")
    
    logger.info("4. Creating orchestrator factory...")
    
    logger.info(f"5. Creating judge ({args.judge_model})...")
    judge_model = OpenAIChat(id=args.judge_model, api_key=args.judge_api_key or api_key)
    judge_orchestrator = React(model=judge_model, tools=[], instructions="You are a strict evaluator determining if answers match.")
    
    logger.info("6. Creating benchmark...")
    benchmark = BrowseCompBenchmark(
        shuffle=args.shuffle, seed=args.seed, limit=args.limit, cache_dir=cache_dir,
        with_browsing=args.with_browsing, judge_orchestrator=judge_orchestrator,
        concurrency=args.concurrency, sample_timeout=args.timeout,
    )
    
    logger.info("\n7. Running benchmark...\n")
    start_time = time.time()
    try:
        # Use orchestrator_factory for parallel execution
        responses = benchmark.generate_responses(orchestrator_factory=create_orchestrator)
        metrics = benchmark.evaluate_responses(responses)
        total_time = time.time() - start_time
        
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
        logger.info(f"  Total Wall Time:     {total_time:.1f}s")
        
        logger.info(f"\nDetailed Results:")
        for uid, r in benchmark.get_results().items():
            logger.info(f"  {'✓' if r.is_correct else '✗'} {uid}")
            logger.info(f"     Q: {r.question[:70]}...")
            logger.info(f"     Expected: {r.answer}")
            logger.info(f"     Got: {r.extracted_answer}")
            logger.info(f"     Confidence: {r.confidence:.0f}%")
        
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark completed successfully!")
        logger.info("=" * 70)
        logger.info(f"\nModel: {args.model}, Judge: {args.judge_model}")
        logger.info(f"Browsing: {'enabled' if args.with_browsing else 'disabled'} (mode: {args.browse_mode})")
        logger.info(f"Concurrency: {args.concurrency}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.1f}%, ECE: {metrics.get('ece_10bin', 0):.1f}%")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
