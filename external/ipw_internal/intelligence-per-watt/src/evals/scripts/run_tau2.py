#!/usr/bin/env python3
"""
Run tau2-bench for model comparison.

tau2-bench evaluates LLM capability on customer service tasks.
Use this to compare different models (GPT-4, Claude, Gemini, etc.)

Note: tau2-bench controls the execution loop, so this is NOT suitable
for evaluating orchestration strategies. Use SWE-bench or GAIA for that.

Usage:
    python run_tau2.py --domain mock --model gpt-4.1
    python run_tau2.py --domain retail --model claude-3-5-sonnet-20241022
    python run_tau2.py --domain airline --model gemini/gemini-2.0-flash
    
    # Custom API endpoint (OpenAI-compatible)
    python run_tau2.py --model my-model --api-base http://localhost:8000/v1
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def ensure_tau2_data():
    """Ensure tau2 data directory exists. Clone from GitHub if missing."""
    # Check if TAU2_DATA_DIR is already set
    if os.environ.get("TAU2_DATA_DIR"):
        data_dir = Path(os.environ["TAU2_DATA_DIR"])
        if data_dir.exists():
            return  # Already configured
    
    # Data location: evals/src/benchmarks/tau/data
    script_dir = Path(__file__).resolve().parent
    tau_dir = script_dir.parent / "src" / "benchmarks" / "tau"
    data_dir = tau_dir / "data"
    
    if not data_dir.exists():
        print(f"Downloading tau2-bench data to {data_dir}...")
        tau_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone only the data directory using sparse checkout
        temp_clone = tau_dir / "_clone_temp"
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://github.com/sierra-research/tau2-bench.git",
                str(temp_clone),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "data"],
            cwd=str(temp_clone),
            check=True,
            capture_output=True,
        )
        # Move data folder and clean up
        (temp_clone / "data").rename(data_dir)
        import shutil
        shutil.rmtree(temp_clone)
        print(f"tau2-bench data downloaded to {data_dir}")
    
    # Set the environment variable
    os.environ["TAU2_DATA_DIR"] = str(data_dir)


# Ensure data is available before importing tau2
ensure_tau2_data()

# tau2 imports (installed via: uv sync)
from tau2.run import run_tasks, get_tasks
from tau2.metrics.agent_metrics import compute_metrics


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from other loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)
    
    return logging.getLogger("tau2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tau2-bench for model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run mock domain with GPT-4.1
    python run_tau2.py --domain mock --model gpt-4.1
    
    # Run retail domain with Claude
    python run_tau2.py --domain retail --model claude-3-5-sonnet-20241022
    
    # Run specific tasks
    python run_tau2.py --domain mock --model gpt-4.1 --task-ids create_task_1,update_task_1
    
    # Run with multiple trials for statistical significance
    python run_tau2.py --domain mock --model gpt-4.1 --trials 3
        """
    )
    
    # Domain and model selection
    parser.add_argument(
        "--domain",
        type=str,
        default="mock",
        choices=["mock", "airline", "retail", "telecom"],
        help="Benchmark domain (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="LLM model ID to evaluate (default: gpt-4.1)",
    )
    parser.add_argument(
        "--user-model", 
        type=str,
        default="gpt-4.1",
        help="LLM model for user simulator (default: gpt-4.1)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Custom API base URL for OpenAI-compatible endpoints",
    )
    
    # Task selection
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)",
    )
    
    # Simulation settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per task (default: 200)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent simulations (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
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
    
    # Build LLM args
    llm_args = {"temperature": args.temperature}
    if args.api_base:
        llm_args["api_base"] = args.api_base
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("tau2-bench Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Model: {args.model}")
    logger.info(f"User Model: {args.user_model}")
    if args.api_base:
        logger.info(f"API Base: {args.api_base}")
    logger.info(f"Max Steps: {args.max_steps}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Temperature: {args.temperature}")
    
    # Parse task IDs
    task_ids = None
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",")]
        logger.info(f"Tasks: {task_ids}")
    elif args.limit:
        logger.info(f"Tasks: first {args.limit}")
    else:
        logger.info("Tasks: all")
    
    logger.info("=" * 60)
    
    # Get tasks
    tasks = get_tasks(
        task_set_name=args.domain,
        task_ids=task_ids,
        num_tasks=args.limit,
    )
    
    logger.info(f"Running {len(tasks)} tasks...")
    
    # Run simulations using tau2's native runner
    start_time = datetime.now()
    
    results = run_tasks(
        domain=args.domain,
        tasks=tasks,
        agent="llm_agent",  # Always use tau2's native agent
        user="user_simulator",
        llm_agent=args.model,
        llm_args_agent=llm_args,
        llm_user=args.user_model,
        llm_args_user={"temperature": args.temperature},
        num_trials=args.trials,
        max_steps=args.max_steps,
        max_concurrency=args.concurrency,
        save_to=None,
        console_display=True,
    )
    
    duration = (datetime.now() - start_time).total_seconds()
    
    # Calculate metrics using tau2's metric calculator
    metrics = compute_metrics(results)
    
    # Extract results
    rewards = [sim.reward_info.reward for sim in results.simulations]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    successes = sum(1 for r in rewards if r > 0)
    success_rate = successes / len(rewards) if rewards else 0.0
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Total Tasks: {len(results.simulations)}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Success Rate: {success_rate:.1%}")
    logger.info(f"Average Reward: {avg_reward:.3f}")
    logger.info("=" * 60)
    
    # Per-task results
    logger.info("Per-Task Results:")
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        status = "✓" if reward > 0 else "✗"
        logger.info(f"  {status} {sim.task_id}: reward={reward:.2f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "-")
    output_file = output_dir / f"tau2_{args.domain}_{model_safe}_{timestamp}.json"
    
    config = {
        "domain": args.domain,
        "model": args.model,
        "user_model": args.user_model,
        "max_steps": args.max_steps,
        "trials": args.trials,
        "temperature": args.temperature,
    }
    if args.api_base:
        config["api_base"] = args.api_base
    
    output_data = {
        "config": config,
        "metrics": {
            "duration_seconds": duration,
            "total_tasks": len(results.simulations),
            "successes": successes,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "pass_at_k": metrics.pass_at_k if hasattr(metrics, 'pass_at_k') else None,
        },
        "results": [
            {
                "task_id": sim.task_id,
                "trial": sim.trial,
                "reward": sim.reward_info.reward if sim.reward_info else 0.0,
                "termination_reason": str(sim.termination_reason) if hasattr(sim, 'termination_reason') else None,
                "agent_cost": sim.agent_cost if hasattr(sim, 'agent_cost') else None,
                "user_cost": sim.user_cost if hasattr(sim, 'user_cost') else None,
            }
            for sim in results.simulations
        ],
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_file}")
    
    return 0 if success_rate > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
