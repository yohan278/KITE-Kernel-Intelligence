#!/usr/bin/env python3
"""Quick test script for GDPval benchmark.

Example usage:
    export MODEL=gpt-5-mini-2025-08-07
    export OPENAI_API_KEY=sk-...
    python scripts/run_gdpval.py
"""

import os
import sys
import logging

from pathlib import Path
from agno.models.google import Gemini

from evals.registry import get_benchmark
from agents import React

def main():
    script_dir = Path(__file__).resolve().parent
    evals_dir = script_dir.parent
    agents_dir = evals_dir.parent / "agents"
    
    sys.path.insert(0, str(evals_dir / "src"))
    sys.path.insert(0, str(agents_dir / "src"))
    
    # Get options
    limit = int(os.getenv("LIMIT", "1"))
    shuffle = os.getenv("SHUFFLE", "false").lower() == "true"
    upload_to_hf = os.getenv("UPLOAD_TO_HF", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )
    
    print("Starting GDPval benchmark...")
    if limit:
        print(f"Limit: {limit} samples")
    print()
    
    # Get model and create orchestrator factory
    model_name = "gemini-2.5-flash"
    api_key = os.environ["GOOGLE_API_KEY"]
    
    if not api_key:
        print("Error: API key required. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
        return 1
    
    # Create model
    
    # Create orchestrator factory
    model = Gemini(id=model_name, api_key=api_key)
    orchestrator_factory = lambda tools: React(model=model, tools=tools)
    print(f"Using model: {model_name}")
    
    # Get and run benchmark
    benchmark_factory = get_benchmark("gdpval")
    benchmark = benchmark_factory({
        "upload_to_hf": upload_to_hf,
        "shuffle": shuffle,
        "limit": limit,
    })
    
    metrics = benchmark.run_benchmark(orchestrator_factory)
    
    print()
    print("Evaluation results:", metrics)
    return 0
    

if __name__ == "__main__":
    sys.exit(main() or 0)

