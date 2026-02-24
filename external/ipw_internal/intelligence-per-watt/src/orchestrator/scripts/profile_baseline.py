#!/usr/bin/env python3
"""Profile all MCP tools on sample tasks and cache telemetry.

This script runs all available tools (local models, cloud APIs, basic tools)
on a sample of tasks from ToolScale and caches the telemetry profiles for
fast RL training.

Usage:
    # Profile all tools on 2000 samples
    python scripts/profile_baseline.py --num-samples 2000

    # Profile specific tools only
    python scripts/profile_baseline.py --tools calculator ollama:llama3.2:1b --num-samples 100

    # Use custom cache location
    python scripts/profile_baseline.py --cache-file data/telemetry_cache.db
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.data.toolscale_loader import ToolScaleDataset
from orchestrator.data.telemetry_cache import TelemetryCache, TelemetryProfile, hash_task
from agents.mcp import (
    CalculatorServer,
    OllamaMCPServer,
    OpenAIMCPServer,
    AnthropicMCPServer,
)


def create_mcp_tools(tool_names: Optional[List[str]] = None) -> dict:
    """Create MCP tool servers.

    Args:
        tool_names: List of tool names to create (None = all available)

    Returns:
        Dictionary mapping tool names to server instances
    """
    tools = {}

    # Always include calculator (no dependencies)
    tools["calculator"] = CalculatorServer()

    # Local models (Ollama) - only if requested and Ollama is running
    ollama_models = [
        "llama3.2:1b",
        "llama3.2:3b",
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
    ]

    for model in ollama_models:
        tool_name = f"ollama:{model}"
        if tool_names is None or tool_name in tool_names:
            try:
                server = OllamaMCPServer(model_name=model)
                if server.health_check():
                    tools[tool_name] = server
                    print(f"✓ {tool_name} available")
                else:
                    print(f"✗ {tool_name} not available (health check failed)")
            except Exception as e:
                print(f"✗ {tool_name} not available: {e}")

    # Cloud APIs - only if requested and API keys available
    if os.getenv("OPENAI_API_KEY"):
        for model in ["gpt-5-mini-2025-08-07", "gpt-4o"]:
            tool_name = f"openai:{model}"
            if tool_names is None or tool_name in tool_names:
                try:
                    tools[tool_name] = OpenAIMCPServer(model_name=model)
                    print(f"✓ {tool_name} available")
                except Exception as e:
                    print(f"✗ {tool_name} not available: {e}")
    else:
        print("ℹ OpenAI API key not found (set OPENAI_API_KEY to enable)")

    if os.getenv("ANTHROPIC_API_KEY"):
        for model in ["claude-3-5-haiku-20241022", "claude-sonnet-4-5-20250929"]:
            tool_name = f"anthropic:{model}"
            if tool_names is None or tool_name in tool_names:
                try:
                    tools[tool_name] = AnthropicMCPServer(model_name=model)
                    print(f"✓ {tool_name} available")
                except Exception as e:
                    print(f"✗ {tool_name} not available: {e}")
    else:
        print("ℹ Anthropic API key not found (set ANTHROPIC_API_KEY to enable)")

    return tools


def profile_tool_on_task(tool_server, task_prompt: str) -> Optional[TelemetryProfile]:
    """Profile a single tool on a task.

    Args:
        tool_server: MCP server instance
        task_prompt: Task prompt to execute

    Returns:
        TelemetryProfile with metrics, or None if execution failed
    """
    try:
        result = tool_server.execute(task_prompt)

        # Compute average energy from telemetry samples
        if result.telemetry_samples:
            energy_readings = [
                s.reading.energy_joules
                for s in result.telemetry_samples
                if hasattr(s.reading, "energy_joules") and s.reading.energy_joules
            ]
            avg_energy = sum(energy_readings) / len(energy_readings) if energy_readings else 0.0

            power_readings = [
                s.reading.power_watts
                for s in result.telemetry_samples
                if hasattr(s.reading, "power_watts") and s.reading.power_watts
            ]
            avg_power = sum(power_readings) / len(power_readings) if power_readings else 0.0
        else:
            avg_energy = 0.0
            avg_power = 0.0

        profile = TelemetryProfile(
            tool_name=tool_server.name,
            task_hash=hash_task(task_prompt),
            avg_energy_joules=avg_energy,
            avg_power_watts=avg_power,
            avg_latency_seconds=result.latency_seconds,
            avg_cost_usd=result.cost_usd or 0.0,
            avg_tokens=result.usage.get("total_tokens", 0),
            stddev_energy=avg_energy * 0.1,  # Estimate 10% stddev
            stddev_latency=result.latency_seconds * 0.1,
            num_samples=1,
        )

        return profile

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Profile MCP tools and cache telemetry"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of tasks to profile (default: 100)",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="data/telemetry_cache.db",
        help="Cache file path (default: data/telemetry_cache.db)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="ToolScale split to use (default: train)",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=None,
        help="Specific tools to profile (default: all available)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have cached profiles",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MCP Tool Profiler - Baseline Telemetry Collection")
    print("=" * 70)
    print()

    # Load dataset
    print(f"Loading ToolScale dataset ({args.dataset_split} split, limit={args.num_samples})...")
    try:
        dataset = ToolScaleDataset(
            split=args.dataset_split,
            limit=args.num_samples,
        )
        print(f"✓ Loaded {len(dataset)} tasks")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("\nNote: ToolScale may not be publicly available yet.")
        print("Using synthetic test tasks instead...")

        # Fallback: create synthetic tasks
        class SyntheticSample:
            def __init__(self, task_id, question):
                self.task_id = task_id
                self.question = question

        dataset = [
            SyntheticSample(f"test_{i}", f"What is {i} + {i}?")
            for i in range(args.num_samples)
        ]
        print(f"✓ Created {len(dataset)} synthetic tasks")

    print()

    # Create tools
    print("Initializing MCP tools...")
    tools = create_mcp_tools(args.tools)

    if not tools:
        print("\n❌ No tools available! Please:")
        print("  1. Install and start Ollama: https://ollama.com")
        print("  2. Pull a model: ollama pull llama3.2:1b")
        print("  3. Or set cloud API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY")
        sys.exit(1)

    print(f"\n✓ {len(tools)} tools available: {list(tools.keys())}")
    print()

    # Initialize cache
    cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = TelemetryCache(cache_path)

    # Profile each tool on each task
    total_profiles = len(tools) * len(dataset)
    completed = 0
    skipped = 0
    failed = 0

    print(f"Profiling {len(tools)} tools on {len(dataset)} tasks ({total_profiles} total profiles)...")
    print("=" * 70)

    start_time = time.time()

    for tool_name, tool_server in tools.items():
        print(f"\n{tool_name}:")
        print("-" * 70)

        for i, sample in enumerate(dataset):
            task_prompt = sample.question if hasattr(sample, "question") else str(sample)

            # Check if already cached
            if args.skip_existing:
                task_hash = hash_task(task_prompt)
                if cache.get_profile(tool_name, task_hash):
                    skipped += 1
                    continue

            # Profile tool
            print(f"  [{i+1}/{len(dataset)}] {task_prompt[:50]}...", end=" ")
            profile = profile_tool_on_task(tool_server, task_prompt)

            if profile:
                cache.save_profile(profile)
                completed += 1
                print(f"✓ ({profile.avg_latency_seconds:.2f}s, ${profile.avg_cost_usd:.4f})")
            else:
                failed += 1

            # Rate limiting for cloud APIs
            if "openai" in tool_name or "anthropic" in tool_name:
                time.sleep(0.5)  # Avoid rate limits

    elapsed = time.time() - start_time

    # Show results
    print()
    print("=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
    print(f"Total profiles: {completed}")
    print(f"Skipped (cached): {skipped}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print()

    # Show cache statistics
    stats = cache.export_statistics()
    print("Cache Statistics:")
    print("-" * 70)
    print(f"Total profiles: {stats['total_profiles']}")
    print(f"Number of tools: {stats['num_tools']}")
    print(f"Avg energy: {stats['avg_energy_joules']:.2f} J")
    print(f"Avg latency: {stats['avg_latency_seconds']:.2f} s")
    print(f"Avg cost: ${stats['avg_cost_usd']:.4f}")
    print()
    print("Coverage by tool:")
    for tool, count in stats['coverage_by_tool'].items():
        print(f"  {tool}: {count} profiles")

    print()
    print(f"Cache saved to: {cache_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
