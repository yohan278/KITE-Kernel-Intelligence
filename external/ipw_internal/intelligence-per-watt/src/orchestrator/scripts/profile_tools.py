#!/usr/bin/env python3
"""Profile tools and populate telemetry cache for training.

This script executes each available tool with sample tasks from the dataset
and records the energy, cost, latency, and forward pass metrics to the
telemetry cache for fast offline training.

Usage:
    # Profile all tools with 100 samples each
    python scripts/profile_tools.py --num-samples 100

    # Profile specific tools
    python scripts/profile_tools.py --tools calculator,ollama:qwen2.5:1.5b --num-samples 50

    # Profile and show cache statistics
    python scripts/profile_tools.py --num-samples 20 --show-stats
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.data.telemetry_cache import TelemetryCache, TelemetryProfile, hash_task
from orchestrator.data.toolscale_loader import ToolScaleDataset


def get_tool_specific_prompts(tool_name: str, num_samples: int) -> List[str]:
    """Generate tool-specific test prompts.

    Args:
        tool_name: Name of the tool
        num_samples: Number of prompts to generate

    Returns:
        List of prompts appropriate for the tool
    """
    if tool_name == "calculator":
        # Calculator needs math expressions
        base_prompts = [
            "2 + 2",
            "15 * 7",
            "calculate 100 / 4",
            "what is 3 ** 4",
            "sqrt(144)",
            "sin(0)",
            "25 + 30 - 10",
            "calculate 1000 * 1.05 ** 10",
            "abs(-42)",
            "round(3.14159)",
        ]
    elif tool_name == "think":
        base_prompts = [
            "Let me think about this step by step...",
            "First, I need to consider the problem.",
            "Breaking down the task: 1) understand, 2) plan, 3) execute",
            "The key insight here is...",
            "Let me reason through this carefully.",
        ]
    elif tool_name == "code_interpreter":
        base_prompts = [
            "print('Hello, World!')",
            "print([x**2 for x in range(5)])",
            "import math; print(math.pi)",
            "print(sum(range(100)))",
            "print('Result:', 2 + 2)",
        ]
    else:
        # Default prompts for LLM-based tools
        base_prompts = [
            "What is 2 + 2?",
            "Calculate 15 * 7",
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
            "What is the square root of 144?",
        ]

    # Repeat and trim to desired length
    prompts = (base_prompts * (num_samples // len(base_prompts) + 1))[:num_samples]
    return prompts


def create_mcp_server(tool_name: str) -> Optional[Any]:
    """Create MCP server instance for the given tool.

    Args:
        tool_name: Name of the tool to create

    Returns:
        MCP server instance or None if creation fails
    """
    try:
        if tool_name == "calculator":
            from agents.mcp.tool_server import CalculatorServer
            return CalculatorServer()

        elif tool_name == "think":
            from agents.mcp.tool_server import ThinkServer
            return ThinkServer()

        elif tool_name == "code_interpreter":
            from agents.mcp.tool_server import CodeInterpreterServer
            return CodeInterpreterServer()

        elif tool_name == "web_search":
            from agents.mcp.tool_server import WebSearchServer
            return WebSearchServer()

        elif tool_name.startswith("ollama:"):
            from agents.mcp.ollama_server import OllamaMCPServer
            model_name = tool_name.split(":", 1)[1]
            return OllamaMCPServer(model_name=model_name)

        elif tool_name.startswith("openai:"):
            from agents.mcp.openai_server import OpenAIMCPServer
            model_name = tool_name.split(":", 1)[1]
            return OpenAIMCPServer(model_name=model_name)

        elif tool_name.startswith("anthropic:"):
            from agents.mcp.anthropic_server import AnthropicMCPServer
            model_name = tool_name.split(":", 1)[1]
            return AnthropicMCPServer(model_name=model_name)

        elif tool_name.startswith("vllm:"):
            from agents.mcp.vllm_server import VLLMMCPServer
            model_name = tool_name.split(":", 1)[1]
            return VLLMMCPServer(model_name=model_name)

        else:
            print(f"Warning: Unknown tool type '{tool_name}'")
            return None

    except ImportError as e:
        print(f"Warning: Could not import MCP server for '{tool_name}': {e}")
        return None
    except Exception as e:
        print(f"Warning: Could not create '{tool_name}': {e}")
        return None


def extract_telemetry_from_result(result, tool_name: str, task_hash: str) -> TelemetryProfile:
    """Extract telemetry profile from MCP tool result.

    Args:
        result: MCPToolResult from tool execution
        tool_name: Name of the tool
        task_hash: Hash of the task prompt

    Returns:
        TelemetryProfile with extracted metrics
    """
    # Extract energy from telemetry samples
    avg_energy = 0.0
    avg_power = 0.0

    if hasattr(result, 'telemetry_samples') and result.telemetry_samples:
        energy_readings = []
        power_readings = []

        for sample in result.telemetry_samples:
            if hasattr(sample, 'reading'):
                if hasattr(sample.reading, 'energy_joules') and sample.reading.energy_joules:
                    energy_readings.append(sample.reading.energy_joules)
                if hasattr(sample.reading, 'power_watts') and sample.reading.power_watts:
                    power_readings.append(sample.reading.power_watts)

        if energy_readings:
            avg_energy = sum(energy_readings) / len(energy_readings)
        if power_readings:
            avg_power = sum(power_readings) / len(power_readings)

    # Get cost
    cost_usd = getattr(result, 'cost_usd', 0.0) or 0.0

    # Get latency
    latency_seconds = getattr(result, 'latency_seconds', 0.0) or 0.0

    # Get tokens
    usage = getattr(result, 'usage', {}) or {}
    total_tokens = usage.get('total_tokens', 0) if isinstance(usage, dict) else 0

    # Determine forward passes (most tools = 1, multi-turn = more)
    num_forward_passes = 1
    if hasattr(result, 'metadata') and result.metadata:
        num_forward_passes = result.metadata.get('forward_passes', 1)

    return TelemetryProfile(
        tool_name=tool_name,
        task_hash=task_hash,
        avg_energy_joules=avg_energy,
        avg_power_watts=avg_power,
        avg_latency_seconds=latency_seconds,
        avg_cost_usd=cost_usd,
        avg_tokens=total_tokens,
        num_forward_passes=num_forward_passes,
        stddev_energy=0.1 * avg_energy if avg_energy > 0 else 0.0,
        stddev_latency=0.1 * latency_seconds if latency_seconds > 0 else 0.0,
    )


def profile_tool(
    tool_name: str,
    server: Any,
    prompts: List[str],
    cache: TelemetryCache,
    verbose: bool = False,
) -> Dict[str, float]:
    """Profile a single tool with multiple prompts.

    Args:
        tool_name: Name of the tool
        server: MCP server instance
        prompts: List of prompts to execute
        cache: Telemetry cache to save profiles
        verbose: Print progress for each prompt

    Returns:
        Dictionary with profiling statistics
    """
    stats = {
        "total_samples": 0,
        "successful": 0,
        "failed": 0,
        "total_energy_joules": 0.0,
        "total_cost_usd": 0.0,
        "total_latency_seconds": 0.0,
    }

    for i, prompt in enumerate(prompts):
        try:
            # Execute tool
            start_time = time.time()
            result = server.execute(prompt)
            elapsed = time.time() - start_time

            # Extract telemetry
            task_hash = hash_task(prompt)
            profile = extract_telemetry_from_result(result, tool_name, task_hash)

            # Use actual elapsed time if telemetry didn't capture it
            if profile.avg_latency_seconds == 0:
                profile = TelemetryProfile(
                    tool_name=profile.tool_name,
                    task_hash=profile.task_hash,
                    avg_energy_joules=profile.avg_energy_joules,
                    avg_power_watts=profile.avg_power_watts,
                    avg_latency_seconds=elapsed,
                    avg_cost_usd=profile.avg_cost_usd,
                    avg_tokens=profile.avg_tokens,
                    num_forward_passes=profile.num_forward_passes,
                    stddev_energy=profile.stddev_energy,
                    stddev_latency=0.1 * elapsed,
                )

            # Save to cache
            cache.save_profile(profile)

            # Update stats
            stats["successful"] += 1
            stats["total_energy_joules"] += profile.avg_energy_joules
            stats["total_cost_usd"] += profile.avg_cost_usd
            stats["total_latency_seconds"] += profile.avg_latency_seconds

            if verbose:
                print(f"    [{i+1}/{len(prompts)}] {elapsed:.2f}s, {profile.avg_energy_joules:.2f}J")

        except Exception as e:
            stats["failed"] += 1
            if verbose:
                import traceback
                print(f"    [{i+1}/{len(prompts)}] FAILED: {e}")
                traceback.print_exc()

        stats["total_samples"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Profile tools and populate telemetry cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tools",
        type=str,
        default="calculator,ollama:qwen2.5:1.5b,ollama:llama3.2:1b",
        help="Comma-separated list of tools to profile",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to profile per tool",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nvidia/ToolScale",
        help="Dataset to sample prompts from",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/telemetry_cache.db",
        help="Path to telemetry cache database",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show cache statistics after profiling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (print each sample)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before profiling",
    )

    args = parser.parse_args()

    # Parse tools
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]

    print("=" * 70)
    print("Tool Profiling for Telemetry Cache")
    print("=" * 70)
    print(f"Tools to profile: {len(tools)}")
    for tool in tools:
        print(f"  - {tool}")
    print(f"Samples per tool: {args.num_samples}")
    print(f"Dataset: {args.dataset}")
    print(f"Cache path: {args.cache_path}")
    print("=" * 70)
    print()

    # Initialize cache
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = TelemetryCache(cache_path)

    if args.clear_cache:
        print("Clearing existing cache...")
        cache.clear()

    # Load dataset
    print("Loading dataset...")
    try:
        dataset = ToolScaleDataset(
            split="train",
            limit=args.num_samples * 2,  # Load extra for filtering
            dataset_path=args.dataset,
        )
        prompts = [sample.question for sample in dataset][:args.num_samples]
        print(f"  Loaded {len(prompts)} sample prompts")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback test prompts...")
        prompts = [
            "What is 2 + 2?",
            "Calculate 15 * 7",
            "What is the capital of France?",
            "Explain photosynthesis",
            "Write a Python function to sort a list",
        ] * (args.num_samples // 5 + 1)
        prompts = prompts[:args.num_samples]

    print()

    # Profile each tool
    all_stats = {}
    for tool_name in tools:
        print(f"Profiling: {tool_name}")

        # Create server
        server = create_mcp_server(tool_name)
        if server is None:
            print(f"  SKIPPED: Could not create server")
            continue

        # Use tool-specific prompts for utility tools
        if tool_name in ("calculator", "think", "code_interpreter"):
            tool_prompts = get_tool_specific_prompts(tool_name, args.num_samples)
            print(f"  Using {len(tool_prompts)} tool-specific prompts")
        else:
            tool_prompts = prompts

        # Profile
        stats = profile_tool(
            tool_name=tool_name,
            server=server,
            prompts=tool_prompts,
            cache=cache,
            verbose=args.verbose,
        )
        all_stats[tool_name] = stats

        # Print summary
        success_rate = stats["successful"] / stats["total_samples"] * 100 if stats["total_samples"] > 0 else 0
        avg_energy = stats["total_energy_joules"] / stats["successful"] if stats["successful"] > 0 else 0
        avg_latency = stats["total_latency_seconds"] / stats["successful"] if stats["successful"] > 0 else 0

        print(f"  Completed: {stats['successful']}/{stats['total_samples']} ({success_rate:.0f}% success)")
        print(f"  Avg energy: {avg_energy:.2f}J, Avg latency: {avg_latency:.2f}s")
        print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
        print()

    # Show cache statistics
    if args.show_stats:
        print("=" * 70)
        print("Cache Statistics")
        print("=" * 70)
        stats = cache.export_statistics()
        print(f"Total profiles: {stats['total_profiles']}")
        print(f"Number of tools: {stats['num_tools']}")
        print(f"Average energy: {stats['avg_energy_joules']:.2f}J")
        print(f"Average latency: {stats['avg_latency_seconds']:.2f}s")
        print(f"Average cost: ${stats['avg_cost_usd']:.4f}")
        print()
        print("Coverage by tool:")
        for tool, count in stats['coverage_by_tool'].items():
            print(f"  {tool}: {count} profiles")
        print("=" * 70)

    print("\nProfiling complete!")
    print(f"Cache saved to: {args.cache_path}")


if __name__ == "__main__":
    main()
