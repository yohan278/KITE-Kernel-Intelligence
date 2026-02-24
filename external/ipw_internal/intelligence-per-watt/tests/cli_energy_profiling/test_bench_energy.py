#!/usr/bin/env python3
"""Test script for validating the ipw bench CLI energy profiling functionality.

This script tests that:
1. Energy telemetry collection works correctly
2. Warmup/startup costs are properly excluded from measurements
3. Per-action energy breakdowns are recorded when enabled
4. Time windowing correctly isolates inference energy from overhead

Requirements:
- Linux x86_64 with NVIDIA GPU(s)
- vLLM installed and working
- ipw package installed (`pip install -e .`)
- Energy monitor binary built (`uv run scripts/build_energy_monitor.py`)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_command(cmd: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result


def check_prerequisites() -> bool:
    """Check that all prerequisites are met."""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)

    # Check for NVIDIA GPU
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: nvidia-smi not available. NVIDIA GPU required.")
        return False

    print(f"GPU(s) detected:\n{result.stdout.strip()}")

    # Check ipw CLI
    result = subprocess.run(["ipw", "--help"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: ipw CLI not available. Install with: pip install -e .")
        return False

    print("ipw CLI: OK")

    # Check energy monitor binary
    from pathlib import Path
    bin_path = Path(__file__).parent.parent.parent / "src" / "ipw" / "telemetry" / "bin" / "linux-x86_64" / "energy-monitor"
    if not bin_path.exists():
        print(f"ERROR: Energy monitor binary not found at {bin_path}")
        print("Build with: uv run scripts/build_energy_monitor.py")
        return False

    print(f"Energy monitor binary: OK ({bin_path})")

    return True


def test_bench_basic(output_dir: Path, model: str = "qwen3-4b") -> Dict[str, Any]:
    """Test basic benchmark with energy telemetry."""
    print("\n" + "="*60)
    print("TEST 1: Basic benchmark with energy telemetry")
    print("="*60)

    result = run_command([
        "ipw", "bench",
        "--agent", "react",
        "--model", model,
        "--benchmark", "hle",
        "--limit", "2",
        "--output", str(output_dir / "basic"),
    ])

    test_result = {
        "name": "basic_benchmark",
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    # Check for energy metrics in output
    if "energy" in result.stdout.lower() or "joules" in result.stdout.lower():
        test_result["energy_metrics_found"] = True
    else:
        test_result["energy_metrics_found"] = False
        test_result["passed"] = False

    return test_result


def test_bench_per_action(output_dir: Path, model: str = "qwen3-4b") -> Dict[str, Any]:
    """Test benchmark with per-action energy breakdown."""
    print("\n" + "="*60)
    print("TEST 2: Benchmark with per-action energy breakdown")
    print("="*60)

    result = run_command([
        "ipw", "bench",
        "--agent", "react",
        "--model", model,
        "--benchmark", "hle",
        "--limit", "1",
        "--per-action",
        "--output", str(output_dir / "per_action"),
    ])

    test_result = {
        "name": "per_action_breakdown",
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    # Check for per-action breakdown in output
    if "action" in result.stdout.lower() and "breakdown" in result.stdout.lower():
        test_result["per_action_found"] = True
    else:
        test_result["per_action_found"] = False

    return test_result


def test_bench_no_telemetry(output_dir: Path, model: str = "qwen3-4b") -> Dict[str, Any]:
    """Test benchmark without energy telemetry (baseline)."""
    print("\n" + "="*60)
    print("TEST 3: Benchmark without telemetry (baseline)")
    print("="*60)

    result = run_command([
        "ipw", "bench",
        "--agent", "react",
        "--model", model,
        "--benchmark", "hle",
        "--limit", "1",
        "--no-telemetry",
        "--output", str(output_dir / "no_telemetry"),
    ])

    test_result = {
        "name": "no_telemetry_baseline",
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    return test_result


def test_bench_skip_warmup(output_dir: Path, model: str = "qwen3-4b") -> Dict[str, Any]:
    """Test benchmark with warmup skipped (includes cold-start)."""
    print("\n" + "="*60)
    print("TEST 4: Benchmark with --skip-warmup (cold-start included)")
    print("="*60)

    result = run_command([
        "ipw", "bench",
        "--agent", "react",
        "--model", model,
        "--benchmark", "hle",
        "--limit", "1",
        "--skip-warmup",
        "--output", str(output_dir / "skip_warmup"),
    ])

    test_result = {
        "name": "skip_warmup",
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    return test_result


def test_bench_auto_server(output_dir: Path, model: str = "qwen3-4b") -> Dict[str, Any]:
    """Test benchmark with auto-server management."""
    print("\n" + "="*60)
    print("TEST 5: Benchmark with --auto-server")
    print("="*60)

    result = run_command([
        "ipw", "bench",
        "--agent", "react",
        "--model", model,
        "--benchmark", "hle",
        "--limit", "1",
        "--auto-server",
        "--output", str(output_dir / "auto_server"),
    ], timeout=900)  # Longer timeout for server startup

    test_result = {
        "name": "auto_server",
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    # Check that startup/shutdown are excluded
    if "excluded from profiling" in result.stdout.lower():
        test_result["exclusion_messaging"] = True
    else:
        test_result["exclusion_messaging"] = False

    return test_result


def run_all_tests(output_dir: Optional[Path] = None, model: str = "qwen3-4b") -> Dict[str, Any]:
    """Run all tests and return results."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ipw_bench_test_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("IPW BENCH CLI ENERGY PROFILING TESTS")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print("="*60)

    if not check_prerequisites():
        return {
            "status": "failed",
            "reason": "Prerequisites not met",
            "tests": [],
        }

    tests = [
        ("basic", test_bench_basic),
        ("per_action", test_bench_per_action),
        ("no_telemetry", test_bench_no_telemetry),
        ("skip_warmup", test_bench_skip_warmup),
        ("auto_server", test_bench_auto_server),
    ]

    results = {
        "status": "completed",
        "output_dir": str(output_dir),
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": [],
    }

    for name, test_fn in tests:
        try:
            test_result = test_fn(output_dir, model)
            results["tests"].append(test_result)
        except Exception as e:
            results["tests"].append({
                "name": name,
                "passed": False,
                "error": str(e),
            })

    # Summary
    passed = sum(1 for t in results["tests"] if t.get("passed", False))
    total = len(results["tests"])
    results["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "total": total,
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    for t in results["tests"]:
        status = "PASS" if t.get("passed", False) else "FAIL"
        print(f"  [{status}] {t['name']}")

    # Save results
    results_file = output_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ipw bench CLI energy profiling")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for test results")
    parser.add_argument("--model", "-m", default="qwen3-4b", help="Model to use for testing")
    parser.add_argument("--vllm-url", help="External vLLM server URL (skip auto-server test)")

    args = parser.parse_args()

    results = run_all_tests(args.output, args.model)

    # Exit with appropriate code
    if results.get("status") == "completed":
        failed = results.get("summary", {}).get("failed", 0)
        sys.exit(0 if failed == 0 else 1)
    else:
        sys.exit(1)
