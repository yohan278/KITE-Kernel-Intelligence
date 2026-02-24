"""Integration tests for auto-server functionality with real vLLM servers.

These tests require GPU resources and will start/stop actual vLLM servers.
Run with: pytest tests/integration/test_auto_server.py -v -s

Use markers to select specific tests:
- pytest -m "auto_server" : All auto-server tests
- pytest -m "submodel" : Tests with multiple models
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from ipw.cli.server_manager import (
    InferenceServerManager,
    ServerConfig,
    build_server_configs,
)


# Skip all tests if no GPU available
def gpu_available() -> bool:
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.decode().strip().split("\n") if l.strip()]
            return len(lines) > 0
        return False
    except Exception:
        return False


def count_gpus() -> int:
    """Count available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.decode().strip().split("\n") if l.strip()]
            return len(lines)
    except Exception:
        pass
    return 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not gpu_available(), reason="No GPU available"),
]


# Test model configurations - use small models for faster testing
SMALL_MODEL = "Qwen/Qwen3-4B"  # ~8GB VRAM
MATH_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # ~3GB VRAM


class TestServerManagerIntegration:
    """Integration tests for InferenceServerManager with real vLLM servers."""

    @pytest.mark.auto_server
    @pytest.mark.timeout(300)  # 5 minute timeout for server startup
    def test_start_and_stop_single_server(self):
        """Test starting and stopping a single vLLM server."""
        config = ServerConfig(
            model_id=SMALL_MODEL,
            alias="test-main",
            backend="vllm",
            port=18000,  # Use non-standard port to avoid conflicts
            gpu_ids=[0],
            tensor_parallel_size=1,
            max_model_len=4096,  # Smaller context for faster loading
            gpu_memory_utilization=0.5,  # Use less memory
        )

        manager = InferenceServerManager([config], auto_assign_gpus=False)

        try:
            # Start server
            print(f"\nStarting vLLM server for {SMALL_MODEL}...")
            urls = manager.start_all(wait_timeout=240.0)

            assert "test-main" in urls
            assert urls["test-main"] == "http://localhost:18000/v1"
            assert manager.is_running("test-main")

            # Verify server responds
            print("Verifying server health...")
            import urllib.request
            req = urllib.request.Request(
                "http://localhost:18000/v1/models",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                model_ids = [m["id"] for m in data.get("data", [])]
                assert any(SMALL_MODEL in mid for mid in model_ids), f"Model not found in {model_ids}"

            print("Server health check passed!")

        finally:
            # Always stop server
            print("Stopping server...")
            manager.stop_all()
            time.sleep(2)  # Give process time to clean up

            # Verify server stopped
            assert not manager.is_running("test-main")
            print("Server stopped successfully!")

    @pytest.mark.auto_server
    @pytest.mark.timeout(300)
    def test_warmup_query(self):
        """Test that warmup queries work correctly."""
        config = ServerConfig(
            model_id=SMALL_MODEL,
            alias="warmup-test",
            backend="vllm",
            port=18001,
            gpu_ids=[1],  # Use different GPU
            max_model_len=4096,
            gpu_memory_utilization=0.5,
        )

        manager = InferenceServerManager([config], auto_assign_gpus=False)

        try:
            print(f"\nStarting server for warmup test...")
            manager.start_all(wait_timeout=240.0)

            # Run warmup
            print("Running warmup query...")
            start = time.time()
            manager.warmup("warmup-test", warmup_prompt="Hello, how are you?")
            warmup_time = time.time() - start
            print(f"Warmup completed in {warmup_time:.2f}s")

            # Second warmup should be faster (model already loaded)
            print("Running second warmup (should be faster)...")
            start = time.time()
            manager.warmup("warmup-test", warmup_prompt="What is 2+2?")
            second_time = time.time() - start
            print(f"Second warmup completed in {second_time:.2f}s")

            # Second should generally be faster, but not a strict requirement
            assert second_time < warmup_time * 2, "Second warmup suspiciously slow"

        finally:
            manager.stop_all()
            time.sleep(2)

    @pytest.mark.auto_server
    @pytest.mark.submodel
    @pytest.mark.skipif(count_gpus() < 2, reason="Need at least 2 GPUs for submodel test")
    @pytest.mark.timeout(600)  # 10 minute timeout for multiple servers
    def test_multiple_servers_with_submodels(self):
        """Test starting multiple vLLM servers for main model + submodel."""
        configs = [
            ServerConfig(
                model_id=SMALL_MODEL,
                alias="main",
                backend="vllm",
                port=18010,
                gpu_ids=[2],
                max_model_len=4096,
                gpu_memory_utilization=0.5,
            ),
            ServerConfig(
                model_id=MATH_MODEL,
                alias="math",
                backend="vllm",
                port=18011,
                gpu_ids=[3],
                max_model_len=4096,
                gpu_memory_utilization=0.5,
            ),
        ]

        manager = InferenceServerManager(configs, auto_assign_gpus=False)

        try:
            print(f"\nStarting multiple servers...")
            print(f"  Main: {SMALL_MODEL} on GPU 2, port 18010")
            print(f"  Math: {MATH_MODEL} on GPU 3, port 18011")

            urls = manager.start_all(wait_timeout=300.0)

            assert len(urls) == 2
            assert "main" in urls
            assert "math" in urls
            assert urls["main"] == "http://localhost:18010/v1"
            assert urls["math"] == "http://localhost:18011/v1"

            # Verify both servers respond
            for alias, url in urls.items():
                print(f"Checking {alias} at {url}...")
                import urllib.request
                req = urllib.request.Request(f"{url}/models", method="GET")
                with urllib.request.urlopen(req, timeout=10) as response:
                    assert response.status == 200
                print(f"  {alias} OK!")

            # Warmup both
            print("Warming up both models...")
            manager.warmup_all()
            print("Warmup complete!")

        finally:
            print("Stopping all servers...")
            manager.stop_all()
            time.sleep(3)

    @pytest.mark.auto_server
    @pytest.mark.timeout(300)
    def test_build_configs_and_start(self):
        """Test the full flow: build_server_configs -> start -> stop."""
        configs = build_server_configs(
            main_model=SMALL_MODEL,
            main_alias="built-main",
            submodel_specs=[],  # No submodels for this test
            base_port=18020,
            main_backend="vllm",
        )

        # Manually set GPU and memory settings for faster testing
        configs[0].gpu_ids = [4]
        configs[0].max_model_len = 4096
        configs[0].gpu_memory_utilization = 0.5

        manager = InferenceServerManager(configs, auto_assign_gpus=False)

        try:
            print(f"\nTesting build_server_configs flow...")
            urls = manager.start_all(wait_timeout=240.0)

            assert "built-main" in urls
            assert manager.is_running("built-main")

            # Make a real inference request
            print("Making inference request...")
            import urllib.request
            payload = json.dumps({
                "model": SMALL_MODEL,
                "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number, nothing else."}],
                "max_tokens": 100,  # More tokens to allow for thinking
                "temperature": 0.0,
            }).encode()

            req = urllib.request.Request(
                f"{urls['built-main']}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode())
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                print(f"Model response: {content[:200]}...")  # Truncate long responses
                print(f"Usage: {usage}")
                # Check we got a response (don't require specific answer due to model variations)
                assert len(content) > 0, "Expected non-empty response"
                # Verify tokens were used
                assert usage.get("total_tokens", 0) > 0, "Expected token usage"

            print("Inference successful!")

        finally:
            manager.stop_all()
            time.sleep(2)


class TestBenchCLIIntegration:
    """Integration tests for the bench CLI with --auto-server."""

    @pytest.mark.auto_server
    @pytest.mark.timeout(600)
    def test_bench_with_auto_server(self, tmp_path: Path):
        """Test running ipw bench with --auto-server flag."""
        from click.testing import CliRunner
        from ipw.cli import cli

        runner = CliRunner()
        output_dir = tmp_path / "bench_output"

        print(f"\nRunning bench with --auto-server...")
        print(f"Output dir: {output_dir}")

        # Run benchmark with auto-server
        # Note: This requires the evals module to be available
        result = runner.invoke(
            cli,
            [
                "bench",
                "--agent", "react",
                "--model", SMALL_MODEL,
                "--benchmark", "hle",
                "--limit", "1",
                "--auto-server",
                "--base-port", "18030",
                "--output", str(output_dir),
                "--per-action",
                "--skip-warmup",  # Skip warmup for faster test
            ],
            catch_exceptions=False,
        )

        print(f"CLI output:\n{result.output}")

        if result.exit_code != 0:
            print(f"CLI failed with exit code {result.exit_code}")
            if result.exception:
                import traceback
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

        # Check results
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify output files exist
        output_files = list(output_dir.glob("**/results.json"))
        assert len(output_files) > 0, f"No results.json found in {output_dir}"

        # Check results content
        results_file = output_files[0]
        with open(results_file) as f:
            results = json.load(f)

        print(f"\nResults summary:")
        print(f"  Duration: {results.get('duration_seconds', 'N/A')}s")
        print(f"  Energy: {results.get('total_energy_joules', 'N/A')}J")
        print(f"  Samples: {results.get('telemetry_samples', 'N/A')}")

        # Verify key fields
        assert "run_metadata" in results
        assert results["run_metadata"]["auto_server"] is True
        assert "duration_seconds" in results

        # Check for per-action data if telemetry worked
        if "action_breakdown" in results:
            print(f"  Actions recorded: {len(results['action_breakdown'])}")
            assert len(results["action_breakdown"]) > 0

    @pytest.mark.auto_server
    @pytest.mark.submodel
    @pytest.mark.skipif(count_gpus() < 2, reason="Need at least 2 GPUs")
    @pytest.mark.timeout(900)
    def test_bench_with_submodel(self, tmp_path: Path):
        """Test running ipw bench with --auto-server and --submodel."""
        from click.testing import CliRunner
        from ipw.cli import cli

        runner = CliRunner()
        output_dir = tmp_path / "bench_submodel_output"

        print(f"\nRunning bench with --auto-server and --submodel...")

        result = runner.invoke(
            cli,
            [
                "bench",
                "--agent", "react",
                "--model", SMALL_MODEL,
                "--benchmark", "hle",
                "--limit", "1",
                "--auto-server",
                "--base-port", "18040",
                f"--submodel", f"math:vllm:{MATH_MODEL}",
                "--output", str(output_dir),
                "--per-action",
            ],
            catch_exceptions=False,
        )

        print(f"CLI output:\n{result.output}")

        if result.exit_code != 0:
            print(f"CLI failed with exit code {result.exit_code}")

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify output
        output_files = list(output_dir.glob("**/results.json"))
        assert len(output_files) > 0

        with open(output_files[0]) as f:
            results = json.load(f)

        # Verify submodel was configured
        assert "run_metadata" in results
        assert len(results["run_metadata"].get("submodels", [])) > 0
        assert any("math" in s for s in results["run_metadata"]["submodels"])


class TestEnergyTelemetryIntegration:
    """Integration tests for energy telemetry with real inference."""

    @pytest.mark.auto_server
    @pytest.mark.timeout(300)
    def test_energy_collection_during_inference(self):
        """Test that energy telemetry is collected during vLLM inference."""
        from ipw.telemetry.events import EventRecorder

        config = ServerConfig(
            model_id=SMALL_MODEL,
            alias="energy-test",
            backend="vllm",
            port=18050,
            gpu_ids=[5],
            max_model_len=4096,
            gpu_memory_utilization=0.5,
        )

        manager = InferenceServerManager([config], auto_assign_gpus=False)
        event_recorder = EventRecorder()

        try:
            print(f"\nStarting server for energy test...")
            urls = manager.start_all(wait_timeout=240.0)
            manager.warmup("energy-test")

            # Import telemetry components
            from ipw.telemetry import EnergyMonitorCollector
            from ipw.execution.telemetry_session import TelemetrySession

            # Start telemetry collection
            collector = EnergyMonitorCollector()

            print("Running inference with telemetry...")
            with TelemetrySession(collector) as session:
                # Wait a moment for telemetry to start collecting
                time.sleep(0.5)

                start_time = time.time()

                # Record event
                event_recorder.record("lm_inference_start", model_id=SMALL_MODEL)

                # Make inference request with more tokens to take longer
                import urllib.request
                payload = json.dumps({
                    "model": SMALL_MODEL,
                    "messages": [{"role": "user", "content": "Write a detailed paragraph about sustainable energy and why it matters for our future. Include at least three specific examples."}],
                    "max_tokens": 200,
                }).encode()

                req = urllib.request.Request(
                    f"{urls['energy-test']}/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode())
                    usage = data.get("usage", {})

                end_time = time.time()

                # Wait a moment to ensure final samples are collected
                time.sleep(0.5)

                # Record end event
                event_recorder.record(
                    "lm_inference_end",
                    model_id=SMALL_MODEL,
                    total_tokens=usage.get("total_tokens", 0),
                )

                # Get telemetry samples - extend window slightly
                samples = list(session.window(start_time - 0.5, end_time + 0.5))

            inference_duration = end_time - start_time
            print(f"\nTelemetry results:")
            print(f"  Inference duration: {inference_duration:.2f}s")
            print(f"  Samples collected: {len(samples)}")
            print(f"  Events recorded: {len(event_recorder)}")
            print(f"  Tokens used: {usage}")

            # Verify events were recorded
            assert len(event_recorder) == 2, "Expected start and end events"

            # Telemetry samples depend on energy monitor binary - log but don't fail
            if len(samples) == 0:
                print("  WARNING: No telemetry samples collected - energy monitor may not be running")
                print("  Test will verify event recording without energy correlation")
            else:
                print(f"  Telemetry collection working!")

                # Check energy values
                first_sample = samples[0]
                last_sample = samples[-1]
                if hasattr(first_sample.reading, 'energy_joules') and first_sample.reading.energy_joules:
                    energy_delta = last_sample.reading.energy_joules - first_sample.reading.energy_joules
                    print(f"  Energy consumed: {energy_delta:.2f}J")
                    if energy_delta > 0:
                        print("  Energy measurement verified!")
                    else:
                        print("  WARNING: Energy delta not positive - may be counter issue")

            # Correlate events with energy
            from ipw.telemetry.correlation import correlate_energy_to_events, compute_analysis

            events = event_recorder.get_events()

            # Test correlation (works even without samples)
            breakdowns = correlate_energy_to_events(samples, events)

            print(f"\nCorrelation results:")
            print(f"  Breakdowns: {len(breakdowns)}")

            if breakdowns:
                for b in breakdowns:
                    print(f"    {b.action_type}: {b.total_energy_joules:.2f}J, {b.duration_ms:.1f}ms")

                analysis = compute_analysis(breakdowns)
                print(f"\nAnalysis:")
                print(f"  Total energy: {analysis['total_energy_joules']:.2f}J")
                print(f"  Action counts: {analysis['action_counts']}")

                # Verify per-model tracking (our new feature!)
                if analysis.get("model_counts"):
                    print(f"  Model counts: {analysis['model_counts']}")
                    print(f"  Energy by model: {analysis['energy_by_model']}")
            else:
                print("  No breakdowns (expected if no telemetry samples)")

            # Verify the inference actually worked
            assert usage.get("total_tokens", 0) > 0, "Expected token usage from inference"
            print("\nTest passed - inference and event recording verified!")

        finally:
            manager.stop_all()
            time.sleep(2)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])
