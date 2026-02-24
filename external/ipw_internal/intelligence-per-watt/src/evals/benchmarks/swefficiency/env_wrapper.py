# benchmarks/swefficiency/env_wrapper.py
"""
SWEfficiency environment wrapper using Docker directly.

Provides a simplified interface for container communication for performance
optimization evaluation. Similar to SWE-bench but with custom image names
per sample and performance benchmarking capabilities.
"""
from __future__ import annotations

import base64
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import docker
from docker.models.containers import Container

if TYPE_CHECKING:
    from .dataset import SWEfficiencySample

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from running a performance benchmark."""

    passed: bool
    execution_time: float  # seconds
    output: str
    error: Optional[str] = None


@dataclass
class TestResult:
    """Result from running tests."""

    passed: int
    failed: int
    total: int
    output: str
    error: Optional[str] = None

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.passed > 0


class SWEfficiencyEnv:
    """
    Environment wrapper for SWEfficiency evaluation using Docker.

    Manages communication with a Docker container for performance optimization
    evaluation. Supports applying patches, running builds, executing tests,
    and measuring performance.

    Usage:
        env = SWEfficiencyEnv.from_sample(sample)
        env.start()

        # Apply patch and rebuild
        env.apply_patch(patch_content)
        env.rebuild()

        # Run tests
        test_result = env.run_tests()

        # Measure performance
        baseline = env.measure_performance(iterations=3)

        env.close()
    """

    def __init__(
        self,
        instance_id: str,
        image_name: str,
        base_commit: str,
        rebuild_cmd: str,
        test_cmd: str,
        covering_tests: Optional[List[str]] = None,
        pass_to_pass: Optional[List[str]] = None,
        workdir: str = "/testbed",
    ):
        """
        Initialize environment for a SWEfficiency instance.

        Args:
            instance_id: Unique instance identifier
            image_name: Docker image name
            base_commit: Git commit to reset the repo to
            rebuild_cmd: Command to rebuild after applying patch
            test_cmd: Command to run tests
            covering_tests: Tests that verify correctness
            pass_to_pass: Tests that should remain passing
            workdir: Working directory in container
        """
        self.instance_id = instance_id
        self.image_name = image_name
        self.base_commit = base_commit
        self.rebuild_cmd = rebuild_cmd
        self.test_cmd = test_cmd
        self.covering_tests = covering_tests or []
        self.pass_to_pass = pass_to_pass or []
        self.workdir = workdir

        logger.info(f"Initializing SWEfficiencyEnv for {instance_id}")
        logger.debug(f"Using Docker image: {image_name}")

        self.client = docker.from_env()
        self.container: Container | None = None
        self._started = False

        # Generate unique container name
        id_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', instance_id)[:32]
        self._container_name = f"swefficiency-{id_safe}-{uuid.uuid4().hex[:8]}"

    @classmethod
    def from_sample(cls, sample: "SWEfficiencySample") -> "SWEfficiencyEnv":
        """Create environment from a SWEfficiencySample."""
        return cls(
            instance_id=sample.instance_id,
            image_name=sample.image_name,
            base_commit=sample.base_commit,
            rebuild_cmd=sample.rebuild_cmd,
            test_cmd=sample.test_cmd,
            covering_tests=sample.covering_tests,
            pass_to_pass=sample.pass_to_pass,
        )

    def start(self) -> None:
        """Start the Docker container and initialize the environment."""
        if self._started:
            logger.warning("Environment already started")
            return

        logger.info(f"Starting container for {self.instance_id}...")

        # Pull image if not available
        try:
            self.client.images.get(self.image_name)
            logger.info(f"Image {self.image_name} found locally")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {self.image_name}...")
            try:
                self.client.images.pull(self.image_name)
            except docker.errors.APIError as e:
                raise RuntimeError(
                    f"Failed to pull Docker image {self.image_name}. "
                    f"Make sure the image exists and you have access: {e}"
                )

        # Start container with auto-remove on exit for cleanup
        self.container = self.client.containers.run(
            self.image_name,
            name=self._container_name,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir=self.workdir,
            auto_remove=True,  # Automatically remove container when stopped
        )
        logger.info(f"Container {self._container_name} started")

        # Reset to base commit
        if self.base_commit:
            logger.info(f"Resetting to base commit: {self.base_commit}")
            output = self.communicate(f"git reset --hard {self.base_commit}")
            logger.debug(f"Git reset output: {output[:200] if output else '(empty)'}")

        self._started = True
        logger.info("Container started successfully")

    def communicate(self, cmd: str, timeout: int = 300) -> str:
        """
        Execute a command in the container.

        Args:
            cmd: Bash command to execute
            timeout: Timeout in seconds (default: 5 minutes)

        Returns:
            Command output as string
        """
        if not self._started and self.container is None:
            raise RuntimeError("Environment not started. Call start() first.")

        # Use exec_run to execute command
        exit_code, output = self.container.exec_run(
            f"bash -c {repr(cmd)}",
            workdir=self.workdir,
        )

        result = output.decode("utf-8", errors="replace")
        return result

    def communicate_with_exit_code(self, cmd: str, timeout: int = 300) -> Tuple[int, str]:
        """
        Execute a command and return both exit code and output.

        Args:
            cmd: Bash command to execute
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, output)
        """
        if not self._started and self.container is None:
            raise RuntimeError("Environment not started. Call start() first.")

        exit_code, output = self.container.exec_run(
            f"bash -c {repr(cmd)}",
            workdir=self.workdir,
        )

        result = output.decode("utf-8", errors="replace")
        return exit_code, result

    def read_file(self, path: str) -> str:
        """Read a file from the container."""
        return self.communicate(f"cat {repr(path)}")

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the container."""
        content_b64 = base64.b64encode(content.encode()).decode()
        self.communicate(f"echo {content_b64} | base64 -d > {repr(path)}")

    def apply_patch(self, patch_content: str) -> Tuple[bool, str]:
        """
        Apply a git patch to the repository.

        Args:
            patch_content: Git patch content (unified diff format)

        Returns:
            Tuple of (success, output)
        """
        if not patch_content.strip():
            return False, "Empty patch content"

        # Write patch to temp file
        patch_path = "/tmp/optimization.patch"
        self.write_file(patch_path, patch_content)

        # Apply the patch
        exit_code, output = self.communicate_with_exit_code(
            f"git apply --verbose {patch_path}"
        )

        if exit_code != 0:
            # Try with --3way for better merge handling
            exit_code, output = self.communicate_with_exit_code(
                f"git apply --3way {patch_path}"
            )

        success = exit_code == 0
        if success:
            logger.info("Patch applied successfully")
        else:
            logger.warning(f"Patch application failed: {output[:200]}")

        return success, output

    def rebuild(self) -> Tuple[bool, str]:
        """
        Rebuild the project after applying a patch.

        Returns:
            Tuple of (success, output)
        """
        if not self.rebuild_cmd:
            logger.info("No rebuild command specified, skipping rebuild")
            return True, ""

        logger.info(f"Running rebuild: {self.rebuild_cmd}")
        exit_code, output = self.communicate_with_exit_code(self.rebuild_cmd)

        success = exit_code == 0
        if success:
            logger.info("Rebuild completed successfully")
        else:
            logger.warning(f"Rebuild failed with exit code {exit_code}")

        return success, output

    def run_tests(self, specific_tests: Optional[List[str]] = None) -> TestResult:
        """
        Run tests to verify correctness.

        Args:
            specific_tests: Optional list of specific tests to run.
                           If None, runs all tests using test_cmd.

        Returns:
            TestResult with pass/fail counts
        """
        if specific_tests:
            # Run specific tests
            test_str = " ".join(specific_tests)
            cmd = f"{self.test_cmd} {test_str}"
        else:
            cmd = self.test_cmd

        if not cmd:
            return TestResult(
                passed=0,
                failed=0,
                total=0,
                output="No test command specified",
            )

        logger.info(f"Running tests: {cmd[:100]}...")
        exit_code, output = self.communicate_with_exit_code(cmd)

        # Parse test results from output
        # This is a heuristic - different test frameworks have different formats
        passed, failed = self._parse_test_output(output, exit_code)

        return TestResult(
            passed=passed,
            failed=failed,
            total=passed + failed,
            output=output,
            error=None if exit_code == 0 else f"Exit code: {exit_code}",
        )

    def _parse_test_output(self, output: str, exit_code: int) -> Tuple[int, int]:
        """Parse test output to extract pass/fail counts."""
        passed = 0
        failed = 0

        # Try pytest format: "X passed, Y failed"
        pytest_match = re.search(r"(\d+)\s+passed", output)
        if pytest_match:
            passed = int(pytest_match.group(1))
        pytest_fail = re.search(r"(\d+)\s+failed", output)
        if pytest_fail:
            failed = int(pytest_fail.group(1))

        # Try unittest format: "OK (N tests)" or "FAILED (failures=X)"
        if passed == 0 and failed == 0:
            unittest_ok = re.search(r"OK\s*\(.*?(\d+)\s*test", output)
            if unittest_ok:
                passed = int(unittest_ok.group(1))
            unittest_fail = re.search(r"failures=(\d+)", output)
            if unittest_fail:
                failed = int(unittest_fail.group(1))

        # If we couldn't parse, use exit code as heuristic
        if passed == 0 and failed == 0:
            if exit_code == 0:
                passed = 1  # Assume at least 1 test passed
            else:
                failed = 1  # Assume at least 1 test failed

        return passed, failed

    def measure_performance(
        self,
        benchmark_cmd: Optional[str] = None,
        iterations: int = 3,
        warmup: int = 1,
    ) -> BenchmarkResult:
        """
        Measure performance by running a benchmark command.

        Args:
            benchmark_cmd: Command to run for benchmarking.
                          If None, uses test_cmd with timing.
            iterations: Number of iterations to average
            warmup: Number of warmup iterations (not counted)

        Returns:
            BenchmarkResult with timing information
        """
        cmd = benchmark_cmd or self.test_cmd
        if not cmd:
            return BenchmarkResult(
                passed=False,
                execution_time=0.0,
                output="No benchmark command specified",
                error="No command",
            )

        times = []
        last_output = ""

        # Warmup runs
        for _ in range(warmup):
            self.communicate(cmd)

        # Timed runs
        for i in range(iterations):
            start = time.time()
            exit_code, output = self.communicate_with_exit_code(cmd)
            elapsed = time.time() - start
            times.append(elapsed)
            last_output = output
            logger.debug(f"Benchmark iteration {i+1}: {elapsed:.3f}s")

        avg_time = sum(times) / len(times) if times else 0.0

        return BenchmarkResult(
            passed=True,
            execution_time=avg_time,
            output=last_output,
        )

    def get_patch(self) -> str:
        """Get the git diff from workdir."""
        return self.communicate("git -c core.fileMode=false diff HEAD")

    def get_status(self) -> str:
        """Get git status from workdir."""
        return self.communicate("git status")

    def reset(self) -> None:
        """Reset the repository to base commit."""
        if self.base_commit:
            self.communicate(f"git reset --hard {self.base_commit}")
            self.communicate("git clean -fd")

    def close(self) -> None:
        """Stop the container and clean up resources."""
        if self.container is None:
            return

        logger.info(f"Closing environment for {self.instance_id}")
        try:
            self.container.stop(timeout=5)
            # Note: auto_remove=True means container is removed when stopped
            # We don't need to call remove() explicitly
        except Exception as e:
            logger.warning(f"Error stopping container: {e}")

        self.container = None
        self._started = False

    def __enter__(self) -> "SWEfficiencyEnv":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
