"""
SWE-bench environment wrapper using Docker directly.

Provides a simplified interface for container communication.
Uses docker-py directly for simpler compatibility with SWE-bench images.
"""
from __future__ import annotations

import logging
import subprocess
import uuid
from typing import TYPE_CHECKING

import docker
from docker.models.containers import Container

if TYPE_CHECKING:
    from .dataset import SWEBenchSample

logger = logging.getLogger(__name__)


class SWEBenchEnv:
    """
    Environment wrapper for SWE-bench evaluation using Docker directly.
    
    Manages communication with a SWE-bench Docker container.
    All commands sent via communicate() execute INSIDE the container.
    
    Usage:
        env = SWEBenchEnv(sample.instance_id, sample.base_commit)
        env.start()
        output = env.communicate("ls /testbed")
        patch = env.get_patch()
        env.close()
    """
    
    def __init__(self, instance_id: str, base_commit: str):
        """
        Initialize environment for a SWE-bench instance.
        
        Args:
            instance_id: SWE-bench instance ID (e.g., "django__django-16379")
            base_commit: Git commit to reset the repo to
        """
        self.instance_id = instance_id
        self.base_commit = base_commit
        
        # Docker image naming from SWE-agent's batch_instances.py
        # Docker doesn't allow double underscore, so replace with magic token
        id_docker = instance_id.replace("__", "_1776_")
        self.image = f"swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
        
        logger.info(f"Initializing SWEBenchEnv for {instance_id}")
        logger.debug(f"Using Docker image: {self.image}")
        
        self.client = docker.from_env()
        self.container: Container | None = None
        self._started = False
        self._container_name = f"swebench-{id_docker}-{uuid.uuid4().hex[:8]}"
    
    @classmethod
    def from_sample(cls, sample: "SWEBenchSample") -> "SWEBenchEnv":
        """Create environment from a SWEBenchSample."""
        return cls(
            instance_id=sample.instance_id,
            base_commit=sample.base_commit,
        )
    
    def start(self) -> None:
        """Start the Docker container and initialize the environment."""
        if self._started:
            logger.warning("Environment already started")
            return
        
        logger.info(f"Starting container for {self.instance_id}...")
        
        # Pull image if not available
        try:
            self.client.images.get(self.image)
            logger.info(f"Image {self.image} found locally")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)
        
        # Start container
        self.container = self.client.containers.run(
            self.image,
            name=self._container_name,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/testbed",
        )
        logger.info(f"Container {self._container_name} started")
        
        # Reset to base commit
        logger.info(f"Resetting to base commit: {self.base_commit}")
        output = self.communicate(f"git reset --hard {self.base_commit}")
        logger.debug(f"Git reset output: {output[:200] if output else '(empty)'}")
        
        self._started = True
        logger.info("Container started successfully")
    
    def communicate(self, cmd: str, timeout: int = 30) -> str:
        """
        Execute a command in the container.
        
        Args:
            cmd: Bash command to execute
            timeout: Timeout in seconds
            
        Returns:
            Command output as string
        """
        if not self._started and self.container is None:
            raise RuntimeError("Environment not started. Call start() first.")
        
        # Use exec_run to execute command
        exit_code, output = self.container.exec_run(
            f"bash -c {repr(cmd)}",
            workdir="/testbed",
        )
        
        result = output.decode("utf-8", errors="replace")
        return result
    
    def read_file(self, path: str) -> str:
        """
        Read a file from the container.
        
        Args:
            path: Absolute path to file in container
            
        Returns:
            File contents as string
        """
        return self.communicate(f"cat {repr(path)}")
    
    def write_file(self, path: str, content: str) -> None:
        """
        Write content to a file in the container.
        
        Args:
            path: Absolute path to file in container
            content: Content to write
        """
        import base64
        # Use base64 encoding to avoid shell escaping issues
        content_b64 = base64.b64encode(content.encode()).decode()
        self.communicate(f"echo {content_b64} | base64 -d > {repr(path)}")
    
    def get_patch(self) -> str:
        """
        Get the git diff from /testbed.
        
        Returns:
            Git diff as string (empty if no changes)
        """
        return self.communicate("git -c core.fileMode=false diff HEAD")
    
    def get_status(self) -> str:
        """Get git status from /testbed."""
        return self.communicate("git status")
    
    def close(self) -> None:
        """Stop the container and clean up resources."""
        if self.container is None:
            return
        
        logger.info(f"Closing environment for {self.instance_id}")
        try:
            self.container.stop(timeout=5)
            self.container.remove(force=True)
        except Exception as e:
            logger.warning(f"Error stopping container: {e}")
        
        self.container = None
        self._started = False
    
    def __enter__(self) -> "SWEBenchEnv":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
