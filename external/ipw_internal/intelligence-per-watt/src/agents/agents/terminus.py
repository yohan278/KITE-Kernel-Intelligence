"""Terminus agent implementation for terminal-based tasks."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import docker
from docker.models.containers import Container
from terminal_bench.agents.terminus_2 import Terminus2
from terminal_bench.terminal.tmux_session import TmuxSession

from agents.base import BaseAgent

if TYPE_CHECKING:
    from ipw.src.telemetry.events import EventRecorder


# Default Docker image with tmux pre-installed
DEFAULT_DOCKER_IMAGE = "ubuntu:22.04"


class Terminus(BaseAgent):
    """Terminus agent for terminal-based task execution in Docker containers."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: str,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        container_name: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Terminus agent.

        Args:
            model: The model name to use (e.g., "gpt-4o").
            docker_image: Docker image to use for the container. Must have tmux installed.
                Defaults to ubuntu:22.04.
            container_name: Optional name for the Docker container. If not provided,
                uses an existing container or creates one named "terminus-container".
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to Terminus2.
        """
        super().__init__(event_recorder=event_recorder)

        self.agent = Terminus2(model_name=model, **kwargs)
        self._docker_image = docker_image
        self._container_name = container_name or "terminus-container"
        self._docker_client: docker.DockerClient | None = None
        self._container: Container | None = None
        self._owns_container = False  # Track if we created the container

    def _get_docker_client(self) -> docker.DockerClient:
        """Get or create the Docker client."""
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    def _get_or_create_container(self) -> Container:
        """Get an existing container or create a new one with tmux installed."""
        if self._container is not None:
            return self._container

        client = self._get_docker_client()

        # Try to get an existing container by name
        try:
            container = client.containers.get(self._container_name)
            if container.status != "running":
                container.start()
            self._container = container
            return container
        except docker.errors.NotFound:
            pass  # Container doesn't exist, create it

        # Create a new container with tmux installed
        # Use a command that keeps the container running and installs tmux
        container = client.containers.run(
            self._docker_image,
            command="/bin/bash -c 'apt-get update && apt-get install -y tmux && tail -f /dev/null'",
            name=self._container_name,
            detach=True,
            tty=True,
            stdin_open=True,
        )
        self._container = container
        self._owns_container = True

        # Wait for tmux installation to complete
        for _ in range(30):  # Wait up to 30 seconds
            exit_code, output = container.exec_run("which tmux")
            if exit_code == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Timeout waiting for tmux installation in container")

        return container

    def get_session(self, tmux_session: TmuxSession | str | None = None) -> TmuxSession:
        """Get or create a TmuxSession.

        Args:
            tmux_session: Either an existing TmuxSession, a session name string,
                or None to create a default session.

        Returns:
            A TmuxSession instance.
        """
        if isinstance(tmux_session, TmuxSession):
            return tmux_session

        container = self._get_or_create_container()
        session_name = tmux_session if isinstance(tmux_session, str) else "terminus-session"

        return TmuxSession(
            session_name=session_name,
            container=container,
            disable_recording=True,  # Disable asciinema recording by default
        )

    def run(
        self,
        input: str,
        tmux_session: TmuxSession | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the Terminus agent.

        Args:
            input: The input message or prompt for the agent.
            tmux_session: Optional TmuxSession or session name. If not provided,
                a default session will be created.
            **kwargs: Additional keyword arguments passed to agent.perform_task().

        Returns:
            The terminal output from the session.
        """
        self._record_event("lm_inference_start", model=str(self.agent))
        try:
            session = self.get_session(tmux_session)
            self.agent.perform_task(input, session=session, **kwargs)

            terminal_output = session.capture_pane(capture_entire=True)
            return terminal_output
        finally:
            self._record_event("lm_inference_end", model=str(self.agent))

    def cleanup(self) -> None:
        """Clean up Docker resources.

        Stops and removes the container if we created it.
        """
        if self._container is not None and self._owns_container:
            try:
                self._container.stop()
                self._container.remove()
            except Exception:
                pass  # Ignore cleanup errors
            self._container = None

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.cleanup()
