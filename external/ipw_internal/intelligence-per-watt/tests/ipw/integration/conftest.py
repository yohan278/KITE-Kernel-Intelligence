"""Shared fixtures for integration tests.

This module provides common fixtures that can be used across all
integration test modules.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs.

    The directory is automatically cleaned up after the test.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory(prefix="ipw_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_openai_api_key() -> Generator[str, None, None]:
    """Set a dummy OpenAI API key for tests that require it.

    Some agents (like OpenHands) require an API key even when using
    a local vLLM server.

    Yields:
        The dummy API key that was set
    """
    original = os.environ.get("OPENAI_API_KEY")
    dummy_key = "test-api-key-for-integration-tests"
    os.environ["OPENAI_API_KEY"] = dummy_key

    yield dummy_key

    if original is not None:
        os.environ["OPENAI_API_KEY"] = original
    elif "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
