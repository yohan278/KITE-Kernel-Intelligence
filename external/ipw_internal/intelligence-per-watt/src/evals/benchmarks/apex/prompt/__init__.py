# prompt/__init__.py
"""APEX prompt templates."""

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent

GRADING_PROMPT_PATH = _PROMPT_DIR / "grading_prompt.txt"
RESPONSE_GENERATION_PROMPT_PATH = _PROMPT_DIR / "response_generation_prompt.txt"


def load_grading_prompt() -> str:
    """Load the grading prompt template."""
    return GRADING_PROMPT_PATH.read_text()


def load_response_generation_prompt() -> str:
    """Load the response generation prompt template."""
    return RESPONSE_GENERATION_PROMPT_PATH.read_text()

