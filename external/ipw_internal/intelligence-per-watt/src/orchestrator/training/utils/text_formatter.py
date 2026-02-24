#!/usr/bin/env python3
"""
Text formatter utilities
"""

from typing import Optional
import re

def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace to a single space and trim ends.
    """
    if text is None:
        return ''
    return re.sub(r'\s+', ' ', text).strip()

def remove_non_ascii(text: str) -> str:
    """
    Remove non-ASCII characters from text.
    """
    if text is None:
        return ''
    return ''.join(ch for ch in text if ord(ch) < 128)

def format_text(text: Optional[str], *, lower: bool = False, remove_non_ascii_chars: bool = False) -> str:
    """
    Format text with options:
    - lower: convert to lowercase
    - remove_non_ascii_chars: drop non-ASCII chars
    """
    if text is None:
        return ''
    if lower:
        text = text.lower()
    text = normalize_whitespace(text)
    if remove_non_ascii_chars:
        text = remove_non_ascii(text)
    return text

__all__ = ['normalize_whitespace', 'remove_non_ascii', 'format_text']
