"""Pytest configuration for intelligence-per-watt tests."""

import sys
from pathlib import Path

# Add the package root to sys.path so grid_eval can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
