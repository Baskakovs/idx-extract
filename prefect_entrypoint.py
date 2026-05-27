"""Prefect entrypoint — thin wrapper that sets up sys.path for the stoxx package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stoxx.sync import sync  # noqa: F401
