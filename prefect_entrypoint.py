"""Prefect entrypoint — thin wrapper that sets up sys.path for the stoxx package."""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading stoxx package from %s", Path(__file__).resolve().parent / "src")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stoxx.sync import sync  # noqa: E402, F401

logger.info("Successfully imported sync flow")
