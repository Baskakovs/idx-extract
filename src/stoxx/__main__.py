"""Entry point for `python -m stoxx`."""

import asyncio
import logging

from .storage import from_env
from .sync import sync

logging.basicConfig(level=logging.INFO)
_storage = from_env()
asyncio.run(sync(_storage))
