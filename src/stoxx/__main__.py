"""Entry point for `python -m stoxx`."""

import asyncio
import sys

from .sync import sync

if __name__ == "__main__":
    if "--serve" in sys.argv:
        sync.serve(name="deployment-idx-extract")
    else:
        asyncio.run(sync())
