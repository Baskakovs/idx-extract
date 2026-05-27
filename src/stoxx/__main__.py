"""Entry point for `python -m stoxx`."""

import asyncio
import sys

from stoxx.sync import sync

if __name__ == "__main__":
    if "--serve" in sys.argv:
        sync.from_source(
            source="https://github.com/Baskakovs/idx-extract.git",
            entrypoint="prefect_entrypoint.py:sync",
        ).serve(name="deployment-idx-extract")
    else:
        asyncio.run(sync())
