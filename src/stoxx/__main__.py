"""Entry point for `python -m stoxx`."""

import asyncio
import sys

from prefect.runner.storage import GitRepository

from stoxx.sync import sync

SOURCE = GitRepository(
    url="https://github.com/Baskakovs/idx-extract.git",
    branch="main",
)

if __name__ == "__main__":
    if "--deploy" in sys.argv:
        sync.from_source(
            source=SOURCE,
            entrypoint="prefect_entrypoint.py:sync",
        ).deploy(  # type: ignore[union-attr]
            name="deployment-idx-extract",
            work_pool_name="idx-extract-stoxx",
        )
    elif "--serve" in sys.argv:
        sync.from_source(
            source=SOURCE,
            entrypoint="prefect_entrypoint.py:sync",
        ).serve(name="deployment-idx-extract")  # type: ignore[union-attr]
    else:
        asyncio.run(sync())
