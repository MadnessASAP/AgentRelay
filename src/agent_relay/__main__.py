"""Allow running as `python -m agent_relay`."""

import asyncio
from .server import main

if __name__ == "__main__":
    asyncio.run(main())
