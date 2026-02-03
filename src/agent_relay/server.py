import asyncio
import argparse

from mcp.server import FastMCP

from .manager import JobManager
from .backends import ACPBackend
from .handlers import create_tool_handlers


async def inject_into_parent(message: str):
    """Injection hook.

    MVP behavior:
      - Print the injected message.

    Replace this with a Codex-host integration to append messages into the parent
    agent run context (the core requirement for non-polling orchestration).
    """
    print("\n=== INJECTED INTO PARENT CONTEXT ===")
    print(message)
    print("===================================\n")


def create_server(manager):
    """Create and configure the MCP server with the given manager.

    Extracted for testability.
    """
    server = FastMCP("mcp-subagent-mvp")
    handlers = create_tool_handlers(manager)

    @server.tool(description="Launch a subagent task (injection-first; result is not returned via tool).")
    async def delegate(prompt: str, block: bool = False):
        return await handlers["delegate"](prompt, block)

    @server.tool(description="Get subagent status (debug/manual; avoid polling in normal operation).")
    async def status(subagent_id: str):
        return await handlers["status"](subagent_id)

    return server


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acp-endpoint", required=True, help="Codex-ACP endpoint URL/address")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent subagent jobs")
    args = parser.parse_args()

    backend = ACPBackend(args.acp_endpoint)
    manager = JobManager(backend, inject_into_parent, max_concurrent=args.max_concurrent)

    server = create_server(manager)
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
