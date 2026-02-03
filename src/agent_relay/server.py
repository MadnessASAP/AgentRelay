import asyncio
import argparse

# NOTE:
# The 'mcp' PyPI package API may differ slightly by version.
# If MCPServer/Tool import paths differ, adjust here.
from mcp.server import MCPServer
from mcp.types import Tool

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
    server = MCPServer("mcp-subagent-mvp")
    handlers = create_tool_handlers(manager)

    server.add_tool(
        Tool(
            name="delegate",
            description="Launch a subagent task (injection-first; result is not returned via tool).",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "block": {"type": "boolean"},
                },
                "required": ["prompt"],
            },
            handler=handlers["delegate"],
        )
    )

    server.add_tool(
        Tool(
            name="status",
            description="Get subagent status (debug/manual; avoid polling in normal operation).",
            parameters={
                "type": "object",
                "properties": {"subagent_id": {"type": "string"}},
                "required": ["subagent_id"],
            },
            handler=handlers["status"],
        )
    )

    return server


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acp-endpoint", required=True, help="Codex-ACP endpoint URL/address")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent subagent jobs")
    args = parser.parse_args()

    backend = ACPBackend(args.acp_endpoint)
    manager = JobManager(backend, inject_into_parent, max_concurrent=args.max_concurrent)

    server = create_server(manager)
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
