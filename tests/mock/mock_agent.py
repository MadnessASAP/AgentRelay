"""Mock agent for testing MCP tool calls."""

import json
from typing import Any, Dict


class MockAgent:
    """Mock agent that makes tool calls through the MCP protocol.

    This agent connects to an actual MCP server and makes tool calls
    through the client session, mimicking how a real agent would interact.
    """

    def __init__(self, client_session):
        self._session = client_session
        self.tool_calls: list = []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Make a tool call through the MCP protocol.

        This sends an actual MCP tools/call request to the server.
        """
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments.copy()
        })

        result = await self._session.call_tool(tool_name, arguments)

        # Extract the result from MCP response
        # MCP returns CallToolResult with content list
        if result.content and len(result.content) > 0:
            content = result.content[0]
            # For structured output, the result is in the text field as JSON
            if hasattr(content, 'text'):
                return json.loads(content.text)

        return {}

    async def delegate_task(self, prompt: str, block: bool = False) -> Dict[str, Any]:
        """Convenience method to call the delegate tool."""
        args = {"prompt": prompt}
        if block:
            args["block"] = block
        return await self.call_tool("delegate", args)

    async def check_status(self, subagent_id: str) -> Dict[str, Any]:
        """Convenience method to call the status tool."""
        return await self.call_tool("status", {"subagent_id": subagent_id})
