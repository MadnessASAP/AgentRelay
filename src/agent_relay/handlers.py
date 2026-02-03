"""Tool handler functions for the MCP server.

Separated from server.py for testability - these handlers don't depend on MCP types.
"""

import asyncio


def create_tool_handlers(manager):
    """Create tool handler functions with the given manager.

    Returns a dict with 'delegate' and 'status' async handler functions.
    """

    async def delegate(prompt: str, block: bool = False):
        """Launch a subagent task (injection-first; result is not returned via tool)."""
        job = manager.create_job(prompt)
        task = asyncio.create_task(manager.run_job(job))

        if not block:
            return {"subagent_id": job.id, "state": job.state.value}

        # block=true:
        # Wait for completion, but DO NOT return the result in the tool response.
        # Completion output is delivered via injection.
        await task
        return {"subagent_id": job.id, "state": job.state.value}

    async def status(subagent_id: str):
        """Get subagent status (debug/manual; avoid polling in normal operation)."""
        job = manager.get_job(subagent_id)
        return {
            "state": job.state.value,
            "result": job.result,  # Debug only; normal flow uses injection.
            "error": job.error,
        }

    return {"delegate": delegate, "status": status}
