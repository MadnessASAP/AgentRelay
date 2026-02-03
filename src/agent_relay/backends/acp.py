import asyncio

# NOTE:
# This is a thin adapter around the codex-acp client API.
# The exact import path / API surface may vary by codex-acp version.
#
# Replace the client usage inside _run_sync() with the exact calls from:
#   https://github.com/zed-industries/codex-acp
#
# The rest of the MVP (JobManager + MCP tools) should not need changes.
try:
    from codex_acp import CodexACPClient  # type: ignore
except Exception:
    CodexACPClient = None  # type: ignore


class ACPBackend:
    """Codex-ACP backend adapter.

    Contract:
      async run(prompt: str) -> str
    """

    def __init__(self, endpoint: str):
        if CodexACPClient is None:
            raise ImportError(
                "codex-acp client import failed. Install/adjust codex-acp and update backends/acp.py."
            )
        self.client = CodexACPClient(endpoint)

    async def run(self, prompt: str) -> str:
        # If codex-acp is synchronous, run in an executor.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, prompt)

    def _run_sync(self, prompt: str) -> str:
        # --- IMPORTANT ---
        # This is placeholder logic: adjust to actual codex-acp API.
        #
        # Pseudocode:
        #   session = self.client.start_session()
        #   session.send(prompt)
        #   response = session.wait()
        #   return response.text
        #
        session = self.client.start_session()
        session.send(prompt)
        response = session.wait()
        return getattr(response, "text", str(response))
