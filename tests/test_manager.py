"""Integration tests for JobManager with the mock backend over MCP.

These tests exercise the real JobManager + MockBackend over an actual MCP server.
They are intended to validate agent creation and prompt injection behavior end-to-end.
"""

import asyncio

import pytest
import mcp.types as types
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.shared.session import RequestResponder

from agent_relay.backends.mock import MockBackend
from agent_relay.manager import JobManager
from agent_relay.server import create_server
from tests.mock import MockAgent


class InjectionRecorder:
    """Capture injection messages from the JobManager."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def __call__(self, message: str) -> None:
        self.messages.append(message)


class PromptCapture:
    """Capture prompt injection notifications on the agent side."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def handle_message(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult]
        | types.ServerNotification
        | Exception,
    ) -> None:
        if isinstance(message, types.ServerNotification):
            if isinstance(message.root, types.LoggingMessageNotification):
                self.messages.append(str(message.root.params.data))


@pytest.fixture
def injection_recorder() -> InjectionRecorder:
    return InjectionRecorder()


@pytest.fixture
def manager(injection_recorder: InjectionRecorder) -> JobManager:
    return JobManager(MockBackend(), inject_message=injection_recorder)


@pytest.fixture
def mcp_server(manager: JobManager):
    return create_server(manager)


class TestJobManagerWithMockBackend:
    """End-to-end tests using the real JobManager with the mock backend."""

    @pytest.mark.asyncio
    async def test_delegate_creates_job_in_manager(self, mcp_server, manager: JobManager):
        """Delegate tool calls should create jobs in the real manager."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="hello from agent")

            job_id = response["subagent_id"]

            assert job_id in manager.jobs
            assert manager.jobs[job_id].prompt == "hello from agent"

    @pytest.mark.asyncio
    async def test_delegate_non_blocking_returns_queued_state(self, mcp_server):
        """Non-blocking delegate calls should return queued state immediately."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="quick task")

            assert response["state"] == "queued"
            assert "result" not in response

    @pytest.mark.asyncio
    async def test_delegate_injects_result_into_calling_agent_prompt(
        self,
        mcp_server,
        manager: JobManager,
        injection_recorder: InjectionRecorder,
    ):
        """Delegate results should be injected into the calling agent's prompt."""
        prompt_capture = PromptCapture()

        async with create_connected_server_and_client_session(
            mcp_server,
            message_handler=prompt_capture.handle_message,
        ) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="return this", block=True)

            job_id = response["subagent_id"]

            assert injection_recorder.messages
            assert any(job_id in message for message in injection_recorder.messages)

            await asyncio.sleep(0.01)

            assert prompt_capture.messages
            assert any(job_id in message for message in prompt_capture.messages)

    @pytest.mark.asyncio
    async def test_delegate_blocking_failure_injects_error(
        self,
        mcp_server,
        injection_recorder: InjectionRecorder,
    ):
        """Backend failures should produce failed state and injected error messages."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="<FAIL>", block=True)

            job_id = response["subagent_id"]

            assert response["state"] == "failed"
            assert injection_recorder.messages
            assert any("failed" in message for message in injection_recorder.messages)
            assert any(job_id in message for message in injection_recorder.messages)

    @pytest.mark.asyncio
    async def test_delegate_tool_requires_prompt(self, mcp_server):
        """Incorrect tool calls should surface errors from the MCP server."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            with pytest.raises(Exception):
                await client.call_tool("delegate", {"block": True})
