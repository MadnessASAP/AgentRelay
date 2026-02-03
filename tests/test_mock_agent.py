"""Integration tests for mock agent making MCP tool calls.

These tests start an actual MCP server and make tool calls through the MCP protocol,
testing the tool call interface as a real agent would experience it.

This does NOT test the prompt injection mechanism - only the tool call interface.
"""

import asyncio
import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from mcp.server import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session


class MockJobState(str, Enum):
    """Mock job states matching the real JobState enum."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MockJob:
    """Mock job object returned by the mock manager."""
    id: str
    prompt: str
    state: MockJobState = MockJobState.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None


class MockManager:
    """Mock manager for testing MCP server tool calls in isolation.

    Provides configurable behavior for testing different scenarios
    without involving the real JobManager, backend, or injection logic.
    """

    def __init__(self):
        self.jobs: Dict[str, MockJob] = {}
        self._job_counter = 0
        self._run_job_behavior = "complete"
        self._run_job_result = "mock result"
        self._run_job_error = "mock error"
        self._run_job_started: Optional[asyncio.Event] = None
        self._run_job_proceed: Optional[asyncio.Event] = None
        # Track calls for verification
        self.create_job_calls: list = []
        self.run_job_calls: list = []
        self.get_job_calls: list = []

    def create_job(self, prompt: str) -> MockJob:
        """Create a new mock job in queued state."""
        self.create_job_calls.append({"prompt": prompt})
        self._job_counter += 1
        job_id = f"mock-job-{self._job_counter}"
        job = MockJob(id=job_id, prompt=prompt, state=MockJobState.QUEUED)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> MockJob:
        """Get a job by ID, raises KeyError if not found."""
        self.get_job_calls.append({"job_id": job_id})
        return self.jobs[job_id]

    async def run_job(self, job: MockJob):
        """Simulate running a job with configurable behavior."""
        self.run_job_calls.append({"job_id": job.id})
        job.state = MockJobState.RUNNING

        if self._run_job_started:
            self._run_job_started.set()

        if self._run_job_proceed:
            await self._run_job_proceed.wait()

        if self._run_job_behavior == "complete":
            job.state = MockJobState.COMPLETED
            job.result = self._run_job_result
        elif self._run_job_behavior == "fail":
            job.state = MockJobState.FAILED
            job.error = self._run_job_error

    def configure_success(self, result: str = "mock result"):
        """Configure run_job to complete successfully."""
        self._run_job_behavior = "complete"
        self._run_job_result = result

    def configure_failure(self, error: str = "mock error"):
        """Configure run_job to fail with an error."""
        self._run_job_behavior = "fail"
        self._run_job_error = error

    def configure_slow_execution(self):
        """Configure run_job to wait for explicit signal before completing."""
        self._run_job_started = asyncio.Event()
        self._run_job_proceed = asyncio.Event()
        return self._run_job_started, self._run_job_proceed


def create_mcp_server(manager: MockManager) -> FastMCP:
    """Create an MCP server with delegate and status tools using the given manager.

    This mirrors the real server setup but uses a mock manager for testing.
    """
    mcp = FastMCP("test-mcp-subagent")

    @mcp.tool(description="Launch a subagent task (injection-first; result is not returned via tool).")
    async def delegate(prompt: str, block: bool = False) -> dict:
        """Launch a subagent task."""
        job = manager.create_job(prompt)
        task = asyncio.create_task(manager.run_job(job))

        if not block:
            return {"subagent_id": job.id, "state": job.state.value}

        # block=true: Wait for completion, but DO NOT return the result
        await task
        return {"subagent_id": job.id, "state": job.state.value}

    @mcp.tool(description="Get subagent status (debug/manual; avoid polling in normal operation).")
    async def status(subagent_id: str) -> dict:
        """Get subagent status."""
        job = manager.get_job(subagent_id)
        return {
            "state": job.state.value,
            "result": job.result,
            "error": job.error,
        }

    return mcp


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
                import json
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


@pytest.fixture
def mock_manager():
    """Create a fresh MockManager for each test."""
    return MockManager()


@pytest.fixture
def mcp_server(mock_manager):
    """Create an MCP server with the mock manager."""
    return create_mcp_server(mock_manager)


@pytest.fixture
async def agent(mcp_server):
    """Create a MockAgent connected to the MCP server."""
    async with create_connected_server_and_client_session(mcp_server) as client_session:
        yield MockAgent(client_session)


class TestMockAgentDelegateToolCalls:
    """Tests for mock agent making delegate tool calls through MCP."""

    @pytest.mark.asyncio
    async def test_agent_delegate_non_blocking_returns_job_info(self, mcp_server, mock_manager):
        """Agent delegate call through MCP returns subagent_id and queued state."""
        mock_manager.configure_success()

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="analyze this code")

            assert "subagent_id" in response
            assert response["state"] == "queued"

    @pytest.mark.asyncio
    async def test_agent_delegate_records_tool_call(self, mcp_server):
        """Agent records the delegate tool call for verification."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            await agent.delegate_task(prompt="test prompt")

            assert len(agent.tool_calls) == 1
            assert agent.tool_calls[0]["tool"] == "delegate"
            assert agent.tool_calls[0]["arguments"]["prompt"] == "test prompt"

    @pytest.mark.asyncio
    async def test_agent_delegate_passes_prompt_to_manager(self, mcp_server, mock_manager):
        """Agent delegate call passes prompt to manager.create_job via MCP."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            await agent.delegate_task(prompt="specific prompt text")

            assert len(mock_manager.create_job_calls) == 1
            assert mock_manager.create_job_calls[0]["prompt"] == "specific prompt text"

    @pytest.mark.asyncio
    async def test_agent_delegate_blocking_waits_for_completion(self, mcp_server, mock_manager):
        """Agent blocking delegate call through MCP waits and returns final state."""
        mock_manager.configure_success(result="task completed")

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="do something", block=True)

            assert response["state"] == "completed"
            assert "subagent_id" in response

    @pytest.mark.asyncio
    async def test_agent_delegate_blocking_returns_failed_on_error(self, mcp_server, mock_manager):
        """Agent blocking delegate call returns failed state on error."""
        mock_manager.configure_failure(error="execution error")

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="failing task", block=True)

            assert response["state"] == "failed"

    @pytest.mark.asyncio
    async def test_agent_delegate_response_excludes_result_content(self, mcp_server, mock_manager):
        """Agent delegate response through MCP does not include actual result content."""
        mock_manager.configure_success(result="sensitive result data")

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response = await agent.delegate_task(prompt="task", block=True)

            # The tool call interface should NOT expose the result
            # (injection-first design - result delivered via injection)
            assert "result" not in response
            assert "output" not in response
            assert "content" not in response

    @pytest.mark.asyncio
    async def test_agent_multiple_delegate_calls_create_multiple_jobs(self, mcp_server, mock_manager):
        """Agent making multiple delegate calls through MCP creates separate jobs."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            response1 = await agent.delegate_task(prompt="task 1")
            response2 = await agent.delegate_task(prompt="task 2")
            response3 = await agent.delegate_task(prompt="task 3")

            # Each call creates a unique job
            ids = {response1["subagent_id"], response2["subagent_id"], response3["subagent_id"]}
            assert len(ids) == 3

            # All prompts recorded
            assert len(mock_manager.create_job_calls) == 3


class TestMockAgentStatusToolCalls:
    """Tests for mock agent making status tool calls through MCP."""

    @pytest.mark.asyncio
    async def test_agent_status_returns_job_state(self, mcp_server, mock_manager):
        """Agent status call through MCP returns the job state."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            # Create a job first
            delegate_response = await agent.delegate_task(prompt="test")
            job_id = delegate_response["subagent_id"]

            # Force queued state for predictable test
            mock_manager.jobs[job_id].state = MockJobState.QUEUED

            response = await agent.check_status(job_id)

            assert response["state"] == "queued"

    @pytest.mark.asyncio
    async def test_agent_status_records_tool_call(self, mcp_server, mock_manager):
        """Agent records the status tool call for verification."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            # Create a job first
            delegate_response = await agent.delegate_task(prompt="test")
            job_id = delegate_response["subagent_id"]

            await agent.check_status(job_id)

            # Find the status call (delegate was also recorded)
            status_calls = [c for c in agent.tool_calls if c["tool"] == "status"]
            assert len(status_calls) == 1
            assert status_calls[0]["arguments"]["subagent_id"] == job_id

    @pytest.mark.asyncio
    async def test_agent_status_passes_subagent_id_to_manager(self, mcp_server, mock_manager):
        """Agent status call passes subagent_id to manager.get_job via MCP."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            delegate_response = await agent.delegate_task(prompt="test")
            job_id = delegate_response["subagent_id"]

            await agent.check_status(job_id)

            assert len(mock_manager.get_job_calls) == 1
            assert mock_manager.get_job_calls[0]["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_agent_status_returns_result_for_completed_job(self, mcp_server, mock_manager):
        """Agent status call through MCP returns result for completed job."""
        mock_manager.configure_success(result="the result")

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            delegate_response = await agent.delegate_task(prompt="test", block=True)
            job_id = delegate_response["subagent_id"]

            response = await agent.check_status(job_id)

            assert response["state"] == "completed"
            assert response["result"] == "the result"
            assert response["error"] is None

    @pytest.mark.asyncio
    async def test_agent_status_returns_error_for_failed_job(self, mcp_server, mock_manager):
        """Agent status call through MCP returns error for failed job."""
        mock_manager.configure_failure(error="failure reason")

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            delegate_response = await agent.delegate_task(prompt="test", block=True)
            job_id = delegate_response["subagent_id"]

            response = await agent.check_status(job_id)

            assert response["state"] == "failed"
            assert response["result"] is None
            assert response["error"] == "failure reason"

    @pytest.mark.asyncio
    async def test_agent_status_response_structure(self, mcp_server, mock_manager):
        """Agent status response through MCP has expected keys."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)
            delegate_response = await agent.delegate_task(prompt="test")
            job_id = delegate_response["subagent_id"]

            response = await agent.check_status(job_id)

            assert set(response.keys()) == {"state", "result", "error"}


class TestMockAgentWorkflow:
    """Tests for mock agent workflow scenarios using MCP tool calls."""

    @pytest.mark.asyncio
    async def test_agent_delegate_then_poll_status(self, mcp_server, mock_manager):
        """Agent can delegate a task and poll for status updates via MCP."""
        started, proceed = mock_manager.configure_slow_execution()

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)

            # Non-blocking delegate
            delegate_response = await agent.delegate_task(prompt="long task")
            job_id = delegate_response["subagent_id"]
            assert delegate_response["state"] == "queued"

            # Wait for job to start
            await started.wait()

            # Poll status - should be running
            status_response = await agent.check_status(job_id)
            assert status_response["state"] == "running"

            # Allow completion
            proceed.set()
            await asyncio.sleep(0.01)  # Let the task complete

            # Poll again - should be completed
            status_response = await agent.check_status(job_id)
            assert status_response["state"] == "completed"

    @pytest.mark.asyncio
    async def test_agent_parallel_delegates(self, mcp_server, mock_manager):
        """Agent can make multiple parallel delegate calls through MCP."""
        mock_manager.configure_success()

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)

            # Make parallel delegate calls
            responses = await asyncio.gather(
                agent.delegate_task(prompt="task 1"),
                agent.delegate_task(prompt="task 2"),
                agent.delegate_task(prompt="task 3"),
            )

            # All should return queued state
            for response in responses:
                assert response["state"] == "queued"

            # All should have unique IDs
            ids = {r["subagent_id"] for r in responses}
            assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_agent_tracks_all_tool_calls(self, mcp_server, mock_manager):
        """Agent tracks all tool calls made during a session via MCP."""
        mock_manager.configure_success()

        async with create_connected_server_and_client_session(mcp_server) as client:
            agent = MockAgent(client)

            # Make several calls
            r1 = await agent.delegate_task(prompt="task 1")
            await agent.check_status(r1["subagent_id"])
            r2 = await agent.delegate_task(prompt="task 2", block=True)
            await agent.check_status(r2["subagent_id"])

            # Verify all calls tracked
            assert len(agent.tool_calls) == 4
            assert agent.tool_calls[0]["tool"] == "delegate"
            assert agent.tool_calls[1]["tool"] == "status"
            assert agent.tool_calls[2]["tool"] == "delegate"
            assert agent.tool_calls[3]["tool"] == "status"

    @pytest.mark.asyncio
    async def test_mcp_server_lists_tools(self, mcp_server):
        """MCP server exposes delegate and status tools via list_tools."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            result = await client.list_tools()
            tool_names = {tool.name for tool in result.tools}

            assert "delegate" in tool_names
            assert "status" in tool_names

    @pytest.mark.asyncio
    async def test_delegate_tool_schema_via_mcp(self, mcp_server):
        """Delegate tool exposed via MCP has correct parameter schema."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            result = await client.list_tools()
            delegate_tool = next(t for t in result.tools if t.name == "delegate")

            # Check the input schema
            schema = delegate_tool.inputSchema
            assert schema["type"] == "object"
            assert "prompt" in schema["properties"]
            assert "block" in schema["properties"]

    @pytest.mark.asyncio
    async def test_status_tool_schema_via_mcp(self, mcp_server):
        """Status tool exposed via MCP has correct parameter schema."""
        async with create_connected_server_and_client_session(mcp_server) as client:
            result = await client.list_tools()
            status_tool = next(t for t in result.tools if t.name == "status")

            # Check the input schema
            schema = status_tool.inputSchema
            assert schema["type"] == "object"
            assert "subagent_id" in schema["properties"]
