"""Tests for mock agent making MCP tool calls.

These tests simulate an agent making tool calls to the `delegate` and `status` tools,
testing the tool call interface (parameter passing, response handling) with a mock manager.

This does NOT test the prompt injection mechanism - only the tool call interface.
"""

import asyncio
import json
import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from agent_relay.handlers import create_tool_handlers


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
    """Mock manager for testing tool calls in isolation.

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


class MockAgent:
    """Mock agent that simulates making MCP tool calls.

    This agent uses the tool call interface (JSON parameters, dict responses)
    to interact with the delegate and status tools, mimicking how a real
    LLM agent would make tool calls through MCP.
    """

    def __init__(self, handlers: Dict[str, Any]):
        self._handlers = handlers
        self.tool_calls: list = []  # Record of all tool calls made

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Make a tool call with JSON-serializable arguments.

        Simulates the MCP tool call interface:
        1. Arguments are passed as a JSON-like dict (as they would be from MCP)
        2. The handler is invoked with unpacked arguments
        3. Response is returned as a JSON-serializable dict

        This tests the tool call interface without testing prompt injection.
        """
        # Record the call
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments.copy()
        })

        # Get the handler
        if tool_name not in self._handlers:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self._handlers[tool_name]

        # Call the handler with unpacked arguments (as MCP would)
        response = await handler(**arguments)

        # Verify response is JSON-serializable (as MCP requires)
        try:
            json.dumps(response)
        except (TypeError, ValueError) as e:
            raise AssertionError(f"Tool response is not JSON-serializable: {e}")

        return response

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
def handlers(mock_manager):
    """Create tool handlers with the mock manager."""
    return create_tool_handlers(mock_manager)


@pytest.fixture
def agent(handlers):
    """Create a MockAgent with the tool handlers."""
    return MockAgent(handlers)


class TestMockAgentDelegateToolCalls:
    """Tests for mock agent making delegate tool calls."""

    @pytest.mark.asyncio
    async def test_agent_delegate_non_blocking_returns_job_info(self, agent, mock_manager):
        """Agent delegate call returns subagent_id and queued state."""
        mock_manager.configure_success()

        response = await agent.delegate_task(prompt="analyze this code")

        assert "subagent_id" in response
        assert response["state"] == "queued"

    @pytest.mark.asyncio
    async def test_agent_delegate_records_tool_call(self, agent):
        """Agent records the delegate tool call for verification."""
        await agent.delegate_task(prompt="test prompt")

        assert len(agent.tool_calls) == 1
        assert agent.tool_calls[0]["tool"] == "delegate"
        assert agent.tool_calls[0]["arguments"]["prompt"] == "test prompt"

    @pytest.mark.asyncio
    async def test_agent_delegate_passes_prompt_to_manager(self, agent, mock_manager):
        """Agent delegate call passes prompt to manager.create_job."""
        await agent.delegate_task(prompt="specific prompt text")

        assert len(mock_manager.create_job_calls) == 1
        assert mock_manager.create_job_calls[0]["prompt"] == "specific prompt text"

    @pytest.mark.asyncio
    async def test_agent_delegate_blocking_waits_for_completion(self, agent, mock_manager):
        """Agent blocking delegate call waits and returns final state."""
        mock_manager.configure_success(result="task completed")

        response = await agent.delegate_task(prompt="do something", block=True)

        assert response["state"] == "completed"
        assert "subagent_id" in response

    @pytest.mark.asyncio
    async def test_agent_delegate_blocking_returns_failed_on_error(self, agent, mock_manager):
        """Agent blocking delegate call returns failed state on error."""
        mock_manager.configure_failure(error="execution error")

        response = await agent.delegate_task(prompt="failing task", block=True)

        assert response["state"] == "failed"

    @pytest.mark.asyncio
    async def test_agent_delegate_response_excludes_result_content(self, agent, mock_manager):
        """Agent delegate response does not include actual result content."""
        mock_manager.configure_success(result="sensitive result data")

        response = await agent.delegate_task(prompt="task", block=True)

        # The tool call interface should NOT expose the result
        # (injection-first design - result delivered via injection)
        assert "result" not in response
        assert "output" not in response
        assert "content" not in response

    @pytest.mark.asyncio
    async def test_agent_delegate_response_is_json_serializable(self, agent, mock_manager):
        """Agent delegate response must be JSON-serializable."""
        mock_manager.configure_success()

        response = await agent.delegate_task(prompt="test")

        # Should not raise - verified in call_tool
        serialized = json.dumps(response)
        assert isinstance(serialized, str)

    @pytest.mark.asyncio
    async def test_agent_multiple_delegate_calls_create_multiple_jobs(self, agent, mock_manager):
        """Agent making multiple delegate calls creates separate jobs."""
        response1 = await agent.delegate_task(prompt="task 1")
        response2 = await agent.delegate_task(prompt="task 2")
        response3 = await agent.delegate_task(prompt="task 3")

        # Each call creates a unique job
        ids = {response1["subagent_id"], response2["subagent_id"], response3["subagent_id"]}
        assert len(ids) == 3

        # All prompts recorded
        assert len(mock_manager.create_job_calls) == 3


class TestMockAgentStatusToolCalls:
    """Tests for mock agent making status tool calls."""

    @pytest.mark.asyncio
    async def test_agent_status_returns_job_state(self, agent, mock_manager):
        """Agent status call returns the job state."""
        # Create a job first
        delegate_response = await agent.delegate_task(prompt="test")
        job_id = delegate_response["subagent_id"]

        # Force queued state for predictable test
        mock_manager.jobs[job_id].state = MockJobState.QUEUED

        response = await agent.check_status(job_id)

        assert response["state"] == "queued"

    @pytest.mark.asyncio
    async def test_agent_status_records_tool_call(self, agent, mock_manager):
        """Agent records the status tool call for verification."""
        # Create a job first
        delegate_response = await agent.delegate_task(prompt="test")
        job_id = delegate_response["subagent_id"]

        await agent.check_status(job_id)

        # Find the status call (delegate was also recorded)
        status_calls = [c for c in agent.tool_calls if c["tool"] == "status"]
        assert len(status_calls) == 1
        assert status_calls[0]["arguments"]["subagent_id"] == job_id

    @pytest.mark.asyncio
    async def test_agent_status_passes_subagent_id_to_manager(self, agent, mock_manager):
        """Agent status call passes subagent_id to manager.get_job."""
        delegate_response = await agent.delegate_task(prompt="test")
        job_id = delegate_response["subagent_id"]

        await agent.check_status(job_id)

        assert len(mock_manager.get_job_calls) == 1
        assert mock_manager.get_job_calls[0]["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_agent_status_returns_result_for_completed_job(self, agent, mock_manager):
        """Agent status call returns result for completed job."""
        mock_manager.configure_success(result="the result")

        delegate_response = await agent.delegate_task(prompt="test", block=True)
        job_id = delegate_response["subagent_id"]

        response = await agent.check_status(job_id)

        assert response["state"] == "completed"
        assert response["result"] == "the result"
        assert response["error"] is None

    @pytest.mark.asyncio
    async def test_agent_status_returns_error_for_failed_job(self, agent, mock_manager):
        """Agent status call returns error for failed job."""
        mock_manager.configure_failure(error="failure reason")

        delegate_response = await agent.delegate_task(prompt="test", block=True)
        job_id = delegate_response["subagent_id"]

        response = await agent.check_status(job_id)

        assert response["state"] == "failed"
        assert response["result"] is None
        assert response["error"] == "failure reason"

    @pytest.mark.asyncio
    async def test_agent_status_response_structure(self, agent, mock_manager):
        """Agent status response has expected keys."""
        delegate_response = await agent.delegate_task(prompt="test")
        job_id = delegate_response["subagent_id"]

        response = await agent.check_status(job_id)

        assert set(response.keys()) == {"state", "result", "error"}

    @pytest.mark.asyncio
    async def test_agent_status_unknown_job_raises_error(self, agent):
        """Agent status call for unknown job raises KeyError."""
        with pytest.raises(KeyError):
            await agent.check_status("nonexistent-job-id")

    @pytest.mark.asyncio
    async def test_agent_status_response_is_json_serializable(self, agent, mock_manager):
        """Agent status response must be JSON-serializable."""
        delegate_response = await agent.delegate_task(prompt="test")
        job_id = delegate_response["subagent_id"]

        response = await agent.check_status(job_id)

        serialized = json.dumps(response)
        assert isinstance(serialized, str)


class TestMockAgentWorkflow:
    """Tests for mock agent workflow scenarios using tool calls."""

    @pytest.mark.asyncio
    async def test_agent_delegate_then_poll_status(self, agent, mock_manager):
        """Agent can delegate a task and poll for status updates."""
        started, proceed = mock_manager.configure_slow_execution()

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
        await asyncio.sleep(0)  # Let the task complete

        # Poll again - should be completed
        status_response = await agent.check_status(job_id)
        assert status_response["state"] == "completed"

    @pytest.mark.asyncio
    async def test_agent_parallel_delegates(self, agent, mock_manager):
        """Agent can make multiple parallel delegate calls."""
        mock_manager.configure_success()

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
    async def test_agent_tool_call_interface_matches_mcp_schema(self, agent, mock_manager):
        """Agent tool calls match the MCP tool parameter schemas."""
        mock_manager.configure_success()

        # delegate: required 'prompt', optional 'block'
        response1 = await agent.call_tool("delegate", {"prompt": "test"})
        assert "subagent_id" in response1

        response2 = await agent.call_tool("delegate", {"prompt": "test", "block": True})
        assert "subagent_id" in response2

        # status: required 'subagent_id'
        job_id = response1["subagent_id"]
        response3 = await agent.call_tool("status", {"subagent_id": job_id})
        assert "state" in response3

    @pytest.mark.asyncio
    async def test_agent_tool_call_with_unknown_tool_raises(self, agent):
        """Agent tool call with unknown tool name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await agent.call_tool("unknown_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_agent_tracks_all_tool_calls(self, agent, mock_manager):
        """Agent tracks all tool calls made during a session."""
        mock_manager.configure_success()

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
