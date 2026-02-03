"""Tests for MCP server tool responses.

These tests verify the immediate responses returned by the `delegate` and `status` tools,
testing what a tool caller would see. Only server.py is tested - the manager is mocked.
"""

import asyncio
import pytest

from agent_relay.handlers import create_tool_handlers
from tests.mock import MockJobState, MockManager


@pytest.fixture
def mock_manager():
    """Create a fresh MockManager for each test."""
    return MockManager()


@pytest.fixture
def handlers(mock_manager):
    """Create tool handlers with the mock manager."""
    return create_tool_handlers(mock_manager)


class TestDelegateTool:
    """Tests for the delegate tool responses."""

    @pytest.mark.asyncio
    async def test_delegate_non_blocking_returns_queued_state(self, handlers):
        """Non-blocking delegate should return immediately with state='queued'."""
        delegate = handlers["delegate"]

        response = await delegate(prompt="test prompt", block=False)

        assert response["state"] == "queued"
        assert "subagent_id" in response

    @pytest.mark.asyncio
    async def test_delegate_non_blocking_response_structure(self, handlers):
        """Non-blocking delegate response should have exactly the expected keys."""
        delegate = handlers["delegate"]

        response = await delegate(prompt="test prompt", block=False)

        # Response should only contain subagent_id and state (no result content)
        assert set(response.keys()) == {"subagent_id", "state"}

    @pytest.mark.asyncio
    async def test_delegate_non_blocking_returns_unique_ids(self, handlers):
        """Each non-blocking delegate call should return a unique subagent_id."""
        delegate = handlers["delegate"]

        response1 = await delegate(prompt="prompt 1", block=False)
        response2 = await delegate(prompt="prompt 2", block=False)
        response3 = await delegate(prompt="prompt 3", block=False)

        ids = [response1["subagent_id"], response2["subagent_id"], response3["subagent_id"]]
        assert len(set(ids)) == 3  # All unique

    @pytest.mark.asyncio
    async def test_delegate_blocking_returns_completed_state(self, handlers, mock_manager):
        """Blocking delegate should wait and return state='completed' on success."""
        delegate = handlers["delegate"]
        mock_manager.configure_success(result="task completed")

        response = await delegate(prompt="test prompt", block=True)

        assert response["state"] == "completed"
        assert "subagent_id" in response

    @pytest.mark.asyncio
    async def test_delegate_blocking_returns_failed_state_on_error(self, handlers, mock_manager):
        """Blocking delegate should return state='failed' when job fails."""
        delegate = handlers["delegate"]
        mock_manager.configure_failure(error="something went wrong")

        response = await delegate(prompt="test prompt", block=True)

        assert response["state"] == "failed"

    @pytest.mark.asyncio
    async def test_delegate_blocking_response_structure(self, handlers, mock_manager):
        """Blocking delegate response should have exactly the expected keys."""
        delegate = handlers["delegate"]
        mock_manager.configure_success()

        response = await delegate(prompt="test prompt", block=True)

        # Response should only contain subagent_id and state (no result content)
        assert set(response.keys()) == {"subagent_id", "state"}

    @pytest.mark.asyncio
    async def test_delegate_blocking_response_does_not_contain_result(self, handlers, mock_manager):
        """Blocking delegate response should NOT contain the actual result (injection-first invariant)."""
        delegate = handlers["delegate"]
        mock_manager.configure_success(result="this should not appear in response")

        response = await delegate(prompt="test prompt", block=True)

        # Result is NOT in the tool response (delivered via injection instead)
        assert "result" not in response
        assert "output" not in response
        assert "content" not in response

    @pytest.mark.asyncio
    async def test_delegate_default_block_is_false(self, handlers):
        """Delegate should default to non-blocking (block=False)."""
        delegate = handlers["delegate"]

        # Call without block parameter - should behave like block=False
        response = await delegate(prompt="test prompt")

        # Non-blocking returns queued immediately
        assert response["state"] == "queued"


class TestStatusTool:
    """Tests for the status tool responses."""

    @pytest.mark.asyncio
    async def test_status_returns_queued_for_new_job(self, handlers, mock_manager):
        """Status should return 'queued' for a newly created job."""
        delegate = handlers["delegate"]
        status = handlers["status"]

        # Create a job via delegate (non-blocking)
        delegate_response = await delegate(prompt="test prompt", block=False)
        job_id = delegate_response["subagent_id"]

        # Immediately check status (job still queued since run_job runs in background)
        # Note: We need to get the job before run_job changes its state
        mock_manager.jobs[job_id].state = MockJobState.QUEUED  # Ensure queued state
        response = await status(subagent_id=job_id)

        assert response["state"] == "queued"
        assert response["result"] is None
        assert response["error"] is None

    @pytest.mark.asyncio
    async def test_status_returns_completed_after_success(self, handlers, mock_manager):
        """Status should return 'completed' and result after successful job."""
        delegate = handlers["delegate"]
        status = handlers["status"]
        mock_manager.configure_success(result="the result")

        # Create and complete a job
        delegate_response = await delegate(prompt="test prompt", block=True)
        job_id = delegate_response["subagent_id"]

        response = await status(subagent_id=job_id)

        assert response["state"] == "completed"
        assert response["result"] == "the result"
        assert response["error"] is None

    @pytest.mark.asyncio
    async def test_status_returns_failed_after_error(self, handlers, mock_manager):
        """Status should return 'failed' and error after job failure."""
        delegate = handlers["delegate"]
        status = handlers["status"]
        mock_manager.configure_failure(error="something went wrong")

        # Create and fail a job
        delegate_response = await delegate(prompt="test prompt", block=True)
        job_id = delegate_response["subagent_id"]

        response = await status(subagent_id=job_id)

        assert response["state"] == "failed"
        assert response["result"] is None
        assert response["error"] == "something went wrong"

    @pytest.mark.asyncio
    async def test_status_response_structure(self, handlers, mock_manager):
        """Status response should have exactly the expected keys."""
        delegate = handlers["delegate"]
        status = handlers["status"]

        delegate_response = await delegate(prompt="test prompt", block=False)
        job_id = delegate_response["subagent_id"]

        # Wait for job to complete
        await asyncio.sleep(0)  # Let the background task run

        response = await status(subagent_id=job_id)

        assert set(response.keys()) == {"state", "result", "error"}

    @pytest.mark.asyncio
    async def test_status_unknown_job_raises_key_error(self, handlers):
        """Status for unknown subagent_id should raise KeyError."""
        status = handlers["status"]

        with pytest.raises(KeyError):
            await status(subagent_id="nonexistent-job-id")

    @pytest.mark.asyncio
    async def test_status_returns_running_during_execution(self, handlers, mock_manager):
        """Status should return 'running' while job is executing."""
        delegate = handlers["delegate"]
        status = handlers["status"]

        # Configure slow execution so we can check state during run
        started, proceed = mock_manager.configure_slow_execution()

        # Start a non-blocking delegate
        delegate_response = await delegate(prompt="test prompt", block=False)
        job_id = delegate_response["subagent_id"]

        # Wait for job to start running
        await started.wait()

        # Check status while running
        response = await status(subagent_id=job_id)
        assert response["state"] == "running"

        # Allow job to complete
        proceed.set()
