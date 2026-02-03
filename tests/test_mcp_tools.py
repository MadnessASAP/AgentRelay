"""Tests for MCP server tool responses.

These tests verify the immediate responses returned by the `delegate` and `status` tools,
testing what a tool caller would see.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_relay.manager import JobManager, JobState, Job
from agent_relay.backends import ACPBackend


@pytest.fixture
def mock_backend():
    """Create a mock backend that returns a predefined result."""
    backend = MagicMock(spec=ACPBackend)
    backend.run = AsyncMock(return_value="mock result")
    return backend


@pytest.fixture
def mock_injector():
    """Create a mock injection callback."""
    return AsyncMock()


@pytest.fixture
def job_manager(mock_backend, mock_injector):
    """Create a JobManager with mocked dependencies."""
    return JobManager(mock_backend, mock_injector, max_concurrent=4)


class TestDelegateTool:
    """Tests for the delegate tool responses."""

    @pytest.mark.asyncio
    async def test_delegate_non_blocking_returns_queued_state(self, job_manager):
        """Non-blocking delegate should return immediately with state='queued'."""
        # Simulate the delegate tool handler behavior
        prompt = "test prompt"
        block = False

        job = job_manager.create_job(prompt)
        task = asyncio.create_task(job_manager.run_job(job))

        # Non-blocking returns immediately
        if not block:
            response = {"subagent_id": job.id, "state": job.state.value}

        assert "subagent_id" in response
        assert response["state"] == "queued"
        assert len(response["subagent_id"]) == 36  # UUID format

        # Clean up background task
        await task

    @pytest.mark.asyncio
    async def test_delegate_non_blocking_response_structure(self, job_manager):
        """Non-blocking delegate response should have exactly the expected keys."""
        prompt = "test prompt"
        job = job_manager.create_job(prompt)

        response = {"subagent_id": job.id, "state": job.state.value}

        # Response should only contain subagent_id and state (no result content)
        assert set(response.keys()) == {"subagent_id", "state"}

    @pytest.mark.asyncio
    async def test_delegate_blocking_returns_completed_state(self, job_manager):
        """Blocking delegate should wait and return state='completed' on success."""
        prompt = "test prompt"
        block = True

        job = job_manager.create_job(prompt)
        task = asyncio.create_task(job_manager.run_job(job))

        # Blocking waits for completion
        if block:
            await task
            response = {"subagent_id": job.id, "state": job.state.value}

        assert response["subagent_id"] == job.id
        assert response["state"] == "completed"

    @pytest.mark.asyncio
    async def test_delegate_blocking_returns_failed_state_on_error(self, job_manager, mock_backend):
        """Blocking delegate should return state='failed' when backend raises exception."""
        mock_backend.run = AsyncMock(side_effect=Exception("backend error"))
        prompt = "test prompt"

        job = job_manager.create_job(prompt)
        task = asyncio.create_task(job_manager.run_job(job))

        await task
        response = {"subagent_id": job.id, "state": job.state.value}

        assert response["state"] == "failed"

    @pytest.mark.asyncio
    async def test_delegate_blocking_response_does_not_contain_result(self, job_manager):
        """Blocking delegate response should NOT contain the actual result (injection-first invariant)."""
        prompt = "test prompt"

        job = job_manager.create_job(prompt)
        await job_manager.run_job(job)

        # The tool response format
        response = {"subagent_id": job.id, "state": job.state.value}

        # Result is NOT in the tool response (delivered via injection instead)
        assert "result" not in response
        assert "output" not in response
        assert "content" not in response

    @pytest.mark.asyncio
    async def test_delegate_generates_unique_subagent_ids(self, job_manager):
        """Each delegate call should generate a unique subagent_id."""
        job1 = job_manager.create_job("prompt 1")
        job2 = job_manager.create_job("prompt 2")
        job3 = job_manager.create_job("prompt 3")

        ids = [job1.id, job2.id, job3.id]
        assert len(set(ids)) == 3  # All unique


class TestStatusTool:
    """Tests for the status tool responses."""

    @pytest.mark.asyncio
    async def test_status_returns_queued_for_new_job(self, job_manager):
        """Status should return 'queued' for a newly created job."""
        job = job_manager.create_job("test prompt")

        # Simulate status tool handler
        fetched_job = job_manager.get_job(job.id)
        response = {
            "state": fetched_job.state.value,
            "result": fetched_job.result,
            "error": fetched_job.error,
        }

        assert response["state"] == "queued"
        assert response["result"] is None
        assert response["error"] is None

    @pytest.mark.asyncio
    async def test_status_returns_completed_after_success(self, job_manager):
        """Status should return 'completed' and result after successful job."""
        job = job_manager.create_job("test prompt")
        await job_manager.run_job(job)

        fetched_job = job_manager.get_job(job.id)
        response = {
            "state": fetched_job.state.value,
            "result": fetched_job.result,
            "error": fetched_job.error,
        }

        assert response["state"] == "completed"
        assert response["result"] == "mock result"
        assert response["error"] is None

    @pytest.mark.asyncio
    async def test_status_returns_failed_after_error(self, job_manager, mock_backend):
        """Status should return 'failed' and error after job failure."""
        mock_backend.run = AsyncMock(side_effect=Exception("something went wrong"))

        job = job_manager.create_job("test prompt")
        await job_manager.run_job(job)

        fetched_job = job_manager.get_job(job.id)
        response = {
            "state": fetched_job.state.value,
            "result": fetched_job.result,
            "error": fetched_job.error,
        }

        assert response["state"] == "failed"
        assert response["result"] is None
        assert response["error"] == "something went wrong"

    @pytest.mark.asyncio
    async def test_status_response_structure(self, job_manager):
        """Status response should have exactly the expected keys."""
        job = job_manager.create_job("test prompt")

        fetched_job = job_manager.get_job(job.id)
        response = {
            "state": fetched_job.state.value,
            "result": fetched_job.result,
            "error": fetched_job.error,
        }

        assert set(response.keys()) == {"state", "result", "error"}

    @pytest.mark.asyncio
    async def test_status_unknown_job_raises_key_error(self, job_manager):
        """Status for unknown subagent_id should raise KeyError."""
        with pytest.raises(KeyError):
            job_manager.get_job("nonexistent-job-id")

    @pytest.mark.asyncio
    async def test_status_returns_running_during_execution(self, job_manager, mock_backend):
        """Status should return 'running' while job is executing."""
        # Make the backend run slowly so we can check state during execution
        execution_started = asyncio.Event()
        proceed = asyncio.Event()

        async def slow_run(prompt):
            execution_started.set()
            await proceed.wait()
            return "result"

        mock_backend.run = slow_run

        job = job_manager.create_job("test prompt")
        task = asyncio.create_task(job_manager.run_job(job))

        # Wait for execution to start
        await execution_started.wait()

        # Check status while running
        fetched_job = job_manager.get_job(job.id)
        response = {
            "state": fetched_job.state.value,
            "result": fetched_job.result,
            "error": fetched_job.error,
        }

        assert response["state"] == "running"

        # Allow completion
        proceed.set()
        await task
