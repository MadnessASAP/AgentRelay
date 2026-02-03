"""Tests for the MCP server interface.

These tests focus on the MCP tool interface (delegate and status tools)
and verify the injection-first invariant: results are delivered via
injection callback, not through tool responses.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_relay.manager import JobManager, JobState, Job


class MockBackend:
    """Mock backend for testing without codex-acp dependency."""

    def __init__(self, result: str = "mock result", delay: float = 0.0, should_fail: bool = False):
        self.result = result
        self.delay = delay
        self.should_fail = should_fail
        self.run_calls = []

    async def run(self, prompt: str) -> str:
        self.run_calls.append(prompt)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("mock backend failure")
        return self.result


class TestDelegateTool:
    """Tests for the delegate tool interface."""

    @pytest.fixture
    def mock_inject(self):
        return AsyncMock()

    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.fixture
    def manager(self, backend, mock_inject):
        return JobManager(backend, mock_inject, max_concurrent=4)

    async def test_delegate_non_blocking_returns_immediately(self, manager):
        """delegate with block=false should return immediately with queued state."""
        job = manager.create_job("test prompt")
        task = asyncio.create_task(manager.run_job(job))

        # Non-blocking response
        result = {"subagent_id": job.id, "state": job.state.value}

        assert "subagent_id" in result
        assert result["state"] == "queued"

        # Clean up the task
        await task

    async def test_delegate_blocking_waits_for_completion(self, manager):
        """delegate with block=true should wait and return completed state."""
        job = manager.create_job("test prompt")
        task = asyncio.create_task(manager.run_job(job))

        # Wait for completion (simulates block=true)
        await task

        result = {"subagent_id": job.id, "state": job.state.value}

        assert result["subagent_id"] == job.id
        assert result["state"] == "completed"

    async def test_delegate_returns_subagent_id(self, manager):
        """delegate should always return a valid subagent_id."""
        job = manager.create_job("test prompt")

        assert job.id is not None
        assert isinstance(job.id, str)
        assert len(job.id) > 0

    async def test_delegate_passes_prompt_to_backend(self, backend, mock_inject):
        """delegate should pass the prompt to the backend."""
        manager = JobManager(backend, mock_inject, max_concurrent=4)
        job = manager.create_job("specific test prompt")

        await manager.run_job(job)

        assert "specific test prompt" in backend.run_calls


class TestStatusTool:
    """Tests for the status tool interface."""

    @pytest.fixture
    def mock_inject(self):
        return AsyncMock()

    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.fixture
    def manager(self, backend, mock_inject):
        return JobManager(backend, mock_inject, max_concurrent=4)

    async def test_status_returns_job_state(self, manager):
        """status should return the current job state."""
        job = manager.create_job("test")

        retrieved = manager.get_job(job.id)
        result = {
            "state": retrieved.state.value,
            "result": retrieved.result,
            "error": retrieved.error,
        }

        assert result["state"] == "queued"

    async def test_status_returns_completed_state_after_job_finishes(self, manager):
        """status should return completed state after job finishes."""
        job = manager.create_job("test")
        await manager.run_job(job)

        retrieved = manager.get_job(job.id)
        result = {
            "state": retrieved.state.value,
            "result": retrieved.result,
            "error": retrieved.error,
        }

        assert result["state"] == "completed"
        assert result["result"] == "mock result"
        assert result["error"] is None

    async def test_status_returns_failed_state_on_error(self, mock_inject):
        """status should return failed state when backend fails."""
        failing_backend = MockBackend(should_fail=True)
        manager = JobManager(failing_backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test")
        await manager.run_job(job)

        retrieved = manager.get_job(job.id)
        result = {
            "state": retrieved.state.value,
            "result": retrieved.result,
            "error": retrieved.error,
        }

        assert result["state"] == "failed"
        assert result["error"] is not None
        assert "mock backend failure" in result["error"]

    async def test_status_raises_on_unknown_job_id(self, manager):
        """status should raise KeyError for unknown job IDs."""
        with pytest.raises(KeyError):
            manager.get_job("nonexistent-job-id")


class TestInjectionInvariant:
    """Tests for the injection-first invariant.

    Core requirement: subagent outputs are NEVER returned via MCP tool
    responses. They are delivered only via the injection callback.
    """

    async def test_injection_called_on_completion(self):
        """Injection callback should be called when job completes."""
        mock_inject = AsyncMock()
        backend = MockBackend(result="test output")
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test prompt")
        await manager.run_job(job)

        mock_inject.assert_called_once()

    async def test_injection_called_on_failure(self):
        """Injection callback should be called when job fails."""
        mock_inject = AsyncMock()
        backend = MockBackend(should_fail=True)
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test prompt")
        await manager.run_job(job)

        mock_inject.assert_called_once()

    async def test_injection_message_contains_subagent_id(self):
        """Injected message should contain the subagent ID."""
        mock_inject = AsyncMock()
        backend = MockBackend(result="test output")
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test prompt")
        await manager.run_job(job)

        injected_message = mock_inject.call_args[0][0]
        assert job.id in injected_message

    async def test_injection_message_contains_result_on_success(self):
        """Injected message should contain result on successful completion."""
        mock_inject = AsyncMock()
        backend = MockBackend(result="expected output value")
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test prompt")
        await manager.run_job(job)

        injected_message = mock_inject.call_args[0][0]
        assert "expected output value" in injected_message
        assert "completed" in injected_message.lower()

    async def test_injection_message_contains_error_on_failure(self):
        """Injected message should contain error on failure."""
        mock_inject = AsyncMock()
        backend = MockBackend(should_fail=True)
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test prompt")
        await manager.run_job(job)

        injected_message = mock_inject.call_args[0][0]
        assert "failed" in injected_message.lower()
        assert "mock backend failure" in injected_message


class TestConcurrency:
    """Tests for concurrency control via semaphore."""

    async def test_concurrent_jobs_limited_by_semaphore(self):
        """Jobs should be limited by max_concurrent setting."""
        mock_inject = AsyncMock()
        backend = MockBackend(delay=0.1)
        max_concurrent = 2
        manager = JobManager(backend, mock_inject, max_concurrent=max_concurrent)

        # Create more jobs than max_concurrent
        jobs = [manager.create_job(f"prompt {i}") for i in range(4)]
        tasks = [asyncio.create_task(manager.run_job(job)) for job in jobs]

        # Wait a bit for some jobs to start
        await asyncio.sleep(0.05)

        # Count running jobs
        running_count = sum(1 for job in jobs if job.state == JobState.RUNNING)

        # Should not exceed max_concurrent
        assert running_count <= max_concurrent

        # Clean up
        await asyncio.gather(*tasks)

    async def test_all_jobs_eventually_complete(self):
        """All queued jobs should eventually complete."""
        mock_inject = AsyncMock()
        backend = MockBackend(delay=0.01)
        manager = JobManager(backend, mock_inject, max_concurrent=2)

        jobs = [manager.create_job(f"prompt {i}") for i in range(4)]
        tasks = [asyncio.create_task(manager.run_job(job)) for job in jobs]

        await asyncio.gather(*tasks)

        for job in jobs:
            assert job.state == JobState.COMPLETED


class TestJobStateTransitions:
    """Tests for job state transitions."""

    async def test_job_starts_in_queued_state(self):
        """New jobs should start in QUEUED state."""
        mock_inject = AsyncMock()
        backend = MockBackend()
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test")

        assert job.state == JobState.QUEUED

    async def test_job_transitions_to_running(self):
        """Job should transition to RUNNING when execution starts."""
        mock_inject = AsyncMock()
        backend = MockBackend(delay=0.1)
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test")
        task = asyncio.create_task(manager.run_job(job))

        # Wait a bit for job to start
        await asyncio.sleep(0.01)

        assert job.state == JobState.RUNNING

        await task

    async def test_job_transitions_to_completed_on_success(self):
        """Job should transition to COMPLETED on successful execution."""
        mock_inject = AsyncMock()
        backend = MockBackend()
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test")
        await manager.run_job(job)

        assert job.state == JobState.COMPLETED

    async def test_job_transitions_to_failed_on_error(self):
        """Job should transition to FAILED when backend raises."""
        mock_inject = AsyncMock()
        backend = MockBackend(should_fail=True)
        manager = JobManager(backend, mock_inject, max_concurrent=4)

        job = manager.create_job("test")
        await manager.run_job(job)

        assert job.state == JobState.FAILED
