"""Mock manager for testing."""

import asyncio
from typing import Dict, Optional

from tests.mock.mock_job import MockJob, MockJobState


class MockManager:
    """Mock manager for testing MCP server tool calls in isolation.

    Provides configurable behavior for testing different scenarios
    without involving the real JobManager, backend, or injection logic.
    """

    def __init__(self):
        self.jobs: Dict[str, MockJob] = {}
        self._job_counter = 0
        self._run_job_behavior = "complete"  # "complete", "fail", or "hang"
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

        # Signal that job has started (for testing running state)
        if self._run_job_started:
            self._run_job_started.set()

        # Wait for permission to proceed (for testing running state)
        if self._run_job_proceed:
            await self._run_job_proceed.wait()

        if self._run_job_behavior == "complete":
            job.state = MockJobState.COMPLETED
            job.result = self._run_job_result
        elif self._run_job_behavior == "fail":
            job.state = MockJobState.FAILED
            job.error = self._run_job_error
        # "hang" behavior: never completes (for timeout testing)

    def configure_success(self, result: str = "mock result"):
        """Configure run_job to complete successfully."""
        self._run_job_behavior = "complete"
        self._run_job_result = result

    def configure_failure(self, error: str = "mock error"):
        """Configure run_job to fail with an error."""
        self._run_job_behavior = "fail"
        self._run_job_error = error

    def configure_slow_execution(self):
        """Configure run_job to wait for explicit signal before completing.

        Returns (started_event, proceed_event) to control execution flow.
        """
        self._run_job_started = asyncio.Event()
        self._run_job_proceed = asyncio.Event()
        return self._run_job_started, self._run_job_proceed
