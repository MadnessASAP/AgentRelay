import asyncio
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Awaitable


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    prompt: str
    state: JobState = JobState.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None


class JobManager:
    """
    In-memory job manager with injection-first result delivery.

    Core invariant:
      - Subagent outputs are *never* returned via MCP tool responses.
      - Outputs are delivered only via the injection callback.
    """

    def __init__(
        self,
        backend,
        inject_message: Callable[[str], Awaitable[None]],
        max_concurrent: int = 4,
    ):
        self.backend = backend
        self.inject_message = inject_message
        self.jobs: Dict[str, Job] = {}
        self.sem = asyncio.Semaphore(max_concurrent)

    def create_job(self, prompt: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, prompt=prompt)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job:
        return self.jobs[job_id]

    async def run_job(self, job: Job):
        async with self.sem:
            job.state = JobState.RUNNING
            try:
                result = await self.backend.run(job.prompt)
                job.state = JobState.COMPLETED
                job.result = result
                await self._inject(job)
            except Exception as e:
                job.state = JobState.FAILED
                job.error = str(e)
                await self._inject(job)

    async def _inject(self, job: Job):
        # Single canonical injected message format.
        if job.state == JobState.COMPLETED:
            msg = (
                f"[Subagent {job.id} completed]\n"
                f"Result:\n{job.result}"
            )
        else:
            msg = (
                f"[Subagent {job.id} failed]\n"
                f"Error:\n{job.error}"
            )
        await self.inject_message(msg)
