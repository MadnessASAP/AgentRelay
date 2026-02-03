"""Mock job classes for testing."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
