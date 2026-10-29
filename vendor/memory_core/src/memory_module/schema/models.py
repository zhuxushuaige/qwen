"""Data model helpers for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ACTIVE = "active"
FORGOTTEN = "forgotten"
SUPERSEDED = "superseded"

DEFAULT_MEMORY_TYPE = "project_memory"
PREFERENCE_MEMORY = "preference"
TASK_MEMORY = "task_memory"
PROJECT_MEMORY = "project_memory"


@dataclass
class MemoryCandidate:
    """Structured atomic memory produced by an extraction model."""

    content: str
    type: str = DEFAULT_MEMORY_TYPE
    subject: str = ""
    attribute: str = ""
    value: str = ""
    polarity: str = "positive"
    scope: str = "project"
    confidence: float = 1.0
    source: str = "memory_extractor"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """Internal persisted memory representation."""

    memory_id: str
    user_id: str
    project_id: str
    content: str
    type: str = DEFAULT_MEMORY_TYPE
    status: str = ACTIVE
    source: str = "memory_writer"
    subject: str = ""
    attribute: str = ""
    value: str = ""
    polarity: str = "positive"
    scope: str = "project"
    confidence: float = 1.0
    version: int = 1
    supersedes_id: str | None = None
    superseded_by_id: str | None = None
    effective_at: str = field(default_factory=lambda: utc_now_iso())
    created_at: str = field(default_factory=lambda: utc_now_iso())
    updated_at: str = field(default_factory=lambda: utc_now_iso())
    expiration_date: str | None = None
    attributed_to: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp without relying on local timezone."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
