"""Chunk data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A document chunk ready for embedding."""

    chunk_id: str
    doc_id: str
    project_id: str
    chunk_index: int
    content: str
    heading_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int | None = None
    content_hash: str = ""
    parent_chunk_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
