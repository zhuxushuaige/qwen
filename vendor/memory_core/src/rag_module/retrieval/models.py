"""Retrieval result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RagCandidate:
    """A candidate result from retrieval, with multi-stage scores."""

    chunk_pk: int
    chunk_id: str
    doc_pk: int
    doc_id: str
    project_id: str
    content: str
    heading_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    source_uri: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RetrievalDiagnostics:
    """Diagnostic info from a retrieval run."""

    dense_count: int = 0
    sparse_count: int = 0
    fusion_count: int = 0
    rerank_count: int = 0
    subqueries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    pre_rerank_candidates: list[dict[str, object]] = field(default_factory=list)
    post_rerank_candidates: list[dict[str, object]] = field(default_factory=list)


@dataclass
class RagResponse:
    """Final retrieval response."""

    results: list[RagCandidate]
    warnings: list[str] = field(default_factory=list)
    diagnostics: RetrievalDiagnostics | None = None
