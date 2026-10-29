"""Fake RAG retriever for orchestration tests and early integration."""

from __future__ import annotations

from typing import Any

from src.contracts.agent import RagResult


class FakeRagRetriever:
    """Return predictable document chunks until the real RAG module is ready."""

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RagResult]:
        return [
            RagResult(
                chunk_id="chunk_demo_001",
                doc_id="doc_competition_demo",
                content="赛题要求系统支持偏好记忆、知识记忆、高效检索和精准遗忘。",
                score=0.88,
                source="fake_rag",
                page=1,
            )
        ][:top_k]
