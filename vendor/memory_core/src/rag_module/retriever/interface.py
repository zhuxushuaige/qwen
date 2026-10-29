"""Interface contract for RAG retrievers used by Agent Orchestrator."""

from __future__ import annotations

from typing import Any, Protocol

from src.contracts.agent import RagResult


class RagRetriever(Protocol):
    """Search external project documents and knowledge chunks."""

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RagResult]:
        """Return document chunks relevant to the query."""
