"""Reranker protocol."""

from __future__ import annotations

from typing import Protocol

from src.rag_module.retrieval.models import RagCandidate


class Reranker(Protocol):
    """Rerank retrieval candidates."""

    def rerank(
        self,
        query: str,
        candidates: list[RagCandidate],
        top_k: int,
    ) -> list[RagCandidate]:
        """Rerank candidates and return top_k results."""
        ...
