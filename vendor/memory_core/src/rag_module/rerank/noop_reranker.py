"""No-op reranker: pass through candidates unchanged."""

from __future__ import annotations

from src.rag_module.retrieval.models import RagCandidate


class NoopReranker:
    """Return candidates as-is, sorted by fusion_score or dense_score."""

    def rerank(
        self,
        query: str,
        candidates: list[RagCandidate],
        top_k: int,
    ) -> list[RagCandidate]:
        def sort_key(c: RagCandidate) -> float:
            if c.fusion_score is not None:
                return c.fusion_score
            if c.rerank_score is not None:
                return c.rerank_score
            if c.dense_score is not None:
                return c.dense_score
            return 0.0

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        return sorted_candidates[:top_k]
