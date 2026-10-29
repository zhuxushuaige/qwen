"""Reciprocal Rank Fusion (RRF) for merging retrieval results."""

from __future__ import annotations

from src.rag_module.retrieval.models import RagCandidate

RRF_K = 60


def rrf_fuse(
    dense_results: list[RagCandidate],
    sparse_results: list[RagCandidate],
    candidate_k: int = 20,
) -> list[RagCandidate]:
    """Merge dense and sparse results using Reciprocal Rank Fusion.

    rrf_score = 1 / (k + rank)
    """
    scores: dict[int, float] = {}
    pk_to_candidate: dict[int, RagCandidate] = {}

    for rank, c in enumerate(dense_results):
        scores[c.chunk_pk] = scores.get(c.chunk_pk, 0.0) + 1.0 / (RRF_K + rank)
        pk_to_candidate[c.chunk_pk] = c

    for rank, c in enumerate(sparse_results):
        scores[c.chunk_pk] = scores.get(c.chunk_pk, 0.0) + 1.0 / (RRF_K + rank)
        if c.chunk_pk not in pk_to_candidate:
            pk_to_candidate[c.chunk_pk] = c
        else:
            pk_to_candidate[c.chunk_pk].sparse_score = c.sparse_score

    sorted_pks = sorted(scores.keys(), key=lambda pk: scores[pk], reverse=True)

    results: list[RagCandidate] = []
    for pk in sorted_pks[:candidate_k]:
        candidate = pk_to_candidate[pk]
        candidate.fusion_score = scores[pk]
        results.append(candidate)

    return results


def rrf_fuse_many(
    ranked_lists: list[list[RagCandidate]],
    candidate_k: int = 20,
) -> list[RagCandidate]:
    """Fuse multiple query variants while preserving candidate stage scores."""

    scores: dict[int, float] = {}
    pk_to_candidate: dict[int, RagCandidate] = {}
    for values in ranked_lists:
        for rank, candidate in enumerate(values):
            scores[candidate.chunk_pk] = scores.get(candidate.chunk_pk, 0.0) + 1.0 / (
                RRF_K + rank
            )
            existing = pk_to_candidate.setdefault(candidate.chunk_pk, candidate)
            if candidate.dense_score is not None and (
                existing.dense_score is None or candidate.dense_score > existing.dense_score
            ):
                existing.dense_score = candidate.dense_score
            if candidate.sparse_score is not None and (
                existing.sparse_score is None or candidate.sparse_score > existing.sparse_score
            ):
                existing.sparse_score = candidate.sparse_score
    sorted_pks = sorted(scores, key=lambda pk: scores[pk], reverse=True)
    result = []
    for pk in sorted_pks[:candidate_k]:
        candidate = pk_to_candidate[pk]
        candidate.metadata["query_fusion_score"] = scores[pk]
        result.append(candidate)
    return result
