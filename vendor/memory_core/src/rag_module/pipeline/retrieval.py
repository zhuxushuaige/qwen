"""Retrieval pipeline: dense + sparse → fusion → rerank → context assembly."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from src.agent_orchestrator.types import RagResult
from src.rag_module.context.assembler import ContextAssembler
from src.rag_module.embedding.base import EmbeddingProvider
from src.rag_module.rerank.base import Reranker
from src.rag_module.rerank.noop_reranker import NoopReranker
from src.rag_module.rerank.policy import RerankPolicy
from src.rag_module.retrieval.dense import DenseRetriever
from src.rag_module.retrieval.fusion import rrf_fuse, rrf_fuse_many
from src.rag_module.retrieval.models import RagCandidate, RagResponse, RetrievalDiagnostics
from src.rag_module.retrieval.query_decomposition import QueryDecomposer
from src.rag_module.retrieval.query_expansion import QueryExpander
from src.rag_module.retrieval.rejection import RejectionConfig, RejectionPolicy
from src.rag_module.retrieval.sparse import SparseRetriever
from src.rag_module.storage.chunk_repo import ChunkRepository
from src.rag_module.storage.document_repo import DocumentRepository
from src.rag_module.storage.vector.base import VectorStore
from src.rag_module.storage.vector.factory import close_vector_store


class RetrievalPipeline:
    """End-to-end retrieval pipeline implementing RagRetriever protocol."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_repo: ChunkRepository,
        doc_repo: DocumentRepository,
        conn: sqlite3.Connection | None = None,
        reranker: Reranker | None = None,
        rerank_policy: RerankPolicy | None = None,
        context_assembler: ContextAssembler | None = None,
        query_expander: QueryExpander | None = None,
        rejection_config: RejectionConfig | None = None,
        query_decomposer: QueryDecomposer | None = None,
        dense_candidate_k: int = 20,
        sparse_candidate_k: int = 20,
        fusion_candidate_k: int = 20,
        rerank_candidate_k: int = 15,
        final_top_k: int = 5,
    ) -> None:
        if rerank_candidate_k <= 0:
            raise ValueError("rerank_candidate_k must be positive")
        self._dense = DenseRetriever(embedding_provider, vector_store, chunk_repo)
        self._vector_store = vector_store
        self._conn = conn
        self._sparse = SparseRetriever(conn) if conn else None
        self._reranker: Reranker = reranker or NoopReranker()
        self._policy = rerank_policy or RerankPolicy()
        self._assembler = context_assembler or ContextAssembler()
        self._query_expander = query_expander
        self._rejection = RejectionPolicy(rejection_config) if rejection_config else None
        self._query_decomposer = query_decomposer or QueryDecomposer()
        self._chunk_repo = chunk_repo
        self._doc_repo = doc_repo
        self._dense_candidate_k = dense_candidate_k
        self._sparse_candidate_k = sparse_candidate_k
        self._fusion_candidate_k = fusion_candidate_k
        self._rerank_candidate_k = rerank_candidate_k
        self._final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        project_id: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RagResponse:
        """Full retrieval pipeline with hybrid search."""
        total_started = time.perf_counter()
        final_k = top_k or self._final_top_k
        warnings: list[str] = []
        diagnostics = RetrievalDiagnostics()

        query_variants = self._query_decomposer.decompose(query) or [query]

        # Add optional LLM-generated variants without losing deterministic
        # subqueries produced by the decomposer.
        if self._query_expander:
            try:
                expanded_queries = self._query_expander.expand(query)
                query_variants = list(dict.fromkeys([*query_variants, *expanded_queries]))
            except Exception:
                warnings.append("query_expansion_failed")

        diagnostics.subqueries = query_variants[1:]
        dense_results = rrf_fuse_many(
            self._dense.retrieve_many(
                query_variants,
                project_id=project_id,
                candidate_k=self._dense_candidate_k,
            ),
            self._dense_candidate_k,
        )
        timings = self._dense.last_timings_ms
        diagnostics.timings_ms["T_embedding"] = timings["embedding"]
        diagnostics.timings_ms["T_dense"] = timings["dense"]
        diagnostics.dense_count = len(dense_results)

        sparse_results: list[RagCandidate] = []
        sparse_started = time.perf_counter()
        if self._sparse:
            try:
                sparse_results = rrf_fuse_many(
                    [
                        self._sparse.retrieve(
                            query=value,
                            project_id=project_id,
                            candidate_k=self._sparse_candidate_k,
                        )
                        for value in query_variants
                    ],
                    self._sparse_candidate_k,
                )
                diagnostics.sparse_count = len(sparse_results)
            except Exception:
                warnings.append("sparse_retrieval_failed")
        diagnostics.timings_ms["T_sparse"] = (time.perf_counter() - sparse_started) * 1000

        fusion_started = time.perf_counter()
        if sparse_results:
            candidates = rrf_fuse(dense_results, sparse_results, self._fusion_candidate_k)
            diagnostics.fusion_count = len(candidates)
        else:
            candidates = dense_results[: self._fusion_candidate_k]
            diagnostics.fusion_count = len(candidates)
        diagnostics.timings_ms["T_fusion"] = (time.perf_counter() - fusion_started) * 1000

        # Resolve provenance before reranking so evaluation can distinguish a
        # first-stage recall miss from a reranker regression.
        for candidate in candidates:
            doc = self._doc_repo.get_by_pk(candidate.doc_pk)
            if doc:
                candidate.doc_id = doc.doc_id
                candidate.source_uri = doc.source_uri
        diagnostics.pre_rerank_candidates = [
            _candidate_diagnostic(candidate, rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]

        if self._policy.should_rerank(final_k, len(candidates)):
            if self._rerank_candidate_k < final_k:
                raise ValueError("rerank_candidate_k cannot be smaller than requested top_k")
            rerank_started = time.perf_counter()
            rerank_candidates = candidates[: self._rerank_candidate_k]
            candidates = self._reranker.rerank(
                query,
                rerank_candidates,
                self._rerank_candidate_k,
            )
            rerank_elapsed_ms = (time.perf_counter() - rerank_started) * 1000
            reranker_timings = getattr(self._reranker, "last_timings_ms", {})
            tokenization_ms = min(
                float(reranker_timings.get("tokenization", 0.0)),
                rerank_elapsed_ms,
            )
            diagnostics.timings_ms["T_tokenization"] = tokenization_ms
            diagnostics.timings_ms["T_reranker"] = rerank_elapsed_ms - tokenization_ms
            diagnostics.rerank_count = len(candidates)
        else:
            candidates = candidates[:self._fusion_candidate_k]
            diagnostics.timings_ms["T_tokenization"] = 0.0
            diagnostics.timings_ms["T_reranker"] = 0.0

        diagnostics.post_rerank_candidates = [
            _candidate_diagnostic(candidate, rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]

        # Apply rejection policy
        top1_has_sparse_hit = bool(candidates and candidates[0].sparse_score is not None)
        if self._rejection and self._rejection.should_reject(candidates, top1_has_sparse_hit):
            diagnostics.timings_ms["T_total"] = (time.perf_counter() - total_started) * 1000
            return RagResponse(
                results=[],
                warnings=warnings + ["rejected_as_unanswerable"],
                diagnostics=diagnostics,
            )

        assembled = self._assembler.assemble(candidates, top_k=final_k)
        warnings.extend(assembled.warnings)
        diagnostics.timings_ms["T_total"] = (time.perf_counter() - total_started) * 1000

        return RagResponse(
            results=assembled.candidates,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RagResult]:
        """RagRetriever protocol compatible interface."""
        response = self.retrieve(query, project_id, top_k, filters)
        return [_to_rag_result(c) for c in response.results]

    def close(self) -> None:
        """Release a persistent Kylin bridge and the owned SQLite connection."""

        close_vector_store(self._vector_store)
        if self._conn is not None:
            self._conn.close()


def _to_rag_result(c: RagCandidate) -> RagResult:
    score = c.rerank_score or c.fusion_score or c.dense_score or 0.0
    return RagResult(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        content=c.content,
        score=score,
        source=c.source_uri or "rag",
        page=c.page_start,
        metadata=c.metadata,
    )


def _candidate_diagnostic(candidate: RagCandidate, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": candidate.chunk_id,
        "doc_id": candidate.doc_id,
        "source": candidate.source_uri,
        "dense_score": candidate.dense_score,
        "sparse_score": candidate.sparse_score,
        "fusion_score": candidate.fusion_score,
        "rerank_score": candidate.rerank_score,
        "content": candidate.content,
    }
