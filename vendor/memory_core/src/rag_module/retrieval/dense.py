"""Dense retriever: embedding + vector search."""

from __future__ import annotations

import time

from src.rag_module.embedding.base import EmbeddingProvider
from src.rag_module.retrieval.models import RagCandidate
from src.rag_module.storage.chunk_repo import ChunkRepository
from src.rag_module.storage.vector.base import VectorStore


class DenseRetriever:
    """Retrieve candidates using dense vector search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_repo: ChunkRepository,
    ) -> None:
        self._embedding = embedding_provider
        self._vectors = vector_store
        self._chunks = chunk_repo
        self.last_timings_ms = {"embedding": 0.0, "dense": 0.0}

    def retrieve(
        self,
        query: str,
        project_id: str,
        candidate_k: int = 20,
    ) -> list[RagCandidate]:
        return self.retrieve_many([query], project_id, candidate_k)[0]

    def retrieve_many(
        self,
        queries: list[str],
        project_id: str,
        candidate_k: int = 20,
    ) -> list[list[RagCandidate]]:
        if not queries:
            self.last_timings_ms = {"embedding": 0.0, "dense": 0.0}
            return []
        embedding_started = time.perf_counter()
        query_vectors = self._embedding.embed_queries(queries)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        dense_started = time.perf_counter()
        search_many = getattr(self._vectors, "search_many", None)
        if callable(search_many) and len(query_vectors) > 1:
            hit_batches = search_many(
                query_vectors,
                candidate_k,
                project_id=project_id,
                is_active=True,
            )
            results = [self._hits_to_candidates(hits) for hits in hit_batches]
        else:
            results = [
                self._retrieve_vector(query_vector, project_id, candidate_k)
                for query_vector in query_vectors
            ]
        self.last_timings_ms = {
            "embedding": embedding_ms,
            "dense": (time.perf_counter() - dense_started) * 1000,
        }
        return results

    def _retrieve_vector(
        self,
        query_vector: list[float],
        project_id: str,
        candidate_k: int,
    ) -> list[RagCandidate]:
        hits = self._vectors.search(
            query_vector=query_vector,
            k=candidate_k,
            project_id=project_id,
            is_active=True,
        )
        return self._hits_to_candidates(hits)

    def _hits_to_candidates(self, hits) -> list[RagCandidate]:
        if not hits:
            return []

        chunk_pks = [h.chunk_pk for h in hits]
        chunk_records = self._chunks.get_by_pks(chunk_pks)

        record_map = {r.chunk_pk: r for r in chunk_records}
        candidates: list[RagCandidate] = []
        for hit in hits:
            record = record_map.get(hit.chunk_pk)
            if record is None:
                continue

            # Use parent content if available (Parent-Child chunking)
            content = record.content
            parent_content = None
            if record.metadata:
                parent_content = record.metadata.get("parent_content")

            candidates.append(
                RagCandidate(
                    chunk_pk=record.chunk_pk,
                    chunk_id=record.chunk_id,
                    doc_pk=record.doc_pk,
                    doc_id="",  # filled by pipeline
                    project_id=record.project_id,
                    content=content,
                    heading_path=record.heading_path,
                    page_start=record.page_start,
                    page_end=record.page_end,
                    chunk_index=record.chunk_index,
                    dense_score=1.0
                    - float((hit.metadata or {}).get("cosine_distance", hit.distance)),
                    metadata={
                        **(record.metadata or {}),
                        "parent_content": parent_content,
                    },
                )
            )
        return candidates
