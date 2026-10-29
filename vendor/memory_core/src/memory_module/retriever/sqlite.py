"""SQLite memory retriever with pluggable vector embeddings."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from src.agent_orchestrator.types import MemoryResult
from src.memory_module.retriever.temporal import NoDecay, TemporalDecay
from src.memory_module.schema.fts_store import FTSMemoryStore
from src.memory_module.schema.layered_store import LayeredMemoryStore
from src.memory_module.schema.models import ACTIVE, MemoryRecord, utc_now_iso
from src.memory_module.schema.sqlite_store import SQLiteMemoryStore
from src.memory_module.schema.vector_store import (
    SQLiteMemoryVectorStore,
    distance_to_cosine_similarity,
)
from src.rag_module.embedding.base import EmbeddingInfo, EmbeddingProvider

DEFAULT_EMBEDDING_DIMENSION = 1024


class FeatureHashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free vectorizer used until a semantic model is configured.

    It hashes words and Chinese character n-grams into a fixed-size normalized
    vector. It is retained for deterministic tests only; production defaults
    to the Kylin embedding SDK.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self._dimension = dimension

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="local",
            model_id=f"feature-hash-{self._dimension}",
            dimension=self._dimension,
            normalized=True,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for feature in _features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % self._dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class SQLiteMemoryRetriever:
    """Retrieve scoped memories by cosine similarity of content vectors."""

    def __init__(
        self,
        db_path: str | Path = ".memory_module.sqlite3",
        embedding_provider: EmbeddingProvider | None = None,
        temporal_decay: TemporalDecay | None = None,
        vector_store_backend: str | None = None,
        vector_store_fallback: str | None = None,
        drop_vector_collection_on_close: bool = False,
        vector_collection_identity: str | None = None,
    ) -> None:
        self.store = SQLiteMemoryStore(db_path)
        self.layered_store = LayeredMemoryStore(db_path)
        if embedding_provider is None:
            from src.rag_module.embedding.registry import (
                create_default_embedding_provider,
            )

            embedding_provider = create_default_embedding_provider()
        self.embedding_provider = embedding_provider
        self.temporal_decay = temporal_decay or NoDecay()
        self.embedding_model_id = (
            f"{self.embedding_provider.info.provider}:{self.embedding_provider.info.model_id}"
        )
        from src.storage.vector.kylin_engine import (
            KylinVectorEngineError,
            default_vector_store_backend,
        )

        selected_backend = (
            (vector_store_backend or default_vector_store_backend())
            .strip()
            .lower()
        )
        selected_fallback = (
            (
                vector_store_fallback
                if vector_store_fallback is not None
                else os.environ.get("VECTOR_STORE_FALLBACK", "none")
            )
            .strip()
            .lower()
        )
        if selected_backend == "sqlite":
            self.vector_store = SQLiteMemoryVectorStore(
                db_path,
                dimension=self.embedding_provider.info.dimension,
            )
        elif selected_backend == "kylin":
            if selected_fallback not in {"none", "sqlite"}:
                raise ValueError("vector-store fallback must be 'none' or 'sqlite'")
            from src.memory_module.schema.kylin_vector_store import KylinMemoryVectorStore

            try:
                self.vector_store = KylinMemoryVectorStore(
                    db_path,
                    dimension=self.embedding_provider.info.dimension,
                    fallback_on_error=selected_fallback == "sqlite",
                    drop_on_close=drop_vector_collection_on_close,
                    collection_identity=vector_collection_identity,
                )
            except KylinVectorEngineError:
                if selected_fallback != "sqlite":
                    raise
                self.vector_store = SQLiteMemoryVectorStore(
                    db_path,
                    dimension=self.embedding_provider.info.dimension,
                )
                self.vector_store.last_backend = "sqlite-vec-fallback"
                self.vector_store.fallback_count = 1
        else:
            raise ValueError("vector-store backend must be 'sqlite' or 'kylin'")
        self.vector_store_backend = selected_backend
        self.vector_store_fallback = selected_fallback
        self.fts_store = FTSMemoryStore(db_path)
        self.last_search_diagnostics: dict[str, Any] = {}

    def search(
        self,
        query: str,
        user_id: str,
        project_id: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        return self.search_many([query], user_id, project_id, top_k, filters)[0]

    def search_many(
        self,
        queries: Sequence[str],
        user_id: str,
        project_id: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[list[MemoryResult]]:
        """Retrieve an ordered batch from one scoped index snapshot."""
        total_started = time.perf_counter()
        self.last_search_diagnostics = {}
        self.last_batch_diagnostics: list[dict[str, Any]] = [{} for _ in queries]
        output: list[list[MemoryResult]] = [[] for _ in queries]
        positions = [index for index, query in enumerate(queries) if query.strip()]
        if top_k <= 0 or not positions:
            return output

        store_started = time.perf_counter()
        filters = filters or {}
        status = str(filters.get("memory_status") or filters.get("status") or ACTIVE)
        memory_type = filters.get("type")
        persona = (
            self.layered_store.get_persona(user_id, project_id)
            if not memory_type or str(memory_type) in {"preference", "persona"}
            else None
        )
        persona_result = (
            MemoryResult(
                memory_id=f"persona:{user_id}:{project_id}",
                content=str(persona["content"]),
                type="persona",
                score=1.0,
                source="tencent_l3_persona",
                status=ACTIVE,
                updated_at=str(persona["updated_at"]),
                metadata={
                    "layer": "L3",
                    "version": persona["version"],
                    "source_scene_ids": persona["source_scene_ids"],
                },
            )
            if persona
            else None
        )
        records = self.store.search(
            user_id=user_id,
            project_id=project_id,
            status=status,
            memory_type=str(memory_type) if memory_type else None,
        )
        store_ms = (time.perf_counter() - store_started) * 1000
        if not records:
            self.last_search_diagnostics = {
                "T_store_lookup": store_ms,
                "T_index_refresh": 0.0,
                "T_embedding": 0.0,
                "T_dense": 0.0,
                "T_total": (time.perf_counter() - total_started) * 1000,
            }
            for index in positions:
                output[index] = [deepcopy(persona_result)] if persona_result else []
                self.last_batch_diagnostics[index] = dict(self.last_search_diagnostics)
            return output

        index_started = time.perf_counter()
        self.index_records(records)
        index_ms = (time.perf_counter() - index_started) * 1000
        embedding_started = time.perf_counter()
        query_vectors = self.embedding_provider.embed_queries(
            [queries[index] for index in positions]
        )
        if len(query_vectors) != len(positions):
            raise ValueError("embedding provider returned an incorrect query batch size")
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        dense_started = time.perf_counter()
        search_kwargs = dict(
            user_id=user_id,
            project_id=project_id,
            status=status,
            model_id=self.embedding_model_id,
            top_k=top_k,
            memory_type=str(memory_type) if memory_type else None,
        )
        search_many = getattr(self.vector_store, "search_many", None)
        if callable(search_many) and len(query_vectors) > 1:
            hit_batches = search_many(query_vectors, **search_kwargs)
        else:
            hit_batches = [
                self.vector_store.search(vector, **search_kwargs) for vector in query_vectors
            ]
        if len(hit_batches) != len(positions):
            raise ValueError("vector store returned an incorrect result batch size")
        dense_ms = (time.perf_counter() - dense_started) * 1000
        records_by_id = {record.memory_id: record for record in records}
        now = utc_now_iso()
        for index, hits in zip(positions, hit_batches):
            results = [
                MemoryResult(
                    memory_id=record.memory_id,
                    content=record.content,
                    type=record.type,
                    score=distance_to_cosine_similarity(hit.distance),
                    source=record.source,
                    status=record.status,
                    updated_at=record.updated_at,
                    metadata=_result_metadata(record),
                )
                for hit in hits
                if (record := records_by_id.get(hit.memory_id)) is not None
            ]
            for result in results:
                record = records_by_id[result.memory_id]
                decay = self.temporal_decay.compute(record.created_at, now)
                result.score *= decay
            results.sort(key=lambda r: r.score, reverse=True)
            if persona_result is not None:
                results.insert(0, deepcopy(persona_result))
            output[index] = results[:top_k]
        self.last_search_diagnostics = {
            "T_store_lookup": store_ms,
            "T_index_refresh": index_ms,
            "T_embedding": embedding_ms,
            "T_dense": dense_ms,
            "T_total": (time.perf_counter() - total_started) * 1000,
        }
        for index in positions:
            self.last_batch_diagnostics[index] = {
                key: value / len(positions) for key, value in self.last_search_diagnostics.items()
            }
        return output

    def index_records(self, records: Sequence[MemoryRecord]) -> None:
        """Keep FTS current, then embed only missing or changed records."""
        self.index_record_batches(records, batch_size=max(1, len(records)))

    def index_record_batches(
        self, records: Sequence[MemoryRecord], *, batch_size: int, prefetch: bool = False,
    ) -> None:
        """Overlap at most one encoding batch with ordered vector writes.

        No queries or concurrent record mutations may run on this retriever
        until indexing returns. The embedding provider has only one caller.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        started = time.perf_counter()
        timings = {key: 0.0 for key in (
            "T_fts_sync", "T_index_state", "T_document_embedding", "T_vector_write",
        )}

        def prepare(batch):
            return self._pending_index_records(batch, timings)

        def encode(pending):
            tick = time.perf_counter()
            vectors = self._embed_index_records(pending) if pending else []
            return vectors, (time.perf_counter() - tick) * 1000

        def write(pending, encoded):
            vectors, elapsed = encoded
            timings["T_document_embedding"] += elapsed
            tick = time.perf_counter()
            if pending:
                self._write_index_vectors(pending, vectors)
            timings["T_vector_write"] += (time.perf_counter() - tick) * 1000

        batches = iter(records[start:start + batch_size]
                       for start in range(0, len(records), batch_size))
        if (prefetch and len(records) > batch_size
                and len({record.memory_id for record in records}) == len(records)):
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-index") as pool:
                pending = prepare(next(batches))
                future = pool.submit(encode, pending)
                for batch in batches:
                    encoded = future.result()
                    next_pending = prepare(batch)
                    future = pool.submit(encode, next_pending)
                    write(pending, encoded)
                    pending = next_pending
                write(pending, future.result())
        else:
            for batch in batches:
                pending = prepare(batch)
                write(pending, encode(pending))
        timings["T_index_total"] = (time.perf_counter() - started) * 1000
        self.last_index_diagnostics = timings

    def _pending_index_records(self, records, timings):
        started = time.perf_counter()

        if not records:
            return []
        self.fts_store.upsert_many(
            (
                record.memory_id,
                record.content,
                record.user_id,
                record.project_id,
            )
            for record in records
            if record.status == ACTIVE
        )
        self.fts_store.delete_many(
            record.memory_id for record in records if record.status != ACTIVE
        )
        timings["T_fts_sync"] += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        indexed = self.vector_store.indexed_states(
            [record.memory_id for record in records],
            self.embedding_model_id,
        )
        pending = [
            record
            for record in records
            if indexed.get(record.memory_id)
            != (_content_hash(record.content), record.status, record.type)
        ]
        timings["T_index_state"] += (time.perf_counter() - started) * 1000
        return pending

    def _embed_index_records(self, pending):
        try:
            vectors = self.embedding_provider.embed_documents(
                [record.content for record in pending]
            )
        except Exception as exc:
            raise ValueError(
                f"embedding failed for {len(pending)} records: {type(exc).__name__}: {exc}"
            ) from exc
        if len(vectors) != len(pending):
            raise ValueError(
                f"embedding provider returned {len(vectors)} vectors for {len(pending)} records"
            )
        return vectors

    def _write_index_vectors(self, pending, vectors):
        self.vector_store.upsert_many(
            (
                record.memory_id,
                vector,
                record.user_id,
                record.project_id,
                record.status,
                record.type,
                self.embedding_model_id,
                _content_hash(record.content),
            )
            for record, vector in zip(pending, vectors)
        )

    def update_status(self, memory_ids: Sequence[str], status: str) -> int:
        """Update vector visibility and remove inactive records from FTS recall."""

        ids = list(dict.fromkeys(memory_ids))
        changed = self.vector_store.update_status(ids, status)
        if status != ACTIVE:
            for memory_id in ids:
                self.fts_store.delete(memory_id)
        return changed

    def close(self) -> None:
        close = getattr(self.vector_store, "close", None)
        if callable(close):
            close()


def _features(text: str) -> list[str]:
    ascii_terms = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1)]
    return ascii_terms + cjk_chars + cjk_bigrams


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _result_metadata(record: MemoryRecord) -> dict[str, Any]:
    """Expose structured fields only when an extractor actually populated them."""

    metadata: dict[str, Any] = {
        **{key: value for key, value in record.metadata.items() if value is not None},
        "version": record.version,
        "effective_at": record.effective_at,
    }
    if record.supersedes_id is not None:
        metadata["supersedes_id"] = record.supersedes_id
    if record.superseded_by_id is not None:
        metadata["superseded_by_id"] = record.superseded_by_id
    if record.subject or record.attribute or record.value:
        metadata.update(
            {
                "subject": record.subject,
                "attribute": record.attribute,
                "value": record.value,
                "polarity": record.polarity,
                "scope": record.scope,
                "confidence": record.confidence,
            }
        )
    return metadata
