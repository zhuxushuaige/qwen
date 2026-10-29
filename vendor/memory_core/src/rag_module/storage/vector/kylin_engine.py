"""RAG vector store backed by the Kylin system vector-engine."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

from src.rag_module.storage.vector.base import VectorHit
from src.rag_module.storage.vector.sqlite_vec import SqliteVecStore
from src.storage.vector.kylin_engine import (
    KylinVectorEngineClient,
    KylinVectorEngineError,
    collection_name,
    scalar_expression,
)


class KylinVectorStore:
    """Use Kylin for dense retrieval and sqlite-vec as an optional mirror/fallback."""

    backend = "kylin-ai-vector-engine"

    def __init__(
        self,
        conn: sqlite3.Connection,
        dimension: int,
        *,
        fallback_on_error: bool = False,
        collection_identity: str | None = None,
        drop_on_close: bool = False,
        client: KylinVectorEngineClient | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self._conn = conn
        self._dimension = dimension
        self._fallback = SqliteVecStore(conn, dimension)
        self._fallback_on_error = fallback_on_error
        self._drop_on_close = drop_on_close
        self._client = client or KylinVectorEngineClient(app_id="hust-kylin-os-memory-rag")
        identity = collection_identity or _connection_identity(conn)
        self.collection = collection_name("rag", identity, dimension)
        self.fallback_count = 0
        self.last_backend = self.backend
        self._closed = False
        if self._drop_on_close:
            self._client.drop_collection(self.collection)
        self._client.ensure_collection(self.collection, dimension)

    def upsert(
        self,
        chunk_pk: int,
        embedding: list[float],
        project_id: str,
        doc_pk: int,
        is_active: bool = True,
    ) -> None:
        self.upsert_batch([(chunk_pk, embedding, project_id, doc_pk, is_active)])

    def upsert_batch(
        self,
        entries: list[tuple[int, list[float] | bytes, str, int, bool]],
    ) -> None:
        normalized = [
            (pk, _vector_values(embedding, self._dimension), project_id, doc_pk, active)
            for pk, embedding, project_id, doc_pk, active in entries
        ]
        self._fallback.upsert_batch(normalized)
        self._engine_call(
            self._client.upsert,
            self.collection,
            [
                _row(pk, vector, project_id, doc_pk, active)
                for pk, vector, project_id, doc_pk, active in normalized
            ],
        )

    def search(
        self,
        query_vector: list[float],
        k: int,
        project_id: str | None = None,
        is_active: bool | None = None,
    ) -> list[VectorHit]:
        expressions = [scalar_expression("kind", "rag")]
        if project_id is not None:
            expressions.append(scalar_expression("project_id", project_id))
        if is_active is not None:
            expressions.append(scalar_expression("active", is_active))
        try:
            hits = self._client.search(
                self.collection,
                query_vector,
                top_k=k,
                expression=" && ".join(expressions),
            )
            self.last_backend = self.backend
            return [
                VectorHit(chunk_pk=int(hit["id"]), distance=1.0 - float(hit["score"]))
                for hit in hits
            ]
        except KylinVectorEngineError:
            if not self._fallback_on_error:
                raise
            self.fallback_count += 1
            self.last_backend = "sqlite-vec-fallback"
            return self._fallback.search(query_vector, k, project_id, is_active)

    def search_many(
        self,
        query_vectors: list[list[float]],
        k: int,
        project_id: str | None = None,
        is_active: bool | None = None,
    ) -> list[list[VectorHit]]:
        if not query_vectors:
            return []
        expressions = [scalar_expression("kind", "rag")]
        if project_id is not None:
            expressions.append(scalar_expression("project_id", project_id))
        if is_active is not None:
            expressions.append(scalar_expression("active", is_active))
        try:
            batches = self._client.search_many(
                self.collection,
                query_vectors,
                top_k=k,
                expression=" && ".join(expressions),
            )
            self.last_backend = self.backend
            return [
                [
                    VectorHit(chunk_pk=int(hit["id"]), distance=1.0 - float(hit["score"]))
                    for hit in hits
                ]
                for hits in batches
            ]
        except KylinVectorEngineError:
            if not self._fallback_on_error:
                raise
            self.fallback_count += 1
            self.last_backend = "sqlite-vec-fallback"
            return [
                self._fallback.search(vector, k, project_id, is_active) for vector in query_vectors
            ]

    def delete_by_doc(self, doc_pk: int) -> None:
        entries = self._fallback.entries_by_doc(doc_pk)
        self._engine_call(self._client.delete, self.collection, [row[0] for row in entries])
        self._fallback.delete_by_doc(doc_pk)

    def deactivate_by_doc(self, doc_pk: int) -> None:
        self._set_doc_active(doc_pk, False)

    def activate_by_doc(self, doc_pk: int) -> None:
        self._set_doc_active(doc_pk, True)

    def _set_doc_active(self, doc_pk: int, active: bool) -> None:
        entries = self._fallback.entries_by_doc(doc_pk)
        if active:
            self._fallback.activate_by_doc(doc_pk)
        else:
            self._fallback.deactivate_by_doc(doc_pk)
        updated = [
            (pk, vector, project_id, stored_doc_pk, active)
            for pk, vector, project_id, stored_doc_pk, _ in entries
        ]
        self._engine_call(
            self._client.upsert,
            self.collection,
            [
                _row(pk, vector, project_id, stored_doc_pk, active)
                for pk, vector, project_id, stored_doc_pk, active in updated
            ],
        )

    def _engine_call(self, operation, *args):
        try:
            result = operation(*args)
            self.last_backend = self.backend
            return result
        except KylinVectorEngineError:
            if not self._fallback_on_error:
                raise
            self.fallback_count += 1
            self.last_backend = "sqlite-vec-fallback"
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._drop_on_close:
                self._client.drop_collection(self.collection)
        finally:
            self._client.close()


def _row(
    chunk_pk: int,
    embedding: list[float],
    project_id: str,
    doc_pk: int,
    active: bool,
) -> dict[str, object]:
    return {
        "id": int(chunk_pk),
        "embedding": embedding,
        "entity_id": str(chunk_pk),
        "kind": "rag",
        "project_id": project_id,
        "user_id": "",
        "doc_pk": int(doc_pk),
        "active": bool(active),
        "status": "active" if active else "inactive",
        "memory_type": "",
        "model_id": "",
        "content_hash": "",
    }


def _connection_identity(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    path = str(row[2]) if row and len(row) > 2 else ""
    return str(Path(path).resolve()) if path else f"sqlite-memory-{id(conn)}"


def _vector_values(value: list[float] | bytes, dimension: int) -> list[float]:
    if isinstance(value, bytes):
        if len(value) != dimension * 4:
            raise ValueError("invalid vector blob length")
        return list(struct.unpack(f"{dimension}f", value))
    return [float(item) for item in value]
