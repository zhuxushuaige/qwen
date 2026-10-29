"""Memory vector store backed by the Kylin system vector-engine."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from src.memory_module.schema.models import ACTIVE
from src.memory_module.schema.vector_store import (
    MemoryVectorHit,
    SQLiteMemoryVectorStore,
)
from src.storage.vector.kylin_engine import (
    KylinVectorEngineClient,
    KylinVectorEngineError,
    collection_name,
    scalar_expression,
)


class KylinMemoryVectorStore:
    """Use Kylin for dense search while retaining sqlite-vec as a mirror/fallback."""

    backend = "kylin-ai-vector-engine"

    def __init__(
        self,
        db_path: str | Path,
        dimension: int,
        *,
        fallback_on_error: bool = False,
        drop_on_close: bool = False,
        collection_identity: str | None = None,
        client: KylinVectorEngineClient | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.dimension = dimension
        self._fallback = SQLiteMemoryVectorStore(db_path, dimension)
        self._fallback_on_error = fallback_on_error
        self._drop_on_close = drop_on_close
        self._client = client or KylinVectorEngineClient(app_id="hust-kylin-os-memory-memory")
        identity = collection_identity or (
            str(Path(self.db_path).resolve()) if self.db_path != ":memory:" else self.db_path
        )
        self.collection = collection_name("memory", identity, dimension)
        self.fallback_count = 0
        self.last_backend = self.backend
        self._id_cache: dict[int, str] = {}
        self._closed = False
        try:
            if self._drop_on_close:
                self._client.drop_collection(self.collection)
            self._client.ensure_collection(self.collection, dimension)
        except Exception:
            self._client.close()
            raise

    def upsert(
        self,
        *,
        memory_id: str,
        embedding: Sequence[float],
        user_id: str,
        project_id: str,
        status: str,
        memory_type: str,
        model_id: str,
        content_hash: str,
    ) -> None:
        self.upsert_many(
            [
                (
                    memory_id,
                    embedding,
                    user_id,
                    project_id,
                    status,
                    memory_type,
                    model_id,
                    content_hash,
                )
            ]
        )

    def upsert_many(
        self,
        rows: Iterable[tuple[str, Sequence[float], str, str, str, str, str, str]],
    ) -> None:
        values = list(rows)
        if not values:
            return
        memory_ids = [row[0] for row in values]
        system_ids = [_system_id(memory_id) for memory_id in memory_ids]
        if len(set(system_ids)) != len(set(memory_ids)):
            raise ValueError("memory IDs collide in the Kylin int64 primary-key mapping")
        self._fallback.upsert_many(values)
        self._id_cache.update(zip(system_ids, memory_ids))
        self._engine_call(
            self._client.upsert,
            self.collection,
            [_system_row(*row) for row in values],
        )

    def indexed_states(
        self,
        memory_ids: Iterable[str],
        model_id: str,
    ) -> dict[str, tuple[str, str, str]]:
        return self._fallback.indexed_states(memory_ids, model_id)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        user_id: str,
        project_id: str,
        status: str,
        model_id: str,
        top_k: int,
        memory_type: str | None = None,
    ) -> list[MemoryVectorHit]:
        expressions = [
            scalar_expression("kind", "memory"),
            scalar_expression("user_id", user_id),
            scalar_expression("project_id", project_id),
            scalar_expression("status", status),
            scalar_expression("model_id", model_id),
        ]
        if memory_type:
            expressions.append(scalar_expression("memory_type", memory_type))
        try:
            hits = self._client.search(
                self.collection,
                query_vector,
                top_k=top_k,
                expression=" && ".join(expressions),
                output_fields=("entity_id",),
            )
            self.last_backend = self.backend
            results: list[MemoryVectorHit] = []
            for hit in hits:
                memory_id = str(hit.get("entity_id") or "") or self._id_cache.get(int(hit["id"]))
                if memory_id is None:
                    continue
                cosine = max(-1.0, min(1.0, float(hit["score"])))
                distance = max(0.0, 2.0 - 2.0 * cosine) ** 0.5
                results.append(MemoryVectorHit(memory_id, distance))
            return results
        except KylinVectorEngineError:
            if not self._fallback_on_error:
                raise
            self.fallback_count += 1
            self.last_backend = "sqlite-vec-fallback"
            return self._fallback.search(
                query_vector,
                user_id=user_id,
                project_id=project_id,
                status=status,
                model_id=model_id,
                top_k=top_k,
                memory_type=memory_type,
            )

    def search_many(
        self,
        query_vectors: Sequence[Sequence[float]],
        *,
        user_id: str,
        project_id: str,
        status: str,
        model_id: str,
        top_k: int,
        memory_type: str | None = None,
    ) -> list[list[MemoryVectorHit]]:
        """Search one scope in a native SDK batch and one bridge round trip."""

        if not query_vectors:
            return []
        expressions = [
            scalar_expression("kind", "memory"),
            scalar_expression("user_id", user_id),
            scalar_expression("project_id", project_id),
            scalar_expression("status", status),
            scalar_expression("model_id", model_id),
        ]
        if memory_type:
            expressions.append(scalar_expression("memory_type", memory_type))
        try:
            batches = self._client.search_many(
                self.collection,
                query_vectors,
                top_k=top_k,
                expression=" && ".join(expressions),
                output_fields=("entity_id",),
            )
            self.last_backend = self.backend
            return [self._memory_hits(hits) for hits in batches]
        except KylinVectorEngineError:
            if not self._fallback_on_error:
                raise
            self.fallback_count += 1
            self.last_backend = "sqlite-vec-fallback"
            return [
                self._fallback.search(
                    vector,
                    user_id=user_id,
                    project_id=project_id,
                    status=status,
                    model_id=model_id,
                    top_k=top_k,
                    memory_type=memory_type,
                )
                for vector in query_vectors
            ]

    def _memory_hits(self, hits: Sequence[dict[str, object]]) -> list[MemoryVectorHit]:
        results: list[MemoryVectorHit] = []
        for hit in hits:
            memory_id = str(hit.get("entity_id") or "") or self._id_cache.get(int(hit["id"]))
            if memory_id is None:
                continue
            cosine = max(-1.0, min(1.0, float(hit["score"])))
            results.append(MemoryVectorHit(memory_id, max(0.0, 2.0 - 2.0 * cosine) ** 0.5))
        return results

    def keyword_search(self, query: str, top_k: int = 100, filters: dict | None = None):
        return self._fallback.keyword_search(query, top_k, filters)

    def update_status(self, memory_ids: Iterable[str], status: str) -> int:
        ids = list(dict.fromkeys(memory_ids))
        changed = self._fallback.update_status(ids, status)
        rows = self._fallback.export_rows(ids)
        if rows:
            self._engine_call(
                self._client.upsert,
                self.collection,
                [_system_row(*row) for row in rows],
            )
        return changed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._drop_on_close:
                self._client.drop_collection(self.collection)
        finally:
            self._client.close()

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


def _system_id(memory_id: str) -> int:
    value = int.from_bytes(hashlib.sha256(memory_id.encode("utf-8")).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def _system_row(
    memory_id: str,
    embedding: Sequence[float],
    user_id: str,
    project_id: str,
    status: str,
    memory_type: str,
    model_id: str,
    content_hash: str,
) -> dict[str, object]:
    return {
        "id": _system_id(memory_id),
        "embedding": [float(value) for value in embedding],
        "entity_id": memory_id,
        "kind": "memory",
        "project_id": project_id,
        "user_id": user_id,
        "doc_pk": 0,
        "active": status == ACTIVE,
        "status": status,
        "memory_type": memory_type,
        "model_id": model_id,
        "content_hash": content_hash,
    }
