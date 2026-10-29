"""Entity store with semantic dedup via Qdrant (mem0-style threshold 0.95)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.memory_module.schema.models import utc_now_iso
from src.rag_module.embedding.base import EmbeddingProvider

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    label TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_normalized
    ON entities(normalized_name, label);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_entity
    ON memory_entities(entity_id);

CREATE INDEX IF NOT EXISTS idx_memory_entities_memory
    ON memory_entities(memory_id);
"""

ENTITY_COLLECTION = "entity_vectors"


@dataclass
class Entity:
    entity_id: str
    name: str
    label: str
    normalized_name: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityStore:
    """Persistent entity registry with semantic dedup via Qdrant."""

    def __init__(
        self,
        db_path: str | Path,
        embedding: EmbeddingProvider,
        qdrant_path: str | Path | None = None,
        qdrant_url: str | None = None,
        dedup_threshold: float = 0.95,
    ) -> None:
        self._db_path = str(db_path)
        self._embedding = embedding
        self._threshold = dedup_threshold
        self._ensure_schema()

        if qdrant_url:
            self._qdrant = QdrantClient(url=qdrant_url)
        elif qdrant_path:
            self._qdrant = QdrantClient(path=str(qdrant_path))
        else:
            self._qdrant = QdrantClient(path=":memory:")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._qdrant.get_collection(ENTITY_COLLECTION)
        except Exception:
            self._qdrant.create_collection(
                collection_name=ENTITY_COLLECTION,
                vectors_config=VectorParams(
                    size=self._embedding.info.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _normalized(self, name: str) -> str:
        return " ".join(name.strip().lower().split())

    def _to_qdrant_id(self, entity_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"entity:{entity_id}"))

    # ------------------------------------------------------------------
    # semantic dedup upsert
    # ------------------------------------------------------------------

    def upsert_entity(self, name: str, label: str) -> str:
        """Insert or get existing entity via semantic dedup.

        First checks SQLite by normalized name, then Qdrant by cosine >= 0.95.
        """
        normalized = self._normalized(name)
        # exact text check (fast path)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT entity_id FROM entities WHERE normalized_name = ? AND label = ?",
                (normalized, label),
            ).fetchone()
            if row:
                return row["entity_id"]

        # semantic dedup via Qdrant
        vec = self._embedding.embed_query(name)
        hits = self._qdrant.query_points(
            collection_name=ENTITY_COLLECTION,
            query=vec,
            query_filter=Filter(
                must=[FieldCondition(key="label", match=MatchValue(value=label))]
            ),
            limit=1,
        ).points

        if hits and hits[0].score >= self._threshold:
            return hits[0].payload.get("entity_id", "")

        # new entity
        entity_id = f"ent_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO entities (entity_id, name, label, normalized_name, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (entity_id, name, label, normalized, utc_now_iso()),
            )
        self._qdrant.upsert(
            collection_name=ENTITY_COLLECTION,
            points=[
                PointStruct(
                    id=self._to_qdrant_id(entity_id),
                    vector=vec,
                    payload={"entity_id": entity_id, "label": label},
                )
            ],
        )
        return entity_id

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
            ).fetchone()
        return _row_to_entity(row) if row else None

    def search_entities(self, names: list[str]) -> list[Entity]:
        if not names:
            return []
        normalized = [self._normalized(n) for n in names]
        placeholders = ",".join("?" * len(normalized))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM entities WHERE normalized_name IN ({placeholders})",
                normalized,
            ).fetchall()
        return [_row_to_entity(row) for row in rows]

    def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.* FROM entities e "
                "JOIN memory_entities me ON e.entity_id = me.entity_id "
                "WHERE me.memory_id = ?",
                (memory_id,),
            ).fetchall()
        return [_row_to_entity(row) for row in rows]

    # ------------------------------------------------------------------
    # linking
    # ------------------------------------------------------------------

    def link(self, memory_id: str, entity_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) "
                "VALUES (?, ?)",
                (memory_id, entity_id),
            )

    def unlink_memory(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))

    def get_memory_ids_by_entities(self, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT memory_id FROM memory_entities "
                f"WHERE entity_id IN ({placeholders})",
                entity_ids,
            ).fetchall()
        return [row["memory_id"] for row in rows]


def _row_to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        entity_id=row["entity_id"],
        name=row["name"],
        label=row["label"],
        normalized_name=row["normalized_name"],
        created_at=row["created_at"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
    )
