"""Persistent sqlite-vec index for memory embeddings."""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class MemoryVectorHit:
    memory_id: str
    distance: float


class SQLiteMemoryVectorStore:
    """Store derived memory vectors in the same SQLite database as memories."""

    backend = "sqlite-vec"

    def __init__(self, db_path: str | Path, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.db_path = str(db_path)
        self.dimension = dimension
        self.last_backend = self.backend
        self.fallback_count = 0
        self._ensure_schema()

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
        rows: Iterable[
            tuple[str, Sequence[float], str, str, str, str, str, str]
        ],
    ) -> None:
        """Write multiple vectors in one sqlite-vec transaction."""

        by_memory_id = {
            memory_id: (
                memory_id,
                _to_blob(_normalize(embedding, self.dimension)),
                user_id,
                project_id,
                status,
                memory_type,
                model_id,
                content_hash,
            )
            for (
                memory_id,
                embedding,
                user_id,
                project_id,
                status,
                memory_type,
                model_id,
                content_hash,
            ) in rows
        }
        values = list(by_memory_id.values())
        if not values:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM memory_vectors WHERE memory_id = ?",
                [(memory_id,) for memory_id, *_rest in values],
            )
            conn.executemany(
                """
                INSERT INTO memory_vectors (
                    memory_id, embedding, user_id, project_id, status,
                    memory_type, model_id, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def indexed_states(
        self,
        memory_ids: Iterable[str],
        model_id: str,
    ) -> dict[str, tuple[str, str, str]]:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id, content_hash, status, memory_type
                FROM memory_vectors
                WHERE memory_id IN ({placeholders}) AND model_id = ?
                """,
                [*ids, model_id],
            ).fetchall()
        return {
            str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
            for row in rows
        }

    def export_rows(
        self,
        memory_ids: Iterable[str],
    ) -> list[tuple[str, list[float], str, str, str, str, str, str]]:
        """Export mirrored vectors for system-backend synchronization."""

        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id, embedding, user_id, project_id, status,
                       memory_type, model_id, content_hash
                FROM memory_vectors
                WHERE memory_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
        return [
            (
                str(row[0]),
                _from_blob(row[1], self.dimension),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
                str(row[6]),
                str(row[7]),
            )
            for row in rows
        ]

    def ids_for_scope(
        self,
        *,
        user_id: str,
        project_id: str,
        status: str,
        model_id: str,
        memory_type: str | None = None,
    ) -> list[str]:
        conditions = ["user_id = ?", "project_id = ?", "status = ?", "model_id = ?"]
        params: list[object] = [user_id, project_id, status, model_id]
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT memory_id FROM memory_vectors WHERE " + " AND ".join(conditions),
                params,
            ).fetchall()
        return [str(row[0]) for row in rows]

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
        if top_k <= 0:
            return []
        conditions = [
            "embedding MATCH ?",
            "user_id = ?",
            "project_id = ?",
            "status = ?",
            "model_id = ?",
        ]
        params: list[object] = [
            _to_blob(_normalize(query_vector, self.dimension)),
            user_id,
            project_id,
            status,
            model_id,
        ]
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        conditions.append("k = ?")
        params.append(top_k)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT memory_id, distance FROM memory_vectors WHERE "
                + " AND ".join(conditions),
                params,
            ).fetchall()
        return [MemoryVectorHit(str(row[0]), float(row[1])) for row in rows]

    def keyword_search(
        self, query: str, top_k: int = 100, filters: dict | None = None
    ) -> list[tuple[str, float]] | None:
        """mem0-compatible keyword search via FTS5.

        Returns ``(memory_id, bm25_score)`` or None if FTS5 is not available.
        """
        try:
            user_id = (filters or {}).get("user_id", "")
            project_id = (filters or {}).get("project_id", "")
            from src.memory_module.schema.fts_store import FTSMemoryStore
            return FTSMemoryStore(self.db_path).search(query, user_id, project_id, top_k)
        except Exception:
            return None

    def update_status(self, memory_ids: Iterable[str], status: str) -> int:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE memory_vectors SET status = ? "
                f"WHERE memory_id IN ({placeholders})",
                [status, *ids],
            )
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
            ).fetchone()
            if row:
                match = re.search(r"embedding FLOAT\[(\d+)\]", str(row[0]))
                existing_dimension = int(match.group(1)) if match else None
                has_content_hash = "content_hash" in str(row[0])
                if existing_dimension != self.dimension or not has_content_hash:
                    # Embeddings are derived data and can be safely regenerated.
                    conn.execute("DROP TABLE memory_vectors")
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.dimension}],
                    user_id TEXT,
                    project_id TEXT,
                    status TEXT,
                    memory_type TEXT,
                    model_id TEXT,
                    content_hash TEXT
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        try:
            try:
                import sqlite_vec
            except ImportError as exc:
                raise RuntimeError(
                    "sqlite-vec is required for persistent memory vectors; "
                    "install the dependencies from requirements.txt"
                ) from exc
            sqlite_vec.load(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()


def distance_to_cosine_similarity(distance: float) -> float:
    """Convert L2 distance between normalized vectors to cosine similarity."""

    return round(max(0.0, min(1.0, 1.0 - (distance * distance) / 2.0)), 4)


def _normalize(vector: Sequence[float], dimension: int) -> list[float]:
    if len(vector) != dimension:
        raise ValueError(
            f"embedding dimension mismatch: expected {dimension}, got {len(vector)}"
        )
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def _to_blob(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _from_blob(value: bytes, dimension: int) -> list[float]:
    raw = bytes(value)
    expected = dimension * 4
    if len(raw) != expected:
        raise ValueError(f"invalid vector blob length: expected {expected}, got {len(raw)}")
    return list(struct.unpack(f"{dimension}f", raw))
