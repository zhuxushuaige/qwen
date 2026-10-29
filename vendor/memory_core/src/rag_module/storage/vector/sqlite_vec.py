"""sqlite-vec vector store implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

from src.rag_module.storage.vector.base import VectorHit


class SqliteVecStore:
    """Vector store using sqlite-vec extension."""

    backend = "sqlite-vec"

    def __init__(self, conn: sqlite3.Connection, dimension: int = 1024) -> None:
        self._conn = conn
        self._dimension = dimension
        self.last_backend = self.backend
        self.fallback_count = 0
        self._ensure_extension()
        self._ensure_table()

    def _ensure_extension(self) -> None:
        self._conn.enable_load_extension(True)
        try:
            import sqlite_vec

            sqlite_vec.load(self._conn)
        except ImportError:
            pass

    def _ensure_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_vectors USING vec0(
                chunk_pk INTEGER PRIMARY KEY,
                embedding FLOAT[{self._dimension}],
                project_id TEXT,
                doc_pk INTEGER,
                is_active BOOLEAN
            )
            """
        )

    def upsert(
        self,
        chunk_pk: int,
        embedding: list[float],
        project_id: str,
        doc_pk: int,
        is_active: bool = True,
    ) -> None:
        self._conn.execute(
            "DELETE FROM doc_vectors WHERE chunk_pk = ?",
            (chunk_pk,),
        )
        self._conn.execute(
            "INSERT INTO doc_vectors (chunk_pk, embedding, project_id, doc_pk, is_active) VALUES (?, ?, ?, ?, ?)",
            (chunk_pk, _to_blob(embedding), project_id, doc_pk, int(is_active)),
        )

    def upsert_batch(
        self,
        entries: list[tuple[int, list[float] | bytes, str, int, bool]],
    ) -> None:
        pks = [e[0] for e in entries]
        if pks:
            placeholders = ",".join("?" for _ in pks)
            self._conn.execute(
                f"DELETE FROM doc_vectors WHERE chunk_pk IN ({placeholders})",
                pks,
            )
        self._conn.executemany(
            "INSERT INTO doc_vectors (chunk_pk, embedding, project_id, doc_pk, is_active) VALUES (?, ?, ?, ?, ?)",
            [(pk, _to_blob(emb), pid, dpk, int(act)) for pk, emb, pid, dpk, act in entries],
        )

    def search(
        self,
        query_vector: list[float],
        k: int,
        project_id: str | None = None,
        is_active: bool | None = None,
    ) -> list[VectorHit]:
        conditions = ["embedding MATCH ?"]
        params: list[Any] = [_to_blob(query_vector)]
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if is_active is not None:
            conditions.append("is_active = ?")
            params.append(int(is_active))
        params.append(k)

        where = " AND ".join(conditions)
        # Keep the existing L2 index/ranking, but expose cosine distance for
        # backend-independent RAG rejection scores (also for existing tables).
        sql = (
            "SELECT chunk_pk, distance, vec_distance_cosine(embedding, ?) "
            f"FROM doc_vectors WHERE {where} AND k = ?"
        )
        rows = self._conn.execute(sql, [_to_blob(query_vector), *params]).fetchall()
        return [
            VectorHit(
                chunk_pk=row[0],
                distance=row[1],
                metadata={"cosine_distance": float(row[2]) if row[2] is not None else 1.0},
            )
            for row in rows
        ]

    def delete_by_doc(self, doc_pk: int) -> None:
        self._conn.execute("DELETE FROM doc_vectors WHERE doc_pk = ?", (doc_pk,))

    def entries_by_doc(
        self,
        doc_pk: int,
    ) -> list[tuple[int, list[float], str, int, bool]]:
        """Export mirrored rows for backend migration or status synchronization."""

        rows = self._conn.execute(
            "SELECT chunk_pk, embedding, project_id, doc_pk, is_active "
            "FROM doc_vectors WHERE doc_pk = ?",
            (doc_pk,),
        ).fetchall()
        return [
            (
                int(row[0]),
                _from_blob(row[1], self._dimension),
                str(row[2]),
                int(row[3]),
                bool(row[4]),
            )
            for row in rows
        ]

    def deactivate_by_doc(self, doc_pk: int) -> None:
        self._conn.execute(
            "UPDATE doc_vectors SET is_active = 0 WHERE doc_pk = ?",
            (doc_pk,),
        )

    def activate_by_doc(self, doc_pk: int) -> None:
        self._conn.execute(
            "UPDATE doc_vectors SET is_active = 1 WHERE doc_pk = ?",
            (doc_pk,),
        )


def _to_blob(vector: list[float] | bytes | bytearray | memoryview) -> bytes:
    if isinstance(vector, (bytes, bytearray, memoryview)):
        return bytes(vector)
    import struct

    return struct.pack(f"{len(vector)}f", *vector)


def _from_blob(value: bytes, dimension: int) -> list[float]:
    import struct

    raw = bytes(value)
    expected = dimension * 4
    if len(raw) != expected:
        raise ValueError(f"invalid vector blob length: expected {expected}, got {len(raw)}")
    return list(struct.unpack(f"{dimension}f", raw))
