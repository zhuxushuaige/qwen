"""Chunk repository for SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from src.rag_module.chunking.models import Chunk


@dataclass
class ChunkRecord:
    chunk_pk: int
    chunk_id: str
    doc_pk: int
    project_id: str
    parent_chunk_pk: int | None
    chunk_index: int
    content: str
    heading_path: str | None
    page_start: int | None
    page_end: int | None
    token_count: int | None
    content_hash: str
    embedding_status: str
    status: str
    metadata: dict | None


class ChunkRepository:
    """CRUD operations for the chunks table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create_batch(self, chunks: list[Chunk], doc_pk: int) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for chunk in chunks:
            self._conn.execute(
                """
                INSERT INTO chunks
                    (chunk_id, doc_pk, project_id, parent_chunk_pk, chunk_index,
                     content, heading_path, page_start, page_end, token_count,
                     content_hash, embedding_status, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'active', ?)
                """,
                (
                    chunk.chunk_id, doc_pk, chunk.project_id,
                    chunk.parent_chunk_id, chunk.chunk_index,
                    chunk.content, chunk.heading_path,
                    chunk.page_start, chunk.page_end, chunk.token_count,
                    chunk.content_hash,
                    json.dumps(chunk.metadata) if chunk.metadata else None,
                ),
            )
            records.append(self.get_by_id(chunk.chunk_id))  # type: ignore[arg-type]
        return records

    def get_by_id(self, chunk_id: str) -> ChunkRecord | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def get_by_pk(self, chunk_pk: int) -> ChunkRecord | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_pk = ?", (chunk_pk,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def get_by_pks(self, chunk_pks: list[int]) -> list[ChunkRecord]:
        if not chunk_pks:
            return []
        placeholders = ",".join("?" for _ in chunk_pks)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_pk IN ({placeholders})",
            chunk_pks,
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def update_embedding_status(self, chunk_pk: int, status: str) -> None:
        self._conn.execute(
            "UPDATE chunks SET embedding_status = ? WHERE chunk_pk = ?",
            (status, chunk_pk),
        )

    def update_embedding_status_batch(self, chunk_pks: list[int], status: str) -> None:
        if not chunk_pks:
            return
        placeholders = ",".join("?" for _ in chunk_pks)
        self._conn.execute(
            f"UPDATE chunks SET embedding_status = ? WHERE chunk_pk IN ({placeholders})",
            [status] + chunk_pks,
        )

    def list_pending(self, doc_pk: int) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_pk = ? AND embedding_status = 'pending' ORDER BY chunk_index",
            (doc_pk,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list_by_doc(self, doc_pk: int) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_pk = ? ORDER BY chunk_index",
            (doc_pk,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def delete_by_doc(self, doc_pk: int) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_pk = ?", (doc_pk,))


def _row_to_record(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        chunk_pk=row["chunk_pk"],
        chunk_id=row["chunk_id"],
        doc_pk=row["doc_pk"],
        project_id=row["project_id"],
        parent_chunk_pk=row["parent_chunk_pk"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        heading_path=row["heading_path"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        token_count=row["token_count"],
        content_hash=row["content_hash"],
        embedding_status=row["embedding_status"],
        status=row["status"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else None,
    )
