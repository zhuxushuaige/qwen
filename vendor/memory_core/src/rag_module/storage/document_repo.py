"""Document repository for SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass


@dataclass
class DocumentRecord:
    doc_pk: int
    doc_id: str
    project_id: str
    filename: str
    source_uri: str | None
    file_type: str | None
    version: int
    content_hash: str
    parser_version: str | None
    chunker_version: str | None
    total_chunks: int
    status: str
    error_message: str | None
    created_at: int
    updated_at: int
    metadata: dict | None


class DocumentRepository:
    """CRUD operations for the documents table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(
        self,
        project_id: str,
        filename: str,
        content_hash: str,
        source_uri: str | None = None,
        file_type: str | None = None,
        parser_version: str | None = None,
        chunker_version: str | None = None,
        metadata: dict | None = None,
        version: int = 1,
    ) -> DocumentRecord:
        doc_id = str(uuid.uuid4())
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO documents
                (doc_id, project_id, filename, source_uri, file_type,
                 version, content_hash, parser_version, chunker_version,
                 status, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing', ?, ?, ?)
            """,
            (
                doc_id, project_id, filename, source_uri, file_type,
                version, content_hash, parser_version, chunker_version,
                now, now, json.dumps(metadata) if metadata else None,
            ),
        )
        return self.get_by_id(doc_id)  # type: ignore[return-value]

    def get_by_id(self, doc_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def get_by_pk(self, doc_pk: int) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_pk = ?", (doc_pk,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def update_status(self, doc_pk: int, status: str, error_message: str | None = None) -> None:
        now = int(time.time())
        self._conn.execute(
            "UPDATE documents SET status = ?, error_message = ?, updated_at = ? WHERE doc_pk = ?",
            (status, error_message, now, doc_pk),
        )

    def update_total_chunks(self, doc_pk: int, total_chunks: int) -> None:
        now = int(time.time())
        self._conn.execute(
            "UPDATE documents SET total_chunks = ?, updated_at = ? WHERE doc_pk = ?",
            (total_chunks, now, doc_pk),
        )

    def get_by_content_hash(self, project_id: str, content_hash: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE project_id = ? AND content_hash = ? AND status != 'inactive'",
            (project_id, content_hash),
        ).fetchone()
        return _row_to_record(row) if row else None

    def get_by_filename(self, project_id: str, filename: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE project_id = ? AND filename = ? AND status != 'inactive' ORDER BY version DESC LIMIT 1",
            (project_id, filename),
        ).fetchone()
        return _row_to_record(row) if row else None

    def deactivate(self, doc_pk: int) -> None:
        self.update_status(doc_pk, "inactive")

    def list_by_project(self, project_id: str, status: str | None = None) -> list[DocumentRecord]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE project_id = ? AND status = ? ORDER BY created_at DESC",
                (project_id, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]


_COLUMNS = [
    "doc_pk", "doc_id", "project_id", "filename", "source_uri", "file_type",
    "version", "content_hash", "parser_version", "chunker_version",
    "total_chunks", "status", "error_message", "created_at", "updated_at", "metadata",
]


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_pk=row["doc_pk"],
        doc_id=row["doc_id"],
        project_id=row["project_id"],
        filename=row["filename"],
        source_uri=row["source_uri"],
        file_type=row["file_type"],
        version=row["version"],
        content_hash=row["content_hash"],
        parser_version=row["parser_version"],
        chunker_version=row["chunker_version"],
        total_chunks=row["total_chunks"],
        status=row["status"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else None,
    )
