"""SQLite-backed storage for long-term memory records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.memory_module.schema.models import (
    ACTIVE,
    FORGOTTEN,
    SUPERSEDED,
    MemoryRecord,
    utc_now_iso,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    attribute TEXT NOT NULL DEFAULT '',
    value TEXT NOT NULL DEFAULT '',
    polarity TEXT NOT NULL DEFAULT 'positive',
    scope TEXT NOT NULL DEFAULT 'project',
    confidence REAL NOT NULL DEFAULT 1.0,
    version INTEGER NOT NULL DEFAULT 1,
    supersedes_id TEXT,
    superseded_by_id TEXT,
    effective_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_scope_status
    ON memories(user_id, project_id, status, updated_at);
CREATE TABLE IF NOT EXISTS memory_organization_jobs (
    job_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    lease_until TEXT,
    last_error TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(memory_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_organization_jobs_claim
    ON memory_organization_jobs(status, lease_until, created_at);
"""


class SQLiteMemoryStore:
    """Small persistence layer shared by retriever, writer, and forgetter."""

    def __init__(self, db_path: str | Path = ".memory_module.sqlite3") -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def upsert(
        self,
        record: MemoryRecord,
        *,
        organization_max_attempts: int | None = None,
    ) -> tuple[MemoryRecord, bool]:
        """Insert or refresh a memory and optionally enqueue organization atomically."""

        if organization_max_attempts is not None and organization_max_attempts <= 0:
            raise ValueError("organization_max_attempts must be positive")

        with self._connect() as conn:
            return self._upsert_with_connection(
                conn,
                record,
                organization_max_attempts=organization_max_attempts,
            )

    def upsert_many(
        self,
        records: Sequence[MemoryRecord],
    ) -> list[tuple[MemoryRecord, bool]]:
        """Insert multiple memories in one transaction, preserving upsert semantics."""

        if not records:
            return []
        with self._connect() as conn:
            return [self._upsert_with_connection(conn, record) for record in records]

    def _upsert_with_connection(
        self,
        conn: sqlite3.Connection,
        record: MemoryRecord,
        *,
        organization_max_attempts: int | None = None,
    ) -> tuple[MemoryRecord, bool]:
        normalized = normalize_content(record.content)
        now = utc_now_iso()
        row = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE user_id = ? AND project_id = ? AND normalized_content = ?
                """,
                (record.user_id, record.project_id, normalized),
            ).fetchone()
        if row:
            memory_id = row["memory_id"]
            created_at = row["created_at"]
            conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, type = ?, status = ?, source = ?, subject = ?,
                        attribute = ?, value = ?, polarity = ?, scope = ?,
                        confidence = ?, effective_at = ?, updated_at = ?, metadata_json = ?
                    WHERE memory_id = ?
                    """,
                    (
                        record.content,
                        record.type,
                        ACTIVE,
                        record.source,
                        record.subject,
                        record.attribute,
                        record.value,
                        record.polarity,
                        record.scope,
                        record.confidence,
                        record.effective_at,
                        now,
                        _to_json(record.metadata),
                        memory_id,
                    ),
                )
            saved = replace(
                record,
                memory_id=memory_id,
                status=ACTIVE,
                version=int(row["version"]),
                supersedes_id=row["supersedes_id"],
                superseded_by_id=row["superseded_by_id"],
                created_at=created_at,
                updated_at=now,
            )
            return saved, False

        memory_id = record.memory_id or f"mem_{uuid.uuid4().hex[:12]}"
        created_at = record.created_at or now
        conn.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, project_id, content, normalized_content,
                    type, status, source, subject, attribute, value, polarity,
                    scope, confidence, version, supersedes_id, superseded_by_id,
                    effective_at, created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    record.user_id,
                    record.project_id,
                    record.content,
                    normalized,
                    record.type,
                    record.status,
                    record.source,
                    record.subject,
                    record.attribute,
                    record.value,
                    record.polarity,
                    record.scope,
                    record.confidence,
                    record.version,
                    record.supersedes_id,
                    record.superseded_by_id,
                    record.effective_at,
                    created_at,
                    now,
                    _to_json(record.metadata),
                ),
            )
        if organization_max_attempts is not None:
            conn.execute(
                """
                INSERT INTO memory_organization_jobs (
                    job_id, memory_id, status, attempts, max_attempts,
                    lease_until, last_error, result_json, created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, NULL, NULL, '{}', ?, ?)
                """,
                (memory_id, memory_id, organization_max_attempts, now, now),
            )
        saved = replace(
            record,
            memory_id=memory_id,
            created_at=created_at,
            updated_at=now,
        )
        return saved, True

    def get_organization_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_organization_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _row_to_organization_job(row) if row else None

    def claim_organization_job(self, lease_seconds: float = 300.0) -> dict[str, Any] | None:
        """Atomically claim the oldest pending or expired organization job."""

        jobs = self.claim_organization_jobs(1, lease_seconds=lease_seconds)
        return jobs[0] if jobs else None

    def claim_organization_jobs(
        self,
        limit: int,
        *,
        lease_seconds: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Atomically claim an ordered batch of durable organization jobs."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat(timespec="seconds")
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat(timespec="seconds")
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE memory_organization_jobs
                SET status = 'failed', lease_until = NULL,
                    last_error = COALESCE(last_error, 'organization lease expired'),
                    updated_at = ?
                WHERE status = 'running' AND lease_until IS NOT NULL
                  AND lease_until <= ? AND attempts >= max_attempts
                """,
                (now_iso, now_iso),
            )
            rows = conn.execute(
                """
                SELECT * FROM memory_organization_jobs
                WHERE attempts < max_attempts
                  AND (
                    status = 'queued'
                    OR (status = 'running' AND lease_until IS NOT NULL AND lease_until <= ?)
                )
                ORDER BY created_at, job_id
                LIMIT ?
                """,
                (now_iso, limit),
            ).fetchall()
            if not rows:
                conn.commit()
                return []
            claimed_ids: list[str] = []
            for row in rows:
                cursor = conn.execute(
                    """
                    UPDATE memory_organization_jobs
                    SET status = 'running', attempts = attempts + 1,
                        lease_until = ?, last_error = NULL, updated_at = ?
                    WHERE job_id = ? AND attempts = ?
                    """,
                    (lease_until, now_iso, row["job_id"], row["attempts"]),
                )
                if cursor.rowcount == 1:
                    claimed_ids.append(str(row["job_id"]))
            placeholders = ",".join("?" for _ in claimed_ids)
            claimed = (
                conn.execute(
                    f"SELECT * FROM memory_organization_jobs WHERE job_id IN ({placeholders}) "
                    "ORDER BY created_at, job_id",
                    claimed_ids,
                ).fetchall()
                if claimed_ids
                else []
            )
            conn.commit()
            return [_row_to_organization_job(row) for row in claimed]
        finally:
            conn.close()

    def complete_organization_job(self, job_id: str, result: dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_organization_jobs
                SET status = 'completed', lease_until = NULL, last_error = NULL,
                    result_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (_to_json(result), now, job_id),
            )

    def fail_organization_job(self, job_id: str, error: str) -> str:
        """Retry a failed job until its durable attempt budget is exhausted."""

        now = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT attempts, max_attempts FROM memory_organization_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return "not_found"
            status = "queued" if int(row["attempts"]) < int(row["max_attempts"]) else "failed"
            conn.execute(
                """
                UPDATE memory_organization_jobs
                SET status = ?, lease_until = NULL, last_error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, error, now, job_id),
            )
        return status

    def pending_organization_jobs(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count FROM memory_organization_jobs
                WHERE status IN ('queued', 'running')
                """
            ).fetchone()
        return int(row["count"])

    def search(
        self,
        user_id: str,
        project_id: str,
        status: str = ACTIVE,
        memory_type: str | None = None,
    ) -> list[MemoryRecord]:
        query = "SELECT * FROM memories WHERE user_id = ? AND project_id = ? AND status = ?"
        params: list[Any] = [user_id, project_id, status]
        if memory_type:
            query += " AND type = ?"
            params.append(memory_type)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def list_memories(
        self,
        user_id: str,
        project_id: str,
        status: str | None = None,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        conditions = ["user_id = ?", "project_id = ?"]
        params: list[Any] = [user_id, project_id]
        if status:
            conditions.append("status = ?")
            params.append(status)
        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE "
                + " AND ".join(conditions)
                + " ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def find_exact(
        self,
        user_id: str,
        project_id: str,
        content: str,
        status: str = ACTIVE,
    ) -> MemoryRecord | None:
        """Find an exact normalized-content match in the requested state."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND project_id = ?
                  AND normalized_content = ? AND status = ?
                """,
                (user_id, project_id, normalize_content(content), status),
            ).fetchone()
        return _row_to_record(row) if row else None

    def find_structured_slot(
        self,
        user_id: str,
        project_id: str,
        memory_type: str,
        subject: str,
        attribute: str,
        scope: str,
        status: str = ACTIVE,
    ) -> list[MemoryRecord]:
        if not subject or not attribute:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND project_id = ? AND type = ?
                  AND subject = ? AND attribute = ? AND scope = ? AND status = ?
                ORDER BY version DESC, updated_at DESC
                """,
                (user_id, project_id, memory_type, subject, attribute, scope, status),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def update(self, memory_id: str, record: MemoryRecord) -> MemoryRecord | None:
        """Replace the content of an existing memory while preserving its identity."""

        now = utc_now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET content = ?, normalized_content = ?, type = ?, status = ?,
                    source = ?, subject = ?, attribute = ?, value = ?, polarity = ?,
                    scope = ?, confidence = ?, effective_at = ?, updated_at = ?,
                    metadata_json = ?
                WHERE memory_id = ? AND user_id = ? AND project_id = ?
                """,
                (
                    record.content,
                    normalize_content(record.content),
                    record.type,
                    ACTIVE,
                    record.source,
                    record.subject,
                    record.attribute,
                    record.value,
                    record.polarity,
                    record.scope,
                    record.confidence,
                    record.effective_at,
                    now,
                    _to_json(record.metadata),
                    memory_id,
                    record.user_id,
                    record.project_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(memory_id)

    def replace_version(
        self,
        previous: MemoryRecord,
        current: MemoryRecord,
    ) -> MemoryRecord:
        """Atomically supersede one memory and insert its next version."""

        return self.replace_versions([previous], current)

    def supersede_with_existing(
        self,
        previous: Sequence[MemoryRecord],
        current: MemoryRecord,
    ) -> MemoryRecord:
        """Archive older nodes in favor of an already-persisted newer node.

        MemOS first stores the new node and resolves conflicts afterwards. This
        helper preserves that ordering while still recording an auditable
        supersession chain in one SQLite transaction.
        """

        targets = list(
            {
                item.memory_id: item for item in previous if item.memory_id != current.memory_id
            }.values()
        )
        if not targets:
            return current
        if any(
            item.user_id != current.user_id or item.project_id != current.project_id
            for item in targets
        ):
            raise ValueError("all supersession targets must share user and project scope")
        now = utc_now_iso()
        target_ids = [item.memory_id for item in targets]
        current.version = max([current.version, *(item.version + 1 for item in targets)])
        current.supersedes_id = target_ids[0]
        current.metadata = {
            **current.metadata,
            "supersedes_ids": target_ids,
        }
        placeholders = ",".join("?" for _ in target_ids)
        with self._connect() as conn:
            current_row = conn.execute(
                """
                SELECT status FROM memories
                WHERE memory_id = ? AND user_id = ? AND project_id = ?
                """,
                (current.memory_id, current.user_id, current.project_id),
            ).fetchone()
            if current_row is None or current_row["status"] != ACTIVE:
                raise ValueError("current MemOS node must already exist and be active")
            rows = conn.execute(
                f"""
                SELECT memory_id, status FROM memories
                WHERE memory_id IN ({placeholders})
                  AND user_id = ? AND project_id = ?
                """,
                [*target_ids, current.user_id, current.project_id],
            ).fetchall()
            active_ids = {str(row["memory_id"]) for row in rows if row["status"] == ACTIVE}
            if active_ids != set(target_ids):
                raise ValueError("all supersession targets must exist and be active")
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET status = ?, superseded_by_id = ?, updated_at = ?
                WHERE memory_id IN ({placeholders}) AND status = ?
                """,
                [SUPERSEDED, current.memory_id, now, *target_ids, ACTIVE],
            )
            if cursor.rowcount != len(target_ids):
                raise RuntimeError("failed to supersede every MemOS target")
            conn.execute(
                """
                UPDATE memories
                SET version = ?, supersedes_id = ?, updated_at = ?, metadata_json = ?
                WHERE memory_id = ? AND status = ?
                """,
                (
                    current.version,
                    current.supersedes_id,
                    now,
                    _to_json(current.metadata),
                    current.memory_id,
                    ACTIVE,
                ),
            )
        saved = self.get(current.memory_id)
        if saved is None:
            raise RuntimeError("MemOS successor disappeared after supersession")
        return saved

    def replace_versions(
        self,
        previous: Sequence[MemoryRecord],
        current: MemoryRecord,
    ) -> MemoryRecord:
        """Atomically supersede multiple active memories with one merged version."""

        targets = list({item.memory_id: item for item in previous}.values())
        if not targets:
            raise ValueError("replace_versions requires at least one target")
        if any(
            item.user_id != current.user_id or item.project_id != current.project_id
            for item in targets
        ):
            raise ValueError("all replacement targets must share user and project scope")
        now = utc_now_iso()
        target_ids = [item.memory_id for item in targets]
        current.version = max(item.version for item in targets) + 1
        current.supersedes_id = target_ids[0]
        current.metadata = {
            **current.metadata,
            "supersedes_ids": target_ids,
        }
        current.created_at = now
        current.updated_at = now
        placeholders = ",".join("?" for _ in target_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id, user_id, project_id, status
                FROM memories
                WHERE memory_id IN ({placeholders})
                """,
                target_ids,
            ).fetchall()
            active_ids = {
                str(row["memory_id"])
                for row in rows
                if row["status"] == ACTIVE
                and row["user_id"] == current.user_id
                and row["project_id"] == current.project_id
            }
            if active_ids != set(target_ids):
                raise ValueError("all replacement targets must exist and be active")
            collision = conn.execute(
                f"""
                SELECT memory_id FROM memories
                WHERE user_id = ? AND project_id = ?
                  AND normalized_content = ? AND status = ?
                  AND memory_id NOT IN ({placeholders})
                """,
                [
                    current.user_id,
                    current.project_id,
                    normalize_content(current.content),
                    ACTIVE,
                    *target_ids,
                ],
            ).fetchone()
            if collision:
                raise ValueError("merged content duplicates an active memory outside target_ids")
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET status = ?, superseded_by_id = ?, updated_at = ?
                WHERE memory_id IN ({placeholders}) AND status = ?
                """,
                [SUPERSEDED, current.memory_id, now, *target_ids, ACTIVE],
            )
            if cursor.rowcount != len(target_ids):
                raise RuntimeError("failed to supersede every replacement target")
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, project_id, content, normalized_content,
                    type, status, source, subject, attribute, value, polarity,
                    scope, confidence, version, supersedes_id, superseded_by_id,
                    effective_at, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current.memory_id,
                    current.user_id,
                    current.project_id,
                    current.content,
                    normalize_content(current.content),
                    current.type,
                    ACTIVE,
                    current.source,
                    current.subject,
                    current.attribute,
                    current.value,
                    current.polarity,
                    current.scope,
                    current.confidence,
                    current.version,
                    current.supersedes_id,
                    None,
                    current.effective_at,
                    now,
                    now,
                    _to_json(current.metadata),
                ),
            )
        saved = self.get(current.memory_id)
        if saved is None:
            raise RuntimeError("failed to load newly inserted memory version")
        return saved

    def history(self, memory_id: str) -> list[MemoryRecord]:
        history: list[MemoryRecord] = []
        current = self.get(memory_id)
        while current is not None:
            history.append(current)
            current = self.get(current.supersedes_id) if current.supersedes_id else None
        return history

    def forget(self, memory_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE memory_id IN ({placeholders}) AND status != ?
                """,
                [FORGOTTEN, utc_now_iso(), *ids, FORGOTTEN],
            )
            return int(cursor.rowcount)

    def delete(self, memory_ids: Iterable[str]) -> int:
        """Hard-delete memories (mem0 style)."""
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE memory_id IN ({placeholders})",
                ids,
            )
            return int(cursor.rowcount)

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def _ensure_schema(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate_schema(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_structured_slot
                ON memories(
                    user_id, project_id, type, subject, attribute, scope, status
                )
                """
            )
            conn.execute("DROP INDEX IF EXISTS idx_memories_scope_content")
            conn.execute(
                """
                CREATE UNIQUE INDEX idx_memories_scope_content
                ON memories(user_id, project_id, normalized_content)
                WHERE status = 'active'
                """
            )

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        additions = {
            "subject": "TEXT NOT NULL DEFAULT ''",
            "attribute": "TEXT NOT NULL DEFAULT ''",
            "value": "TEXT NOT NULL DEFAULT ''",
            "polarity": "TEXT NOT NULL DEFAULT 'positive'",
            "scope": "TEXT NOT NULL DEFAULT 'project'",
            "confidence": "REAL NOT NULL DEFAULT 1.0",
            "version": "INTEGER NOT NULL DEFAULT 1",
            "supersedes_id": "TEXT",
            "superseded_by_id": "TEXT",
            "effective_at": "TEXT NOT NULL DEFAULT ''",
            "expiration_date": "TEXT",
            "attributed_to": "TEXT NOT NULL DEFAULT 'user'",
        }
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def normalize_content(content: str) -> str:
    return " ".join(content.strip().lower().split())


def _to_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _from_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_to_organization_job(row: sqlite3.Row) -> dict[str, Any]:
    result = _from_json(row["result_json"])
    return {
        "job_id": str(row["job_id"]),
        "memory_id": str(row["memory_id"]),
        "status": str(row["status"]),
        "attempts": int(row["attempts"]),
        "max_attempts": int(row["max_attempts"]),
        "lease_until": row["lease_until"],
        "error": row["last_error"],
        "result": result,
        **result,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        project_id=row["project_id"],
        content=row["content"],
        type=row["type"],
        status=row["status"],
        source=row["source"],
        subject=row["subject"],
        attribute=row["attribute"],
        value=row["value"],
        polarity=row["polarity"],
        scope=row["scope"],
        confidence=float(row["confidence"]),
        version=int(row["version"]),
        supersedes_id=row["supersedes_id"],
        superseded_by_id=row["superseded_by_id"],
        effective_at=row["effective_at"] or row["created_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expiration_date=row["expiration_date"] if "expiration_date" in row.keys() else None,
        attributed_to=row["attributed_to"] if "attributed_to" in row.keys() else "user",
        metadata=_from_json(row["metadata_json"]),
    )
