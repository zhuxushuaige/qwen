"""Persistent L0/L2/L3 state for TencentDB-style personalized memory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.memory_module.schema.models import utc_now_iso

L0_ACTIVE_TTL_MINUTES = 30
MEDIUM_TERM_TTL_DAYS = 30
MEDIUM_KIND_SCENE = "scene"
MEDIUM_KIND_SESSION_SUMMARY = "session_summary"
MEDIUM_KIND_TASK_STATE = "task_state"
MEDIUM_KIND_SCRATCHPAD = "scratchpad"
LIFECYCLE_ACTIVE = "active"
LIFECYCLE_CONSOLIDATED = "consolidated"
LIFECYCLE_PROMOTED = "promoted"
LIFECYCLE_ARCHIVED = "archived"
LIFECYCLE_EXPIRED = "expired"


class LayeredMemoryStore:
    """Store raw conversations, scene blocks, extraction state, and personas."""

    def __init__(self, db_path: str | Path = ".memory_module.sqlite3") -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def capture_turn(
        self,
        user_id: str,
        project_id: str,
        user_content: str,
        assistant_content: str | None = None,
        *,
        conversation_id: str | None = None,
        extra_messages: Sequence[tuple[str, str]] = (),
    ) -> list[str]:
        # conversation_id is a session identifier and may repeat across turns.
        # Append a nonce so every turn remains present in the immutable L0 log.
        turn_id = f"{conversation_id or 'turn'}:{uuid.uuid4().hex}"
        now = utc_now_iso()
        expires_at = _future_iso(minutes=L0_ACTIVE_TTL_MINUTES)
        captured: list[str] = []
        with self._connect() as conn:
            messages = [
                ("user", user_content),
                ("assistant", assistant_content or ""),
                *extra_messages,
            ]
            for role, content in messages:
                if not content.strip():
                    continue
                if role not in {"user", "assistant", "tool", "system"}:
                    raise ValueError(f"unsupported L0 message role: {role}")
                message_id = f"msg_{uuid.uuid4().hex[:16]}"
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO l0_conversations (
                        message_id, user_id, project_id, conversation_id,
                        role, content, lifecycle_status, expires_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        user_id,
                        project_id,
                        turn_id,
                        role,
                        content.strip(),
                        LIFECYCLE_ACTIVE,
                        expires_at,
                        now,
                    ),
                )
                if cursor.rowcount:
                    captured.append(message_id)
            conn.execute(
                """
                INSERT INTO layered_extraction_state (
                    user_id, project_id, total_turns, next_trigger_turn,
                    extracted_through_rowid, previous_scene_name,
                    l1_memory_count, persona_memory_count
                ) VALUES (?, ?, 1, 1, 0, '', 0, 0)
                ON CONFLICT(user_id, project_id) DO UPDATE SET
                    total_turns = total_turns + 1
                """,
                (user_id, project_id),
            )
        return captured

    def recent_messages(
        self,
        user_id: str,
        project_id: str,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return unexpired L0 provenance for extraction and exact backtracking."""

        if limit <= 0:
            return []
        conditions = [
            "user_id = ?",
            "project_id = ?",
            "lifecycle_status != ?",
            "(expires_at IS NULL OR expires_at > ?)",
        ]
        params: list[Any] = [
            user_id,
            project_id,
            LIFECYCLE_EXPIRED,
            utc_now_iso(),
        ]
        if conversation_id:
            conditions.append("conversation_id LIKE ?")
            params.append(f"{conversation_id}:%")
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rowid, message_id, conversation_id, role, content, "
                "lifecycle_status, expires_at, created_at FROM l0_conversations WHERE "
                + " AND ".join(conditions)
                + " ORDER BY rowid DESC LIMIT ?",
                params,
            ).fetchall()
        return [_message_payload(row) for row in reversed(rows)]

    def extraction_due(
        self,
        user_id: str,
        project_id: str,
        *,
        force: bool = False,
    ) -> bool:
        state = self.get_state(user_id, project_id)
        return force or state["total_turns"] >= state["next_trigger_turn"]

    def extraction_messages(
        self,
        user_id: str,
        project_id: str,
        *,
        max_new: int = 10,
        max_background: int = 5,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        state = self.get_state(user_id, project_id)
        with self._connect() as conn:
            new_rows = conn.execute(
                """
                SELECT rowid, message_id, role, content, created_at
                FROM l0_conversations
                WHERE user_id = ? AND project_id = ? AND rowid > ?
                ORDER BY rowid ASC LIMIT ?
                """,
                (
                    user_id,
                    project_id,
                    state["extracted_through_rowid"],
                    max_new,
                ),
            ).fetchall()
            first_rowid = (
                int(new_rows[0]["rowid"]) if new_rows else state["extracted_through_rowid"] + 1
            )
            background_rows = conn.execute(
                """
                SELECT rowid, message_id, role, content, created_at FROM (
                    SELECT rowid, message_id, role, content, created_at
                    FROM l0_conversations
                    WHERE user_id = ? AND project_id = ? AND rowid < ?
                    ORDER BY rowid DESC LIMIT ?
                ) ORDER BY rowid ASC
                """,
                (user_id, project_id, first_rowid, max_background),
            ).fetchall()
        return (
            [_message_payload(row) for row in new_rows],
            [_message_payload(row) for row in background_rows],
        )

    def complete_extraction(
        self,
        user_id: str,
        project_id: str,
        *,
        through_rowid: int,
        previous_scene_name: str,
        extracted_count: int,
        every_n_turns: int,
    ) -> None:
        state = self.get_state(user_id, project_id)
        current_trigger = state["next_trigger_turn"]
        if current_trigger * 2 < every_n_turns:
            next_trigger = max(state["total_turns"] + 1, current_trigger * 2)
        else:
            next_trigger = state["total_turns"] + every_n_turns
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE layered_extraction_state
                SET extracted_through_rowid = ?, previous_scene_name = ?,
                    l1_memory_count = l1_memory_count + ?,
                    next_trigger_turn = ?
                WHERE user_id = ? AND project_id = ?
                """,
                (
                    through_rowid,
                    previous_scene_name,
                    extracted_count,
                    next_trigger,
                    user_id,
                    project_id,
                ),
            )
            conn.execute(
                """
                UPDATE l0_conversations
                SET lifecycle_status = ?, consolidated_at = ?
                WHERE user_id = ? AND project_id = ? AND rowid <= ?
                  AND lifecycle_status = ?
                """,
                (
                    LIFECYCLE_CONSOLIDATED,
                    utc_now_iso(),
                    user_id,
                    project_id,
                    through_rowid,
                    LIFECYCLE_ACTIVE,
                ),
            )

    def upsert_scenes(
        self,
        user_id: str,
        project_id: str,
        scenes: Sequence[dict[str, Any]],
    ) -> list[str]:
        ids: list[str] = []
        now = utc_now_iso()
        expires_at = _future_iso(days=MEDIUM_TERM_TTL_DAYS)
        with self._connect() as conn:
            # Serialize the read-merge-write cycle so concurrent consolidations
            # cannot both observe an absent scene and insert the same key.
            conn.execute("BEGIN IMMEDIATE")
            for scene in scenes:
                name = str(scene["scene_name"]).strip()
                existing = conn.execute(
                    """
                    SELECT scene_id, summary, source_message_ids_json, importance
                    FROM memory_scenes
                    WHERE user_id = ? AND project_id = ? AND scene_name = ?
                    """,
                    (user_id, project_id, name),
                ).fetchone()
                summary = str(scene["summary"]).strip()
                source_ids = list(dict.fromkeys(scene["source_message_ids"]))
                conversation_id = str(scene.get("conversation_id", ""))
                task_id = str(scene.get("task_id", ""))
                kind = str(scene.get("kind", MEDIUM_KIND_SCENE))
                state = scene.get("state", {})
                importance = min(1.0, max(0.0, float(scene.get("importance", 0.5))))
                if existing:
                    scene_id = str(existing["scene_id"])
                    old_summary = str(existing["summary"])
                    if summary and summary not in old_summary:
                        summary = f"{old_summary}\n{summary}"
                    else:
                        summary = old_summary
                    source_ids = list(
                        dict.fromkeys(
                            [
                                *_from_json_list(existing["source_message_ids_json"]),
                                *source_ids,
                            ]
                        )
                    )
                    conn.execute(
                        """
                        UPDATE memory_scenes
                        SET summary = ?, source_message_ids_json = ?,
                            changed = 1, lifecycle_status = ?, expires_at = ?,
                            conversation_id = ?, task_id = ?, kind = ?,
                            state_json = ?, importance = ?, archived_at = NULL,
                            updated_at = ?
                        WHERE scene_id = ?
                        """,
                        (
                            summary,
                            _to_json(source_ids),
                            LIFECYCLE_ACTIVE,
                            expires_at,
                            conversation_id,
                            task_id,
                            kind,
                            _to_json(state),
                            max(float(existing["importance"]), importance),
                            now,
                            scene_id,
                        ),
                    )
                else:
                    scene_id = f"scene_{uuid.uuid4().hex[:16]}"
                    conn.execute(
                        """
                        INSERT INTO memory_scenes (
                            scene_id, user_id, project_id, scene_name, summary,
                            source_message_ids_json, changed, lifecycle_status,
                            conversation_id, task_id, kind, state_json, importance,
                            expires_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scene_id,
                            user_id,
                            project_id,
                            name,
                            summary,
                            _to_json(source_ids),
                            LIFECYCLE_ACTIVE,
                            conversation_id,
                            task_id,
                            kind,
                            _to_json(state),
                            importance,
                            expires_at,
                            now,
                            now,
                        ),
                    )
                ids.append(scene_id)
        return ids

    def changed_scenes(
        self,
        user_id: str,
        project_id: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT scene_id, scene_name, summary, source_message_ids_json
                FROM memory_scenes
                WHERE user_id = ? AND project_id = ? AND changed = 1
                  AND lifecycle_status = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at ASC
                """,
                (user_id, project_id, LIFECYCLE_ACTIVE, utc_now_iso()),
            ).fetchall()
        return [
            {
                "scene_id": row["scene_id"],
                "scene_name": row["scene_name"],
                "summary": row["summary"],
                "source_message_ids": _from_json_list(row["source_message_ids_json"]),
            }
            for row in rows
        ]

    def list_scenes(
        self,
        user_id: str,
        project_id: str,
        *,
        limit: int | None = None,
        task_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the latest L2 scenes for default macro-memory injection."""

        if limit is not None and limit <= 0:
            raise ValueError("scene limit must be positive")
        query = """
            SELECT scene_id, scene_name, summary, source_message_ids_json,
                   changed, lifecycle_status, access_count, last_accessed_at,
                   promoted_memory_id, expires_at, updated_at,
                   conversation_id, task_id, kind, state_json, importance,
                   archived_at
            FROM memory_scenes
            WHERE user_id = ? AND project_id = ? AND lifecycle_status = ?
              AND (expires_at IS NULL OR expires_at > ?)
        """
        parameters: list[Any] = [
            user_id,
            project_id,
            LIFECYCLE_ACTIVE,
            utc_now_iso(),
        ]
        if task_id:
            query += " AND (task_id = '' OR task_id = ?)"
            parameters.append(task_id)
        if conversation_id:
            query += " AND (conversation_id = '' OR conversation_id = ?)"
            parameters.append(conversation_id)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [
            {
                "scene_id": row["scene_id"],
                "scene_name": row["scene_name"],
                "summary": row["summary"],
                "source_message_ids": _from_json_list(row["source_message_ids_json"]),
                "changed": bool(row["changed"]),
                "lifecycle_status": row["lifecycle_status"],
                "access_count": int(row["access_count"]),
                "last_accessed_at": row["last_accessed_at"],
                "promoted_memory_id": row["promoted_memory_id"],
                "expires_at": row["expires_at"],
                "updated_at": row["updated_at"],
                "conversation_id": row["conversation_id"],
                "task_id": row["task_id"],
                "kind": row["kind"],
                "state": _from_json_dict(row["state_json"]),
                "importance": float(row["importance"]),
                "archived_at": row["archived_at"],
            }
            for row in rows
        ]

    def save_medium_memory(
        self,
        user_id: str,
        project_id: str,
        *,
        scene_name: str,
        summary: str,
        kind: str,
        conversation_id: str = "",
        task_id: str = "",
        state: dict[str, Any] | None = None,
        source_message_ids: Sequence[str] = (),
        importance: float = 0.5,
    ) -> dict[str, Any]:
        """Persist one session summary, task state, scene, or scratchpad entry."""

        scene_id = self.upsert_scenes(
            user_id,
            project_id,
            [
                {
                    "scene_name": scene_name,
                    "summary": summary,
                    "source_message_ids": list(source_message_ids),
                    "conversation_id": conversation_id,
                    "task_id": task_id,
                    "kind": kind,
                    "state": state or {},
                    "importance": importance,
                }
            ],
        )[0]
        scene = self.get_scene(scene_id)
        assert scene is not None
        return scene

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT scene_id, user_id, project_id, scene_name, summary,
                       source_message_ids_json, changed, lifecycle_status,
                       access_count, last_accessed_at, promoted_memory_id,
                       expires_at, updated_at, conversation_id, task_id, kind,
                       state_json, importance, archived_at
                FROM memory_scenes WHERE scene_id = ?
                """,
                (scene_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "scene_id": row["scene_id"],
            "user_id": row["user_id"],
            "project_id": row["project_id"],
            "scene_name": row["scene_name"],
            "summary": row["summary"],
            "source_message_ids": _from_json_list(row["source_message_ids_json"]),
            "changed": bool(row["changed"]),
            "lifecycle_status": row["lifecycle_status"],
            "access_count": int(row["access_count"]),
            "last_accessed_at": row["last_accessed_at"],
            "promoted_memory_id": row["promoted_memory_id"],
            "expires_at": row["expires_at"],
            "updated_at": row["updated_at"],
            "conversation_id": row["conversation_id"],
            "task_id": row["task_id"],
            "kind": row["kind"],
            "state": _from_json_dict(row["state_json"]),
            "importance": float(row["importance"]),
            "archived_at": row["archived_at"],
        }

    def mark_scene_promoted(self, scene_id: str, memory_id: str) -> int:
        if not scene_id or not memory_id:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_scenes
                SET lifecycle_status = ?, promoted_memory_id = ?, updated_at = ?
                WHERE scene_id = ? AND lifecycle_status IN (?, ?)
                """,
                (
                    LIFECYCLE_PROMOTED,
                    memory_id,
                    utc_now_iso(),
                    scene_id,
                    LIFECYCLE_ACTIVE,
                    LIFECYCLE_PROMOTED,
                ),
            )
        return int(cursor.rowcount)

    def promotion_candidates(
        self,
        user_id: str,
        project_id: str,
        *,
        min_importance: float = 0.8,
        min_access_count: int = 3,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT scene_id FROM memory_scenes
                WHERE user_id = ? AND project_id = ? AND lifecycle_status = ?
                  AND promoted_memory_id IS NULL
                  AND (importance >= ? OR access_count >= ?)
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY importance DESC, access_count DESC, updated_at DESC
                LIMIT ?
                """,
                (
                    user_id,
                    project_id,
                    LIFECYCLE_ACTIVE,
                    min_importance,
                    min_access_count,
                    utc_now_iso(),
                    limit,
                ),
            ).fetchall()
        return [scene for row in rows if (scene := self.get_scene(str(row["scene_id"])))]

    def messages_by_ids(
        self,
        user_id: str,
        project_id: str,
        message_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Resolve L1/L2 provenance IDs back to chronological L0 evidence."""

        unique_ids = list(dict.fromkeys(str(item) for item in message_ids if str(item)))
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT rowid, message_id, conversation_id, role, content, created_at
                FROM l0_conversations
                WHERE user_id = ? AND project_id = ?
                  AND message_id IN ({placeholders})
                ORDER BY rowid ASC
                """,
                [user_id, project_id, *unique_ids],
            ).fetchall()
        return [_message_payload(row) for row in rows]

    def record_scene_access(self, scene_ids: Sequence[str]) -> None:
        ids = list(dict.fromkeys(scene_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE memory_scenes
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE scene_id IN ({placeholders}) AND lifecycle_status = ?
                """,
                [utc_now_iso(), *ids, LIFECYCLE_ACTIVE],
            )

    def expire_lifecycle(self) -> dict[str, int]:
        """Expire active L0 provenance and archive stale medium-term records."""

        now = utc_now_iso()
        with self._connect() as conn:
            l0_provenance = conn.execute(
                """
                UPDATE l0_conversations SET lifecycle_status = ?
                WHERE lifecycle_status != ? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (LIFECYCLE_EXPIRED, LIFECYCLE_EXPIRED, now),
            ).rowcount
            medium = conn.execute(
                """
                UPDATE memory_scenes SET lifecycle_status = ?, archived_at = ?
                WHERE lifecycle_status = ? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (LIFECYCLE_ARCHIVED, now, LIFECYCLE_ACTIVE, now),
            ).rowcount
        return {"l0_provenance": int(l0_provenance), "medium_term": int(medium)}

    def persona_due(
        self,
        user_id: str,
        project_id: str,
        every_n_memories: int,
    ) -> bool:
        state = self.get_state(user_id, project_id)
        return state["l1_memory_count"] - state["persona_memory_count"] >= every_n_memories

    def save_persona(
        self,
        user_id: str,
        project_id: str,
        content: str,
        scene_ids: Sequence[str],
    ) -> None:
        now = utc_now_iso()
        state = self.get_state(user_id, project_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_personas (
                    user_id, project_id, content, source_scene_ids_json,
                    version, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(user_id, project_id) DO UPDATE SET
                    content = excluded.content,
                    source_scene_ids_json = excluded.source_scene_ids_json,
                    version = memory_personas.version + 1,
                    updated_at = excluded.updated_at
                """,
                (user_id, project_id, content, _to_json(list(scene_ids)), now),
            )
            if scene_ids:
                placeholders = ",".join("?" for _ in scene_ids)
                conn.execute(
                    f"""
                    UPDATE memory_scenes SET changed = 0
                    WHERE scene_id IN ({placeholders})
                    """,
                    list(scene_ids),
                )
            conn.execute(
                """
                UPDATE layered_extraction_state
                SET persona_memory_count = ?
                WHERE user_id = ? AND project_id = ?
                """,
                (state["l1_memory_count"], user_id, project_id),
            )

    def get_persona(self, user_id: str, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT content, source_scene_ids_json, version, updated_at
                FROM memory_personas
                WHERE user_id = ? AND project_id = ?
                """,
                (user_id, project_id),
            ).fetchone()
        if not row:
            return None
        return {
            "content": row["content"],
            "source_scene_ids": _from_json_list(row["source_scene_ids_json"]),
            "version": int(row["version"]),
            "updated_at": row["updated_at"],
        }

    def get_state(self, user_id: str, project_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM layered_extraction_state
                WHERE user_id = ? AND project_id = ?
                """,
                (user_id, project_id),
            ).fetchone()
        if not row:
            return {
                "total_turns": 0,
                "next_trigger_turn": 1,
                "extracted_through_rowid": 0,
                "previous_scene_name": "",
                "l1_memory_count": 0,
                "persona_memory_count": 0,
            }
        return {key: row[key] for key in row.keys()}

    def _ensure_schema(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS l0_conversations (
                    message_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL DEFAULT 'active',
                    expires_at TEXT,
                    consolidated_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, project_id, conversation_id, role)
                );
                CREATE INDEX IF NOT EXISTS idx_l0_scope_row
                ON l0_conversations(user_id, project_id, created_at);

                CREATE TABLE IF NOT EXISTS layered_extraction_state (
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    total_turns INTEGER NOT NULL DEFAULT 0,
                    next_trigger_turn INTEGER NOT NULL DEFAULT 1,
                    extracted_through_rowid INTEGER NOT NULL DEFAULT 0,
                    previous_scene_name TEXT NOT NULL DEFAULT '',
                    l1_memory_count INTEGER NOT NULL DEFAULT 0,
                    persona_memory_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id, project_id)
                );

                CREATE TABLE IF NOT EXISTS memory_scenes (
                    scene_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    scene_name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    changed INTEGER NOT NULL DEFAULT 1,
                    lifecycle_status TEXT NOT NULL DEFAULT 'active',
                    conversation_id TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'scene',
                    state_json TEXT NOT NULL DEFAULT '{}',
                    importance REAL NOT NULL DEFAULT 0.5,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT,
                    expires_at TEXT,
                    promoted_memory_id TEXT,
                    archived_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, project_id, scene_name)
                );

                CREATE TABLE IF NOT EXISTS memory_personas (
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_scene_ids_json TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, project_id)
                );
                """
            )
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        additions = {
            "l0_conversations": {
                "lifecycle_status": "TEXT NOT NULL DEFAULT 'active'",
                "expires_at": "TEXT",
                "consolidated_at": "TEXT",
            },
            "memory_scenes": {
                "lifecycle_status": "TEXT NOT NULL DEFAULT 'active'",
                "importance": "REAL NOT NULL DEFAULT 0.5",
                "access_count": "INTEGER NOT NULL DEFAULT 0",
                "last_accessed_at": "TEXT",
                "expires_at": "TEXT",
                "promoted_memory_id": "TEXT",
                "conversation_id": "TEXT NOT NULL DEFAULT ''",
                "task_id": "TEXT NOT NULL DEFAULT ''",
                "kind": "TEXT NOT NULL DEFAULT 'scene'",
                "state_json": "TEXT NOT NULL DEFAULT '{}'",
                "archived_at": "TEXT",
            },
        }
        for table, columns in additions.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _message_payload(row: sqlite3.Row) -> dict[str, Any]:
    keys = set(row.keys())
    payload = {
        "rowid": int(row["rowid"]),
        "message_id": row["message_id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
    if "conversation_id" in keys:
        payload["conversation_id"] = row["conversation_id"]
    if "lifecycle_status" in keys:
        payload["lifecycle_status"] = row["lifecycle_status"]
    if "expires_at" in keys:
        payload["expires_at"] = row["expires_at"]
    return payload


def _future_iso(*, minutes: int = 0, days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes, days=days)).isoformat(
        timespec="seconds"
    )


def _to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _from_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _from_json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}
