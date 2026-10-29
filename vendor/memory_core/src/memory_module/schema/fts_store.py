"""SQLite FTS5 full-text index over memory content for BM25 retrieval."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id,
    user_id,
    project_id,
    content,
    tokenize='unicode61 remove_diacritics 2'
);
"""

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]+")


class FTSMemoryStore:
    """FTS5 index providing BM25-style keyword search over memory content.

    Must be kept in sync with the ``memories`` table by the writer layer.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_FTS_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'memories_fts_ids'"
            ).fetchone()
            if not exists:
                # Keep existing FTS rowids and corpus statistics during migration.
                conn.execute(
                    "CREATE TABLE memories_fts_ids ("
                    "fts_rowid INTEGER PRIMARY KEY, memory_id TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE INDEX memories_fts_ids_memory_id ON memories_fts_ids(memory_id)"
                )
                conn.execute(
                    "INSERT INTO memories_fts_ids SELECT rowid, memory_id FROM memories_fts"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # sync helpers — called after memory create / update / delete
    # ------------------------------------------------------------------

    def upsert(self, memory_id: str, content: str, user_id: str, project_id: str) -> None:
        self.upsert_many([(memory_id, content, user_id, project_id)])

    def upsert_many(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        """Synchronize multiple active memories in one transaction."""

        values = list(rows)
        if not values:
            return
        grouped: dict[str, list[tuple[str, str, str]]] = {}
        for memory_id, content, user_id, project_id in values:
            grouped.setdefault(memory_id, []).append((content, user_id, project_id))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = set()
            for memory_id, desired in grouped.items():
                current = conn.execute(
                    "SELECT f.content, f.user_id, f.project_id "
                    "FROM memories_fts_ids AS ids JOIN memories_fts AS f "
                    "ON f.rowid = ids.fts_rowid WHERE ids.memory_id = ? "
                    "ORDER BY ids.fts_rowid", (memory_id,),
                ).fetchall()
                if current != desired:
                    changed.add(memory_id)
                    self._delete_ids(conn, memory_id)
            for memory_id, content, user_id, project_id in values:
                if memory_id not in changed:
                    continue
                cursor = conn.execute(
                    "INSERT INTO memories_fts(memory_id, user_id, project_id, content) "
                    "VALUES (?, ?, ?, ?)", (memory_id, user_id, project_id, content),
                )
                conn.execute(
                    "INSERT INTO memories_fts_ids VALUES (?, ?)",
                    (cursor.lastrowid, memory_id),
                )

    @staticmethod
    def _delete_ids(conn: sqlite3.Connection, memory_id: str) -> None:
        conn.execute(
            "DELETE FROM memories_fts WHERE rowid IN "
            "(SELECT fts_rowid FROM memories_fts_ids WHERE memory_id = ?)",
            (memory_id,),
        )
        conn.execute("DELETE FROM memories_fts_ids WHERE memory_id = ?", (memory_id,))

    def delete(self, memory_id: str) -> None:
        self.delete_many([memory_id])

    def delete_many(self, memory_ids: Iterable[str]) -> None:
        """Remove multiple inactive memories in one transaction."""

        values = list(dict.fromkeys(memory_ids))
        if not values:
            return
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for memory_id in values:
                self._delete_ids(conn, memory_id)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str,
        project_id: str,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Return ``(memory_id, bm25_score)`` sorted by relevance descending.

        FTS5 ``rank`` is a negative number (closer to 0 = more relevant).
        We negate it so higher scores mean more relevant, then normalize
        to [0, 1] in ``score_and_rank()``.
        """
        if top_k <= 0 or not query.strip():
            return []

        safe_query = _fts5_query(query)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, -rank AS bm25_score
                FROM memories_fts
                WHERE memories_fts MATCH ?
                  AND user_id = ?
                  AND project_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, user_id, project_id, top_k),
            ).fetchall()
        return [(row[0], float(row[1])) for row in rows]


def _fts5_query(query: str) -> str:
    """Escape user input for FTS5 MATCH.

    CJK characters are left unquoted so the unicode61 tokenizer can match
    them as single-character tokens.  Latin tokens are double-quoted.
    """
    tokens = query.strip().split()
    if not tokens:
        return '""'

    parts: list[str] = []
    for token in tokens:
        if _CJK_RE.fullmatch(token):
            # CJK: join individual chars with AND for phrase-like matching
            parts.append(" ".join(token))
        else:
            parts.append(f'"{token}"')
    return " AND ".join(parts)
