"""SQLite schema initialization for RAG module."""

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL UNIQUE,
    project_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_uri TEXT,
    file_type TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    content_hash TEXT NOT NULL,
    parser_version TEXT,
    chunker_version TEXT,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing'
        CHECK(status IN ('processing','ready','failed','inactive')),
    error_message TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_project_status
    ON documents(project_id, status);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_pk INTEGER NOT NULL,
    project_id TEXT NOT NULL,
    parent_chunk_pk INTEGER,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    heading_path TEXT,
    page_start INTEGER,
    page_end INTEGER,
    token_count INTEGER,
    content_hash TEXT NOT NULL,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(embedding_status IN ('pending','processing','ready','failed')),
    status TEXT NOT NULL DEFAULT 'active',
    metadata TEXT,
    FOREIGN KEY (doc_pk) REFERENCES documents(doc_pk) ON DELETE CASCADE,
    FOREIGN KEY (parent_chunk_pk) REFERENCES chunks(chunk_pk),
    UNIQUE(doc_pk, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_pk, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_project_status
    ON chunks(project_id, status, embedding_status);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading_path,
    content,
    content='chunks',
    content_rowid='chunk_pk',
    tokenize='trigram'
);

CREATE TABLE IF NOT EXISTS embedding_indexes (
    index_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    distance_metric TEXT NOT NULL,
    normalized INTEGER NOT NULL,
    chunker_version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
"""

FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading_path, content)
    VALUES (new.chunk_pk, new.heading_path, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, content)
    VALUES ('delete', old.chunk_pk, old.heading_path, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, content)
    VALUES ('delete', old.chunk_pk, old.heading_path, old.content);
    INSERT INTO chunks_fts(rowid, heading_path, content)
    VALUES (new.chunk_pk, new.heading_path, new.content);
END;
"""


def init_database(conn: sqlite3.Connection) -> None:
    """Initialize RAG schema in the given connection."""
    if conn.row_factory is None:
        conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(FTS_TRIGGERS_SQL)
