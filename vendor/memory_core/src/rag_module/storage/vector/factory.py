"""Vector-store backend selection shared by production and evaluation paths."""

from __future__ import annotations

import os
import sqlite3

from src.rag_module.storage.vector.base import VectorStore
from src.rag_module.storage.vector.sqlite_vec import SqliteVecStore
from src.storage.vector.kylin_engine import KylinVectorEngineError, default_vector_store_backend

VECTOR_STORE_BACKENDS = ("sqlite", "kylin")


def create_vector_store(
    conn: sqlite3.Connection,
    dimension: int,
    *,
    backend: str | None = None,
    fallback: str | None = None,
    drop_on_close: bool = False,
    collection_identity: str | None = None,
) -> VectorStore:
    selected = (backend or default_vector_store_backend()).strip().lower()
    fallback_name = (
        fallback if fallback is not None else os.environ.get("VECTOR_STORE_FALLBACK", "none")
    ).strip().lower()
    if selected not in VECTOR_STORE_BACKENDS:
        raise ValueError(
            f"unknown vector-store backend {selected!r}; expected one of {VECTOR_STORE_BACKENDS}"
        )
    if fallback_name not in {"none", "sqlite"}:
        raise ValueError("vector-store fallback must be 'none' or 'sqlite'")
    if selected == "sqlite":
        return SqliteVecStore(conn, dimension=dimension)
    from src.rag_module.storage.vector.kylin_engine import KylinVectorStore

    try:
        return KylinVectorStore(
            conn,
            dimension,
            fallback_on_error=fallback_name == "sqlite",
            drop_on_close=drop_on_close,
            collection_identity=collection_identity,
        )
    except KylinVectorEngineError:
        if fallback_name != "sqlite":
            raise
        fallback_store = SqliteVecStore(conn, dimension=dimension)
        fallback_store.last_backend = "sqlite-vec-fallback"
        fallback_store.fallback_count = 1
        return fallback_store


def close_vector_store(store: VectorStore) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        close()
