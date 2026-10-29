"""Memory module schema package with lazy imports to avoid qdrant_client dependency at import time."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MemoryQwenEmbeddingProvider":
        from src.memory_module.schema.embedding import MemoryQwenEmbeddingProvider
        return MemoryQwenEmbeddingProvider
    if name in ("Entity", "EntityStore"):
        from src.memory_module.schema.entity_store import Entity, EntityStore
        return {"Entity": Entity, "EntityStore": EntityStore}[name]
    if name == "FTSMemoryStore":
        from src.memory_module.schema.fts_store import FTSMemoryStore
        return FTSMemoryStore
    if name == "LayeredMemoryStore":
        from src.memory_module.schema.layered_store import LayeredMemoryStore
        return LayeredMemoryStore
    if name in ("ACTIVE", "FORGOTTEN", "SUPERSEDED", "DEFAULT_MEMORY_TYPE",
                 "PREFERENCE_MEMORY", "TASK_MEMORY", "PROJECT_MEMORY",
                 "MemoryCandidate", "MemoryRecord"):
        from src.memory_module.schema import models
        return getattr(models, name)
    if name == "SQLiteMemoryStore":
        from src.memory_module.schema.sqlite_store import SQLiteMemoryStore
        return SQLiteMemoryStore
    if name in ("SQLiteMemoryVectorStore", "MemoryVectorHit", "distance_to_cosine_similarity"):
        from src.memory_module.schema import vector_store
        return getattr(vector_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemoryQwenEmbeddingProvider",
    "Entity",
    "EntityStore",
    "FTSMemoryStore",
    "LayeredMemoryStore",
    "ACTIVE",
    "FORGOTTEN",
    "SUPERSEDED",
    "DEFAULT_MEMORY_TYPE",
    "PREFERENCE_MEMORY",
    "TASK_MEMORY",
    "PROJECT_MEMORY",
    "MemoryCandidate",
    "MemoryRecord",
    "SQLiteMemoryStore",
    "SQLiteMemoryVectorStore",
    "MemoryVectorHit",
    "distance_to_cosine_similarity",
]
