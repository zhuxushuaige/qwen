"""Memory module package with lazy imports to avoid qdrant_client dependency at import time."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MemoryQwenEmbeddingProvider":
        from src.memory_module.schema.embedding import MemoryQwenEmbeddingProvider
        return MemoryQwenEmbeddingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MemoryQwenEmbeddingProvider"]
