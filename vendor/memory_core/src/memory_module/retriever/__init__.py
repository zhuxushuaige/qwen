"""Retriever implementations with optional backends loaded on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "HybridMemoryRetriever": ("hybrid", "HybridMemoryRetriever"),
    "TieredMemoryRetriever": ("tiered", "TieredMemoryRetriever"),
    "MemoryRetriever": ("interface", "MemoryRetriever"),
    "FeatureHashEmbeddingProvider": ("sqlite", "FeatureHashEmbeddingProvider"),
    "SQLiteMemoryRetriever": ("sqlite", "SQLiteMemoryRetriever"),
    "TemporalDecay": ("temporal", "TemporalDecay"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
