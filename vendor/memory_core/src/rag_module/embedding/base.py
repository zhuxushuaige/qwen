"""EmbeddingProvider interface and EmbeddingInfo."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EmbeddingInfo:
    """Metadata about an embedding model."""

    provider: str
    model_id: str
    dimension: int
    normalized: bool


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def info(self) -> EmbeddingInfo:
        """Return metadata about this embedding provider."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple document texts."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries in a batch while preserving query-specific semantics.

        Providers with a dedicated query instruction should override this method.
        The compatibility implementation intentionally delegates one query at a
        time instead of assuming document and query encoding are identical.
        """

        return [self.embed_query(text) for text in texts]

    @property
    def cache_revision(self) -> str:
        """Return a stable revision for persistent embedding caches."""

        return "1"

    def release(self) -> None:
        """Release optional model/runtime resources held by the provider."""

        return None
