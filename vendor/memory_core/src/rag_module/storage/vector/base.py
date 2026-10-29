"""VectorStore protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class VectorHit:
    """A vector hit with native ranking distance.

    Backends not returning cosine distance expose ``cosine_distance`` in
    metadata for RAG scoring. Native distance remains available to callers.
    """

    chunk_pk: int
    distance: float
    metadata: dict[str, Any] | None = None


class VectorStore(Protocol):
    """Store and search vector embeddings."""

    def upsert(
        self,
        chunk_pk: int,
        embedding: list[float],
        project_id: str,
        doc_pk: int,
        is_active: bool = True,
    ) -> None:
        """Insert or update a vector entry."""
        ...

    def upsert_batch(
        self,
        entries: list[tuple[int, list[float], str, int, bool]],
    ) -> None:
        """Batch insert or update vector entries.

        Each entry is (chunk_pk, embedding, project_id, doc_pk, is_active).
        """
        ...

    def search(
        self,
        query_vector: list[float],
        k: int,
        project_id: str | None = None,
        is_active: bool | None = None,
    ) -> list[VectorHit]:
        """Search for nearest neighbors."""
        ...

    def delete_by_doc(self, doc_pk: int) -> None:
        """Delete all vectors for a document."""
        ...

    def deactivate_by_doc(self, doc_pk: int) -> None:
        """Mark all vectors for a document as inactive."""
        ...

    def activate_by_doc(self, doc_pk: int) -> None:
        """Mark all vectors for a document as active."""
        ...
