"""Chunker protocol."""

from __future__ import annotations

from typing import Protocol

from src.rag_module.chunking.models import Chunk
from src.rag_module.parser.models import ParsedDocument


class Chunker(Protocol):
    """Split a parsed document into chunks."""

    def chunk(
        self,
        doc: ParsedDocument,
        doc_id: str,
        project_id: str,
    ) -> list[Chunk]:
        """Split document into chunks with metadata."""
        ...
