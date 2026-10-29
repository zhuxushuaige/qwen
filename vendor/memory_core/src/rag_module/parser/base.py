"""Document parser protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.rag_module.parser.models import ParsedDocument


class DocumentParser(Protocol):
    """Parse a document file into structured blocks."""

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this parser handles."""
        ...

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a file into a ParsedDocument."""
        ...
