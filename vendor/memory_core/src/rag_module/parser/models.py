"""Parser intermediate representation (DocumentIR)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentBlock:
    """A structured block from document parsing."""

    block_type: str  # heading/paragraph/list/table/code/quote
    text: str
    heading_level: int | None = None
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Document parsed into structured blocks."""

    title: str | None
    blocks: list[DocumentBlock]
    metadata: dict[str, object] = field(default_factory=dict)
