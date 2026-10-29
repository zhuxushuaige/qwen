"""Plain text document parser."""

from __future__ import annotations

from pathlib import Path

from src.rag_module.parser.models import DocumentBlock, ParsedDocument


class TxtParser:
    """Parse plain text files into structured DocumentBlocks."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8")
        paragraphs = text.split("\n\n")
        blocks: list[DocumentBlock] = []

        for para in paragraphs:
            stripped = para.strip()
            if stripped:
                blocks.append(
                    DocumentBlock(
                        block_type="paragraph",
                        text=stripped,
                        heading_path=(),
                    )
                )

        return ParsedDocument(
            title=file_path.stem,
            blocks=blocks,
            metadata={"source": str(file_path)},
        )
