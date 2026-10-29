"""DOCX document parser using python-docx."""

from __future__ import annotations

from pathlib import Path

from src.rag_module.parser.models import DocumentBlock, ParsedDocument


class DocxParser:
    """Parse DOCX files into structured DocumentBlocks."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".docx"]

    def parse(self, file_path: Path) -> ParsedDocument:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(str(file_path))
        blocks: list[DocumentBlock] = []
        title: str | None = None
        heading_stack: list[str] = []

        for element in _iter_block_items(doc):
            if isinstance(element, Paragraph):
                text = element.text.strip()
                if not text:
                    continue

                style_name = element.style.name.lower() if element.style else ""

                if "heading" in style_name:
                    level = element.style.name.replace("Heading ", "").strip()
                    try:
                        heading_level = int(level)
                    except ValueError:
                        heading_level = 1

                    heading_stack = heading_stack[: heading_level - 1]
                    heading_stack.append(text)

                    if heading_level == 1 and title is None:
                        title = text

                    blocks.append(
                        DocumentBlock(
                            block_type="heading",
                            text=text,
                            heading_level=heading_level,
                            heading_path=tuple(heading_stack),
                        )
                    )
                else:
                    blocks.append(
                        DocumentBlock(
                            block_type="paragraph",
                            text=text,
                            heading_path=tuple(heading_stack),
                        )
                    )

            elif isinstance(element, Table):
                table_lines: list[str] = []
                for row in element.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    table_lines.append("| " + " | ".join(cells) + " |")

                if table_lines:
                    blocks.append(
                        DocumentBlock(
                            block_type="table",
                            text="\n".join(table_lines),
                            heading_path=tuple(heading_stack),
                        )
                    )

        if title is None:
            props = doc.core_properties
            if props.title:
                title = props.title

        return ParsedDocument(
            title=title,
            blocks=blocks,
            metadata={"source": str(file_path)},
        )


def _iter_block_items(doc):
    """Iterate over paragraphs and tables in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)
