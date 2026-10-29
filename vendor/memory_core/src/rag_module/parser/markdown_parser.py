"""Markdown document parser."""

from __future__ import annotations

import re
from pathlib import Path

from src.rag_module.parser.models import DocumentBlock, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```")


class MarkdownParser:
    """Parse Markdown files into structured DocumentBlocks."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8")
        lines = text.split("\n")
        blocks: list[DocumentBlock] = []
        heading_stack: list[str] = []
        title: str | None = None

        i = 0
        while i < len(lines):
            line = lines[i]

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                if title is None and level == 1:
                    title = heading_text
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading_text)
                blocks.append(
                    DocumentBlock(
                        block_type="heading",
                        text=heading_text,
                        heading_level=level,
                        heading_path=tuple(heading_stack),
                    )
                )
                i += 1
                continue

            if _CODE_FENCE_RE.match(line):
                code_lines: list[str] = []
                code_lines.append(line)
                i += 1
                while i < len(lines) and not _CODE_FENCE_RE.match(lines[i]):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])
                    i += 1
                blocks.append(
                    DocumentBlock(
                        block_type="code",
                        text="\n".join(code_lines),
                        heading_path=tuple(heading_stack),
                    )
                )
                continue

            if line.startswith("|") and "|" in line[1:]:
                table_lines: list[str] = []
                while i < len(lines) and lines[i].startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(
                    DocumentBlock(
                        block_type="table",
                        text="\n".join(table_lines),
                        heading_path=tuple(heading_stack),
                    )
                )
                continue

            if line.strip() == "":
                i += 1
                continue

            para_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "" and not _HEADING_RE.match(lines[i]) and not _CODE_FENCE_RE.match(lines[i]):
                para_lines.append(lines[i])
                i += 1
            if para_lines:
                blocks.append(
                    DocumentBlock(
                        block_type="paragraph",
                        text="\n".join(para_lines),
                        heading_path=tuple(heading_stack),
                    )
                )

        return ParsedDocument(title=title, blocks=blocks, metadata={"source": str(file_path)})
