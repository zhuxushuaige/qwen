"""Hybrid chunker: Parent-Child strategy with structure-aware splitting."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from src.rag_module.chunking.models import Chunk
from src.rag_module.parser.models import DocumentBlock, ParsedDocument

# Child chunks: 80-200 tokens for vector search
DEFAULT_CHILD_TARGET_TOKENS = 120
DEFAULT_CHILD_MAX_TOKENS = 200
DEFAULT_CHILD_MIN_TOKENS = 10

# Parent chunks: 300-700 tokens for LLM context
DEFAULT_PARENT_TARGET_TOKENS = 500
DEFAULT_PARENT_MAX_TOKENS = 700
DEFAULT_SEMANTIC_UNIT_MAX_TOKENS = 600

DEFAULT_OVERLAP_RATIO = 0.12
RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]
_QUESTION_LINE_RE = re.compile(r"^\s*(?:question|q|问题|问)\s*[:：]", re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"^\s*(?:answer|a|答案|答)\s*[:：]", re.IGNORECASE)


@dataclass
class ChunkerConfig:
    child_target_tokens: int = DEFAULT_CHILD_TARGET_TOKENS
    child_max_tokens: int = DEFAULT_CHILD_MAX_TOKENS
    child_min_tokens: int = DEFAULT_CHILD_MIN_TOKENS
    parent_target_tokens: int = DEFAULT_PARENT_TARGET_TOKENS
    parent_max_tokens: int = DEFAULT_PARENT_MAX_TOKENS
    semantic_unit_max_tokens: int = DEFAULT_SEMANTIC_UNIT_MAX_TOKENS
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO
    chunker_version: str = "parent_child_v2"


class HybridChunker:
    """Parent-Child chunking: small chunks for search, large chunks for LLM context."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(
        self,
        doc: ParsedDocument,
        doc_id: str,
        project_id: str,
    ) -> list[Chunk]:
        """Generate parent and child chunks. Returns only child chunks for indexing.

        Parent chunks are stored as metadata on child chunks so the retrieval
        pipeline can expand to parent context when needed.
        """
        # Step 1: Build parent-level blocks by merging small blocks
        parent_blocks = self._build_parent_blocks(doc.blocks)

        # Step 2: For each parent block, generate child chunks
        all_chunks: list[Chunk] = []
        for parent_block in parent_blocks:
            parent_text = parent_block.text.strip()
            if not parent_text:
                continue

            # Skip standalone headings with no body (only heading text, no paragraph content)
            if parent_block.block_type == "heading" and "\n" not in parent_text and len(parent_text) < 20:
                continue

            heading_path = " > ".join(parent_block.heading_path) if parent_block.heading_path else None

            # Generate children for this parent
            children_texts = self._split_into_children(parent_block)

            if not children_texts:
                continue

            parent_hash = self._content_hash(parent_text, heading_path)
            parent_id = self._parent_id(
                doc_id=doc_id,
                parent_hash=parent_hash,
                page_start=parent_block.page_start,
            )

            # Create child chunks
            for child_idx, child_text in enumerate(children_texts):
                child_text = child_text.strip()
                if not child_text or _estimate_tokens(child_text) < self.config.child_min_tokens:
                    continue

                chunk_id = str(uuid.uuid4())
                child_hash = self._content_hash(child_text, heading_path)

                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        project_id=project_id,
                        chunk_index=len(all_chunks),
                        content=child_text,
                        heading_path=heading_path,
                        page_start=parent_block.page_start,
                        page_end=parent_block.page_end,
                        token_count=_estimate_tokens(child_text),
                        content_hash=child_hash,
                        parent_chunk_id=None,  # Don't set FK, store in metadata
                        metadata={
                            "chunker_version": self.config.chunker_version,
                            "parent_id": parent_id,
                            "parent_hash": parent_hash,
                            "parent_content": parent_text,
                            "parent_token_count": _estimate_tokens(parent_text),
                            "child_index": child_idx,
                        },
                    )
                )

        return all_chunks

    def _build_parent_blocks(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        """Merge small blocks into parent-level blocks of 300-700 tokens."""
        if not blocks:
            return []

        merged: list[DocumentBlock] = []
        current: DocumentBlock | None = None

        def flush() -> None:
            nonlocal current
            if current is not None and current.text.strip():
                merged.extend(self._split_oversized_parent(current))
            current = None

        for block in blocks:
            # Headings are represented by heading_path and form a hard section
            # boundary. They are not indexed as standalone chunks.
            if block.block_type == "heading":
                flush()
                continue

            # Structured blocks keep their type so table/list child splitting
            # can apply the appropriate strategy.
            if block.block_type in ("code", "table"):
                flush()
                merged.extend(self._split_oversized_parent(self._copy_block(block)))
                continue

            if current is None:
                current = self._copy_block(block)
                continue

            same_section = current.heading_path == block.heading_path
            # Preserve the source paragraph boundary.  A single newline loses
            # the distinction between related paragraphs and arbitrary wrapped
            # lines, making later semantic grouping needlessly destructive.
            combined = current.text + "\n\n" + block.text
            if same_section and _estimate_tokens(combined) <= self.config.parent_max_tokens:
                current.text = combined
                current.page_end = block.page_end if block.page_end is not None else current.page_end
            else:
                flush()
                current = self._copy_block(block)

        flush()
        return merged

    @staticmethod
    def _copy_block(block: DocumentBlock, *, text: str | None = None) -> DocumentBlock:
        return DocumentBlock(
            block_type=block.block_type,
            text=block.text if text is None else text,
            heading_level=block.heading_level,
            heading_path=block.heading_path,
            page_start=block.page_start,
            page_end=block.page_end,
            metadata=dict(block.metadata),
        )

    def _split_oversized_parent(self, block: DocumentBlock) -> list[DocumentBlock]:
        if _estimate_tokens(block.text) <= self.config.parent_max_tokens:
            return [block]
        pieces = self._split_preserving_semantic_units(
            block.text,
            self.config.parent_max_tokens,
        )
        return [self._copy_block(block, text=piece) for piece in pieces if piece.strip()]

    def _split_into_children(self, block: DocumentBlock) -> list[str]:
        """Split a parent block into child chunks of 80-200 tokens."""
        text = block.text.strip()
        if not text:
            return []

        tokens = _estimate_tokens(text)

        # If small enough, return as single child
        if tokens <= self.config.child_max_tokens:
            return [text]

        # Handle tables specially: split by rows
        if block.block_type == "table":
            return self._split_table(text)

        # Handle lists: split by semantic groups
        if block.block_type == "list":
            return self._split_list(text)

        # Paragraphs often encode an atomic question/answer record or a short
        # lead paragraph followed by its explanation.  Pack those semantic
        # units first and use recursive splitting only inside an oversized
        # ordinary unit.  A Q/A pair may exceed child_max_tokens, up to the
        # explicit semantic-unit ceiling, because separating its answer makes
        # the resulting retrieval chunk unusable.
        return self._split_preserving_semantic_units(
            text,
            self.config.child_max_tokens,
            allow_qa_overflow=True,
        )

    def _split_preserving_semantic_units(
        self,
        text: str,
        max_tokens: int,
        *,
        allow_qa_overflow: bool = False,
    ) -> list[str]:
        units = _semantic_units(text)
        if not units:
            return []

        pieces: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                pieces.append("\n\n".join(current).strip())
                current.clear()

        for unit in units:
            unit_tokens = _estimate_tokens(unit)
            qa_atomic = (
                allow_qa_overflow
                and _is_question_answer_unit(unit)
                and unit_tokens <= self.config.semantic_unit_max_tokens
            )
            if unit_tokens > max_tokens and not qa_atomic:
                flush()
                pieces.extend(self._recursive_split_to_limit(unit, max_tokens, 0))
                continue
            candidate = "\n\n".join([*current, unit])
            if current and _estimate_tokens(candidate) > max_tokens:
                flush()
            current.append(unit)
            if qa_atomic and unit_tokens > max_tokens:
                flush()
        flush()
        return [piece for piece in pieces if piece]

    def _split_table(self, text: str) -> list[str]:
        """Split table into header + row groups."""
        lines = text.strip().split("\n")
        if len(lines) <= 1:
            return self._recursive_split_to_limit(text, self.config.child_max_tokens, 0)

        # Find header (first non-separator line)
        header_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not all(c in "-| " for c in line):
                header_idx = i
                break

        header = lines[header_idx]

        # Find separator line
        sep_idx = header_idx + 1
        if sep_idx < len(lines) and all(c in "-| " for c in lines[sep_idx].strip()):
            sep_idx += 1

        # Group data rows
        children: list[str] = []
        current_rows: list[str] = []
        current_tokens = _estimate_tokens(header)

        for line in lines[sep_idx:]:
            if not line.strip():
                continue
            row_tokens = _estimate_tokens(line)
            if current_tokens + row_tokens > self.config.child_max_tokens and current_rows:
                # Emit child with header + rows
                children.append(header + "\n" + "\n".join(current_rows))
                current_rows = []
                current_tokens = _estimate_tokens(header)
            current_rows.append(line)
            current_tokens += row_tokens

        if current_rows:
            children.append(header + "\n" + "\n".join(current_rows))

        # A single wide row (or the repeated header plus one row) can exceed
        # the budget even when ordinary row grouping fits. Enforce the bound.
        return [
            piece
            for child in (children if children else [text])
            for piece in self._recursive_split_to_limit(child, self.config.child_max_tokens, 0)
        ]

    def _split_list(self, text: str) -> list[str]:
        """Split list into semantic groups."""
        lines = text.strip().split("\n")
        if len(lines) <= 1:
            return [text]

        children: list[str] = []
        current_group: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = _estimate_tokens(line)
            if current_tokens + line_tokens > self.config.child_max_tokens and current_group:
                children.append("\n".join(current_group))
                current_group = []
                current_tokens = 0
            current_group.append(line)
            current_tokens += line_tokens

        if current_group:
            children.append("\n".join(current_group))

        return children if children else [text]

    def _recursive_split(self, text: str, sep_index: int) -> list[str]:
        """Recursively split text into child-sized chunks."""
        return self._recursive_split_to_limit(text, self.config.child_max_tokens, sep_index)

    def _recursive_split_to_limit(
        self,
        text: str,
        max_tokens: int,
        sep_index: int,
    ) -> list[str]:
        if _estimate_tokens(text) <= max_tokens:
            return [text.strip()] if text.strip() else []

        if sep_index >= len(RECURSIVE_SEPARATORS):
            return self._hard_split(text, max_tokens)

        sep = RECURSIVE_SEPARATORS[sep_index]
        if sep == "":
            return self._hard_split(text, max_tokens)

        parts = text.split(sep)
        if len(parts) <= 1:
            return self._recursive_split_to_limit(text, max_tokens, sep_index + 1)

        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if _estimate_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current.strip():
                    pieces.append(current.strip())
                if _estimate_tokens(part) > max_tokens:
                    pieces.extend(self._recursive_split_to_limit(part, max_tokens, sep_index + 1))
                else:
                    current = part
                    continue
                current = ""
        if current.strip():
            pieces.append(current.strip())

        return pieces

    def _hard_split(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Hard split by character count as last resort."""
        target = min(self.config.child_target_tokens, max_tokens or self.config.child_max_tokens)
        approx_chars = target * 2
        pieces: list[str] = []
        for i in range(0, len(text), approx_chars):
            chunk = text[i : i + approx_chars].strip()
            if chunk:
                pieces.append(chunk)
        return pieces

    def _content_hash(self, text: str, heading_path: str | None) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        if heading_path:
            h.update(heading_path.encode("utf-8"))
        h.update(self.config.chunker_version.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _parent_id(
        *,
        doc_id: str,
        parent_hash: str,
        page_start: int | None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(doc_id.encode("utf-8"))
        digest.update(parent_hash.encode("ascii"))
        digest.update(str(page_start).encode("ascii"))
        return f"parent-{digest.hexdigest()[:24]}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


def _semantic_units(text: str) -> list[str]:
    """Return paragraph units while binding adjacent Q/A lines together."""

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]
    units: list[str] = []
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        lines = paragraph.splitlines()
        has_question = any(_QUESTION_LINE_RE.match(line) for line in lines)
        has_answer = any(_ANSWER_LINE_RE.match(line) for line in lines)
        if has_question and not has_answer and index + 1 < len(paragraphs):
            next_lines = paragraphs[index + 1].splitlines()
            if any(_ANSWER_LINE_RE.match(line) for line in next_lines):
                units.append(paragraph + "\n\n" + paragraphs[index + 1])
                index += 2
                continue
        units.append(paragraph)
        index += 1
    return units


def _is_question_answer_unit(text: str) -> bool:
    lines = text.splitlines()
    return any(_QUESTION_LINE_RE.match(line) for line in lines) and any(
        _ANSWER_LINE_RE.match(line) for line in lines
    )
