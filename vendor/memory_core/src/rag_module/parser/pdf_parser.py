"""Layout-aware PDF extraction with bounded, local OCR for image-only pages."""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from src.rag_module.parser.models import DocumentBlock, ParsedDocument


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    # OCR inserts spaces between Chinese glyphs, including widely tracked headings.
    return re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text).strip()


def _join_words(words: list[dict]) -> str:
    return _clean(" ".join(word["text"] for word in words))


def _lines(words: list[dict]) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if not rows or abs(word["top"] - rows[-1][0]["top"]) > 3:
            rows.append([word])
        else:
            rows[-1].append(word)
    return [sorted(row, key=lambda w: w["x0"]) for row in rows]


def _cells(row: list[dict]) -> list[list[dict]]:
    cells: list[list[dict]] = []
    for word in row:
        if not cells or word["x0"] - cells[-1][-1]["x1"] > max(8, word["size"]):
            cells.append([word])
        else:
            cells[-1].append(word)
    return cells


def _is_numeric_row(row: list[dict]) -> bool:
    cells = [_join_words(cell) for cell in _cells(row)]
    return len(cells) >= 3 and sum(bool(re.match(r"^[~<>+\-]?\d", c)) for c in cells) >= 2


def _heading_level(row: list[dict], body_size: float) -> int | None:
    text = _join_words(row)
    if not text or len(text) > 120 or re.search(r"[.!?。！？;；:]$", text):
        return None
    size = statistics.median(w["size"] for w in row)
    bold = sum(
        len(w["text"]) for w in row
        if re.search(r"bold|medi|demi|black", w.get("fontname", ""), re.I)
    ) / max(1, sum(len(w["text"]) for w in row))
    if size >= body_size * 1.45:
        return 1
    if size >= body_size * 1.12 or (bold >= 0.8 and len(text) <= 80):
        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+\S", text)
        return min(6, match[1].count(".") + 1) if match else 2
    return None


class PdfParser:
    """Preserve physical pages, headings and table rows without a remote model."""

    def __init__(self, *, ocr_enabled: bool = True, ocr_languages: str = "chi_sim",
                 ocr_dpi: int = 300, ocr_timeout: int = 120) -> None:
        self.ocr_enabled = ocr_enabled
        self.ocr_languages = ocr_languages
        self.ocr_dpi = ocr_dpi
        self.ocr_timeout = ocr_timeout

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        import pdfplumber

        blocks: list[DocumentBlock] = []
        ocr_pages: list[int] = []
        heading_path: list[str] = []
        title = None
        with pdfplumber.open(file_path) as reader:
            for page in reader.pages:
                # Ignore rotated marginal labels, not upright document contents.
                upright = page.filter(lambda obj: obj.get("upright", True))
                words = upright.extract_words(x_tolerance=1, y_tolerance=3,
                                              extra_attrs=["fontname", "size"])
                if words:
                    page_blocks = self._text_blocks(words, page.page_number)
                elif self.ocr_enabled and (page.images or page.curves or page.rects):
                    # Some publishers convert every glyph to vector outlines;
                    # these pages have visible text but no image/text objects.
                    page_blocks = self._ocr_page(file_path, page.page_number)
                    ocr_pages.append(page.page_number)
                else:
                    page_blocks = []
                for block in page_blocks:
                    if block.block_type == "heading":
                        level = block.heading_level or 1
                        heading_path = heading_path[:level - 1] + [block.text]
                        if title is None:
                            title = block.text
                    block.heading_path = tuple(heading_path)
                    blocks.append(block)
                page.close()
            title = title or (reader.metadata or {}).get("Title")
            return ParsedDocument(title=title, blocks=blocks, metadata={
                "source": str(file_path), "total_pages": len(reader.pages),
                "parser_version": "pdf-layout-ocr-v1", "ocr_pages": ocr_pages,
                "ocr_languages": self.ocr_languages if ocr_pages else None,
            })

    def _text_blocks(self, words: list[dict], page_number: int) -> list[DocumentBlock]:
        sizes: Counter = Counter()
        for word in words:
            sizes[round(word["size"], 1)] += len(word["text"])
        body_size = sizes.most_common(1)[0][0]
        lines = _lines(words)
        result: list[DocumentBlock] = []
        pending: list[str] = []

        def add(kind: str, text: str, level: int | None = None) -> None:
            result.append(DocumentBlock(kind, text, heading_level=level,
                                        page_start=page_number, page_end=page_number))

        def flush() -> None:
            if pending:
                add("paragraph", _clean("\n".join(pending)))
                pending.clear()

        i = 0
        while i < len(lines):
            row = lines[i]
            # Require multiple numeric rows, not merely aligned prose or a TOC.
            if _is_numeric_row(row) and i + 1 < len(lines) and _is_numeric_row(lines[i + 1]):
                flush()
                table_rows: list[str] = []
                while i < len(lines) and _is_numeric_row(lines[i]):
                    table_rows.append(" | ".join(_join_words(c) for c in _cells(lines[i])))
                    i += 1
                add("table", "\n".join(table_rows))
                continue
            level = _heading_level(row, body_size)
            if level is not None:
                flush()
                add("heading", _join_words(row), level)
            else:
                if i and row[0]["top"] - max(w["bottom"] for w in lines[i - 1]) > body_size * 0.7:
                    flush()
                pending.append(_join_words(row))
            i += 1
        flush()
        return result

    def _ocr_page(self, path: Path, page_number: int) -> list[DocumentBlock]:
        if not shutil.which("pdftoppm") or not shutil.which("tesseract"):
            raise RuntimeError("Scanned PDF requires local poppler-utils and tesseract-ocr "
                               "with chi_sim/eng language data (or explicitly disable OCR)")
        with tempfile.TemporaryDirectory(prefix="kylin-pdf-ocr-") as tmp:
            prefix = str(Path(tmp) / "page")
            subprocess.run(["pdftoppm", "-f", str(page_number), "-l", str(page_number),
                            "-singlefile", "-r", str(self.ocr_dpi), "-png", str(path.resolve()), prefix],
                           check=True, capture_output=True, timeout=self.ocr_timeout)
            env = dict(os.environ, OMP_THREAD_LIMIT="2")

            def recognize(image: str) -> str:
                return subprocess.run(
                    ["tesseract", image, "stdout", "-l", self.ocr_languages,
                     "--psm", "3", "tsv"], check=True, capture_output=True, text=True,
                    encoding="utf-8", timeout=self.ocr_timeout, env=env,
                ).stdout

            output = recognize(prefix + ".png")
            words = _ocr_words(output)
            # Sparse covers with shaded lettering benefit from thresholding;
            # avoid thresholding dense body pages where fine strokes may vanish.
            if len(words) < 60:
                from PIL import Image, ImageOps

                with Image.open(prefix + ".png") as source:
                    binary = ImageOps.autocontrast(ImageOps.grayscale(source)).point(
                        lambda value: 0 if value < 190 else 255)
                    binary.save(prefix + "-binary.png")
                candidate = recognize(prefix + "-binary.png")
                if _ocr_confidence(candidate) > _ocr_confidence(output):
                    output = candidate
            return self._ocr_blocks(output, page_number)

    @staticmethod
    def _ocr_blocks(tsv: str, page_number: int) -> list[DocumentBlock]:
        groups: dict[tuple[str, str], dict[str, list[dict]]] = {}
        heights: list[float] = []
        for word in _ocr_words(tsv):
            groups.setdefault((word["block_num"], word["par_num"]), {}).setdefault(
                word["line_num"], []).append(word)
            heights.append(float(word["height"]))
        body_height = statistics.median(heights) if heights else 1
        blocks: list[DocumentBlock] = []
        for lines in groups.values():
            text = _clean("\n".join(_clean(" ".join(w["text"] for w in line))
                                   for line in lines.values()))
            line_words = [w for line in lines.values() for w in line]
            size = statistics.median(float(w["height"]) for w in line_words)
            is_heading = (len(lines) <= 2 and len(text) <= 80 and size >= body_height * 1.2
                          and not re.search(r"[。！？;；:]$", text))
            blocks.append(DocumentBlock("heading" if is_heading else "paragraph", text,
                                        heading_level=1 if is_heading else None,
                                        page_start=page_number, page_end=page_number,
                                        metadata={
                                            "extraction": "tesseract", "ocr": True,
                                            "bbox": [min(float(w["left"]) for w in line_words),
                                                     min(float(w["top"]) for w in line_words),
                                                     max(float(w["left"]) + float(w["width"]) for w in line_words),
                                                     max(float(w["top"]) + float(w["height"]) for w in line_words)],
                                        }))
        # Tesseract sometimes splits a widely tracked heading into independent
        # blocks on the same baseline. Join only single-line heading fragments.
        merged: list[DocumentBlock] = []
        for block in blocks:
            box = block.metadata["bbox"]
            match = next((b for b in merged if b.block_type == block.block_type == "heading"
                          and "\n" not in b.text and "\n" not in block.text
                          and abs(b.metadata["bbox"][1] - box[1]) < (box[3] - box[1]) * 0.35), None)
            if match is not None:
                left = match.metadata["bbox"]
                match.text = _clean(" ".join([match.text, block.text] if left[0] < box[0]
                                             else [block.text, match.text]))
                match.metadata["bbox"] = [min(left[0], box[0]), min(left[1], box[1]),
                                          max(left[2], box[2]), max(left[3], box[3])]
            else:
                merged.append(block)
        return merged


def _ocr_words(tsv: str) -> list[dict]:
    return [w for w in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
            if w.get("level") == "5" and (w.get("text") or "").strip()]


def _ocr_confidence(tsv: str) -> float:
    words = _ocr_words(tsv)
    return sum(float(w["conf"]) * len(w["text"]) for w in words) / max(
        1, sum(len(w["text"]) for w in words))
