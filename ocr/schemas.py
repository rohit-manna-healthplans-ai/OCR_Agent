from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Word:
    text: str
    conf: float
    bbox: List[int]          # [x1, y1, x2, y2]


@dataclass
class Line:
    text: str
    conf: float
    bbox: List[int]
    words: List[Word]
    line_index: int = 0      # reading-order index within its block


@dataclass
class Block:
    """
    A contiguous group of lines that share a common layout role.
    block_type values:
        paragraph   - flowing body text
        heading     - large / bold standalone line
        table_row   - line that is part of a detected table
        table       - reconstructed table (lines = rows, each row has pipe-separated cells)
        label_value - "Label:  Value" pattern (form fields)
        list_item   - bullet / numbered list line
        caption     - small text under an image or table
        header      - page header region
        footer      - page footer region
        sidebar     - narrow left/right column
        noise       - low-confidence or icon-only line (excluded from text output)
    """
    block_type: str
    lines: List[Line]
    bbox: List[int]
    reading_order: int = 0   # global reading order across the page


@dataclass
class Column:
    """One logical column on the page (for multi-column layouts)."""
    col_index: int
    x_range: List[int]       # [x_start, x_end]
    blocks: List[Block] = field(default_factory=list)


@dataclass
class PageResult:
    page_index: int
    width: int
    height: int
    text: str                # flat reading-order text (backward compat)
    lines: List[Line]        # all lines flat (backward compat)
    columns: List[Column] = field(default_factory=list)   # layout columns
    blocks: List[Block] = field(default_factory=list)     # reading-order blocks
    tables: List[Dict[str, Any]] = field(default_factory=list)  # extracted tables


def _block_to_dict(b: Block) -> Dict[str, Any]:
    return {
        "block_type": b.block_type,
        "reading_order": b.reading_order,
        "bbox": b.bbox,
        "text": "\n".join(ln.text for ln in b.lines),
        "lines": [
            {
                "line_index": ln.line_index,
                "text": ln.text,
                "conf": ln.conf,
                "bbox": ln.bbox,
                "words": [
                    {"text": w.text, "conf": w.conf, "bbox": w.bbox}
                    for w in ln.words
                ],
            }
            for ln in b.lines
        ],
    }


def _column_to_dict(c: Column) -> Dict[str, Any]:
    return {
        "col_index": c.col_index,
        "x_range": c.x_range,
        "blocks": [_block_to_dict(b) for b in c.blocks],
        "text": "\n".join(
            "\n".join(ln.text for ln in b.lines)
            for b in c.blocks
        ),
    }


def to_dict(page_results: List[PageResult], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": meta,
        "text": "\n\n".join([p.text for p in page_results]).strip(),
        "pages": [
            {
                "page_index": p.page_index,
                "width": p.width,
                "height": p.height,
                "text": p.text,
                # -- Layout-aware fields -------------------------------
                "columns": [_column_to_dict(c) for c in p.columns],
                "blocks": [_block_to_dict(b) for b in p.blocks],
                "tables": p.tables,
                # -- Backward-compatible flat lines --------------------
                "lines": [
                    {
                        "text": ln.text,
                        "conf": ln.conf,
                        "bbox": ln.bbox,
                        "words": [
                            {"text": w.text, "conf": w.conf, "bbox": w.bbox}
                            for w in ln.words
                        ],
                    }
                    for ln in p.lines
                ],
            }
            for p in page_results
        ],
    }
