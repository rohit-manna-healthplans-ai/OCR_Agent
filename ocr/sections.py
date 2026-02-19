
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _bbox_area(b: List[int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection(a: List[int], b: List[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _overlap_ratio(a: List[int], b: List[int]) -> float:
    inter = _bbox_intersection(a, b)
    if inter <= 0:
        return 0.0
    denom = float(min(_bbox_area(a), _bbox_area(b)) or 1)
    return inter / denom


def _lines_in_bbox(page: Dict[str, Any], bbox: List[int], min_ratio: float = 0.25) -> List[Dict[str, Any]]:
    lines = page.get("lines") or []
    out: List[Dict[str, Any]] = []
    for ln in lines:
        lb = ln.get("bbox")
        if not lb:
            continue
        if _overlap_ratio(lb, bbox) >= min_ratio:
            out.append(ln)
    return out


def _sort_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(b: Dict[str, Any]) -> Tuple[int, int]:
        bb = b.get("bbox") or [0, 0, 0, 0]
        return (int(bb[1]), int(bb[0]))
    return sorted(blocks, key=key)


def build_sections(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produce a structured reading order of the page.

    If layout blocks are available:
      - Sort blocks by (y1, x1)
      - Emit section items by type
      - For text/title/header/footer: prefer layout text, else collect OCR lines within bbox
      - For table: include cells when present in layout/table extraction; otherwise best-effort from detected tables

    If layout is unavailable:
      - Return a single text section with the page text.
    """
    layout = page.get("layout") or {}
    blocks = layout.get("blocks") or []
    available = bool(layout.get("available")) and bool(blocks)

    if not available:
        txt = (page.get("text") or "").strip()
        return [{"type": "text", "bbox": [0, 0, int(page.get("width") or 0), int(page.get("height") or 0)], "text": txt}]

    detected_tables = page.get("tables") or []
    blocks_sorted = _sort_blocks(blocks)

    sections: List[Dict[str, Any]] = []
    for b in blocks_sorted:
        btype = (b.get("type") or "unknown").lower()
        bbox = b.get("bbox") or [0, 0, 0, 0]

        if btype == "table":
            cells: List[List[str]] = []
            # Prefer cells that were parsed from this exact block (pipeline will attach)
            if "cells" in b and isinstance(b.get("cells"), list):
                cells = b.get("cells") or []
            # Otherwise, match best overlapping detected table
            if not cells and detected_tables:
                best = None
                best_r = 0.0
                for tb in detected_tables:
                    tbbox = tb.get("bbox") or [0, 0, 0, 0]
                    r = _overlap_ratio(tbbox, bbox)
                    if r > best_r:
                        best_r = r
                        best = tb
                if best is not None and best_r >= 0.25:
                    cells = best.get("cells") or []
            sections.append({"type": "table", "bbox": bbox, "cells": cells})
            continue

        # textual-ish blocks
        text = (b.get("text") or "").strip()
        if not text:
            # Collect OCR lines inside the bbox
            lines = _lines_in_bbox(page, bbox, min_ratio=0.25)
            text = "\n".join((ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip()).strip()

        # normalize type set
        if btype in ("header", "footer", "title", "text"):
            stype = btype
        else:
            # keep other types (figure, equation, list, etc.) as-is with text fallback
            stype = btype

        if stype in ("header", "footer") and not text:
            # Avoid empty header/footer noise
            continue

        sections.append({"type": stype, "bbox": bbox, "text": text})

    # Optional: merge consecutive text blocks to reduce fragmentation
    merged: List[Dict[str, Any]] = []
    for s in sections:
        if merged and s.get("type") == "text" and merged[-1].get("type") == "text":
            prev = merged[-1]
            prev_text = (prev.get("text") or "").strip()
            cur_text = (s.get("text") or "").strip()
            if prev_text and cur_text:
                prev["text"] = (prev_text + "\n" + cur_text).strip()
                # bbox union
                pb = prev.get("bbox") or [0, 0, 0, 0]
                cb = s.get("bbox") or [0, 0, 0, 0]
                prev["bbox"] = [min(pb[0], cb[0]), min(pb[1], cb[1]), max(pb[2], cb[2]), max(pb[3], cb[3])]
                continue
        merged.append(s)

    return merged
