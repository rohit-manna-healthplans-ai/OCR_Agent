from __future__ import annotations

"""
Compatibility table detector.

Your pipeline currently calls:
    pg["tables"] = detect_tables_from_page(pg, prefer_layout=True)

So this function MUST:
- accept (page, prefer_layout=..., **kwargs)
- return a list of tables
- (optionally) also write into page["tables"] for downstream consumers

This implementation:
- keeps a lightweight heuristic candidate builder from OCR words/lines
- filters false positives (especially UI screenshots) using table_validator
- does NOT crop screenshots
"""

from typing import Any, Dict, List, Optional

try:
    from .table_validator import validate_table, estimate_main_content_bbox
except Exception:
    validate_table = None  # type: ignore
    estimate_main_content_bbox = None  # type: ignore


def detect_tables_from_page(
    page: Dict[str, Any],
    prefer_layout: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Args:
      page: dict containing OCR results (expects "lines", "width", "height", optional "text")
      prefer_layout: accepted for backward compatibility. This implementation ignores it
                     unless your pipeline injects layout-derived tables elsewhere.
      **kwargs: ignored (compat)

    Returns:
      list of accepted table dicts.
    """
    lines: List[Dict[str, Any]] = page.get("lines") or []
    if not lines:
        tables = page.get("tables") or []
        page["tables"] = tables
        return tables

    page_w = int(page.get("width") or 0)
    page_h = int(page.get("height") or 0)
    page_text = page.get("text") or ""

    # If your pipeline already populated layout tables somewhere, you can keep them:
    # e.g. page.get("layout_tables") or similar. We won't assume that here.

    candidates: List[Dict[str, Any]] = _heuristic_tables_from_lines(lines)

    main_bbox: Optional[List[int]] = None
    if estimate_main_content_bbox is not None and page_w and page_h:
        try:
            main_bbox = estimate_main_content_bbox(lines, page_w, page_h)
        except Exception:
            main_bbox = None

    accepted: List[Dict[str, Any]] = []
    for t in candidates:
        if validate_table is None:
            accepted.append(t)
            continue

        ok, score, reasons = validate_table(
            t, page_w=page_w, page_h=page_h, page_text=page_text, main_content_bbox=main_bbox
        )
        t["score"] = float(score)
        t["reasons"] = list(reasons)
        if main_bbox is not None:
            t["main_content_bbox"] = main_bbox

        if ok:
            accepted.append(t)

    page["tables"] = accepted
    return accepted


# -------------------------
# Heuristic candidate builder
# -------------------------

def _heuristic_tables_from_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Collect word boxes
    words = []
    for ln in lines:
        for w in (ln.get("words") or []):
            txt = (w.get("text") or "").strip()
            if not txt:
                continue
            bbox = w.get("bbox") or None
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox]
            words.append(
                {
                    "text": txt,
                    "bbox": [x0, y0, x1, y1],
                    "cx": (x0 + x1) / 2.0,
                    "cy": (y0 + y1) / 2.0,
                }
            )

    if len(words) < 12:
        return []

    words.sort(key=lambda d: (d["cy"], d["cx"]))

    # Row clustering by y
    rows: List[List[Dict[str, Any]]] = []
    y_tol = 12
    for w in words:
        if not rows:
            rows.append([w])
            continue
        last_row = rows[-1]
        if abs(w["cy"] - last_row[-1]["cy"]) <= y_tol:
            last_row.append(w)
        else:
            rows.append([w])

    rows = [r for r in rows if len(r) >= 3]
    if len(rows) < 2:
        return []

    xs = sorted([w["cx"] for r in rows for w in r])
    if len(xs) < 6:
        return []

    # Column anchors
    col_tol = 26
    cols: List[float] = []
    for x in xs:
        if not cols or abs(x - cols[-1]) > col_tol:
            cols.append(x)
        else:
            cols[-1] = (cols[-1] + x) / 2.0

    if len(cols) < 2:
        return []

    # Build grid
    grid: List[List[str]] = []
    for r in rows:
        row_cells = [""] * len(cols)
        for w in r:
            ci = min(range(len(cols)), key=lambda i: abs(w["cx"] - cols[i]))
            row_cells[ci] = (row_cells[ci] + " " + w["text"]).strip() if row_cells[ci] else w["text"]
        grid.append([c.strip() for c in row_cells])

    x0 = min(w["bbox"][0] for w in words)
    y0 = min(w["bbox"][1] for w in words)
    x1 = max(w["bbox"][2] for w in words)
    y1 = max(w["bbox"][3] for w in words)

    return [
        {
            "bbox": [x0, y0, x1, y1],
            "n_rows": len(grid),
            "n_cols": len(cols),
            "cells": grid,
            "source": "heuristic",
            "score": 0.0,
            "reasons": [],
        }
    ]