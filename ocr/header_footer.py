from __future__ import annotations

from typing import Any, Dict, List


def _line_center_y(line: Dict[str, Any]) -> float:
    b = line.get("bbox") or [0, 0, 0, 0]
    return (float(b[1]) + float(b[3])) / 2.0


def detect_header_footer_from_page(page: Dict[str, Any]) -> Dict[str, Any]:
    # Header/Footer detection from line bboxes (no extra OCR pass).
    h = int(page.get("height") or 0)
    lines: List[Dict[str, Any]] = page.get("lines") or []
    if not lines or h <= 0:
        return {
            "available": False,
            "header_text": "",
            "footer_text": "",
            "header_lines_idx": [],
            "footer_lines_idx": [],
        }

    top_y = 0.12 * h
    bottom_y = 0.88 * h

    header_idx = []
    footer_idx = []
    for i, ln in enumerate(lines):
        cy = _line_center_y(ln)
        if cy <= top_y:
            header_idx.append(i)
        elif cy >= bottom_y:
            footer_idx.append(i)

    header_text = "\n".join((lines[i].get("text") or "").strip() for i in header_idx).strip()
    footer_text = "\n".join((lines[i].get("text") or "").strip() for i in footer_idx).strip()

    if len(header_text) < 3:
        header_text = ""
        header_idx = []
    if len(footer_text) < 3:
        footer_text = ""
        footer_idx = []

    return {
        "available": bool(header_text or footer_text),
        "header_text": header_text,
        "footer_text": footer_text,
        "header_lines_idx": header_idx,
        "footer_lines_idx": footer_idx,
    }
