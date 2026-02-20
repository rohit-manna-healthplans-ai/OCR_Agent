from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _format_table_cells(cells: List[List[str]]) -> str:
    """Return a readable, tag-friendly table text (pipe-separated rows)."""
    lines: List[str] = []
    for row in cells:
        row = [(_safe_str(c)).strip() for c in (row or [])]
        lines.append(" | ".join(row).rstrip())
    return "\n".join(lines).strip()


def _split_page_parts(pg: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Build header/main/footer text from page['lines'] and page['header_footer'] indices.
    If lines are missing, fallback to pg['text'] as main.
    """
    hf = (pg.get("header_footer") or {})
    header_idx = set(hf.get("header_lines_idx") or [])
    footer_idx = set(hf.get("footer_lines_idx") or [])

    lines = pg.get("lines") or []
    if not isinstance(lines, list) or not lines:
        # No per-line info; return pg['text'] in main
        return "", _safe_str(pg.get("text")).strip(), ""

    header_lines: List[str] = []
    main_lines: ListList[str] = []  # type: ignore[name-defined]
    footer_lines: List[str] = []

    for i, ln in enumerate(lines):
        txt = _safe_str((ln or {}).get("text")).strip()
        if not txt:
            continue
        if i in header_idx:
            header_lines.append(txt)
        elif i in footer_idx:
            footer_lines.append(txt)
        else:
            main_lines.append(txt)

    header_text = "\n".join(header_lines).strip()
    main_text = "\n".join(main_lines).strip()
    footer_text = "\n".join(footer_lines).strip()

    # If detector didn't mark anything, keep header/footer empty and treat all as main.
    if not header_text and not footer_text and _safe_str(pg.get("text")).strip():
        main_text = _safe_str(pg.get("text")).strip()

    return header_text, main_text, footer_text


def build_formatted_text(result: Dict[str, Any]) -> str:
    """
    Build UI-facing extracted text with page order + HEADER/FOOTER/TABLE tags.
    This is PURE OCR formatting (NO LLM).
    """
    pages = result.get("pages") or []
    out_lines: List[str] = []

    for pg in pages:
        page_no = pg.get("page_index")
        # some callers may prefer 1-based; keep existing page_index but show 1-based too if int
        try:
            shown_no = int(page_no) + 1
        except Exception:
            shown_no = page_no

        out_lines.append(f"=== PAGE {shown_no} ===")

        header_text, main_text, footer_text = _split_page_parts(pg)

        if header_text:
            out_lines.append("<HEADER>")
            out_lines.append(header_text)
            out_lines.append("</HEADER>")

        if main_text:
            out_lines.append("<BODY>")
            out_lines.append(main_text)
            out_lines.append("</BODY>")

        tables = pg.get("tables") or []
        if isinstance(tables, list) and tables:
            for ti, tb in enumerate(tables, start=1):
                cells = (tb or {}).get("cells") or []
                n_rows = (tb or {}).get("n_rows") or (len(cells) if isinstance(cells, list) else 0)
                n_cols = (tb or {}).get("n_cols") or (len(cells[0]) if isinstance(cells, list) and cells else 0)

                out_lines.append(f'<TABLE index="{ti}" rows="{n_rows}" cols="{n_cols}">')
                if isinstance(cells, list) and cells:
                    out_lines.append(_format_table_cells(cells))
                out_lines.append("</TABLE>")

        if footer_text:
            out_lines.append("<FOOTER>")
            out_lines.append(footer_text)
            out_lines.append("</FOOTER>")

        out_lines.append("")  # spacer

    return "\n".join(out_lines).strip()


def build_final_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build UI-facing structured JSON (NO LLM).
    - raw_text_full: original concatenated OCR text
    - cleaned_text_full: formatted text with tags (header/footer/table)
    - pages: per-page sections + blocks (text/tables)
    """
    pages = result.get("pages") or []
    formatted_text = build_formatted_text(result)

    structured_pages: List[Dict[str, Any]] = []
    for pg in pages:
        header_text, main_text, footer_text = _split_page_parts(pg)

        # Blocks: keep a simple, stable ordering: header -> main -> tables -> footer
        blocks: List[Dict[str, Any]] = []
        order = 1

        if header_text:
            blocks.append({"type": "header", "order": order, "raw": header_text, "cleaned": header_text, "bbox": None})
            order += 1

        if main_text:
            blocks.append({"type": "text", "order": order, "raw": main_text, "cleaned": main_text, "bbox": None})
            order += 1

        for tb in (pg.get("tables") or []) if isinstance(pg.get("tables"), list) else []:
            cells = (tb or {}).get("cells") or []
            blocks.append({
                "type": "table",
                "order": order,
                "raw": _format_table_cells(cells) if isinstance(cells, list) else "",
                "cleaned": _format_table_cells(cells) if isinstance(cells, list) else "",
                "bbox": (tb or {}).get("bbox"),
                "meta": {
                    "n_rows": (tb or {}).get("n_rows"),
                    "n_cols": (tb or {}).get("n_cols"),
                    "source": (tb or {}).get("source"),
                }
            })
            order += 1

        if footer_text:
            blocks.append({"type": "footer", "order": order, "raw": footer_text, "cleaned": footer_text, "bbox": None})
            order += 1

        structured_pages.append({
            "page_index": pg.get("page_index"),
            "header": {"raw": header_text, "cleaned": header_text, "line_idxs": (pg.get("header_footer") or {}).get("header_lines_idx") or []},
            "main": {"raw": main_text, "cleaned": main_text, "blocks": blocks},
            "footer": {"raw": footer_text, "cleaned": footer_text, "line_idxs": (pg.get("header_footer") or {}).get("footer_lines_idx") or []},
        })

    return {
        "raw_text_full": _safe_str(result.get("text")).strip(),
        "cleaned_text_full": formatted_text,
        "active_tab": None,
        "ui_elements": [],
        "tables": [],  # kept for backward-compat; per-page tables are in blocks
        "pages": structured_pages,
    }
