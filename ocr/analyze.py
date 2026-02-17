from __future__ import annotations

from typing import Any, Dict, List


def _avg_conf(lines: List[dict]) -> float:
    if not lines:
        return 0.0
    vals = []
    for ln in lines:
        c = ln.get("conf")
        if isinstance(c, (int, float)):
            vals.append(float(c))
    return sum(vals) / max(1, len(vals))


def analyze_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a production-friendly "analysis" block:
    - quality score (0..100)
    - warnings
    - per-page stats
    - what was extracted (text/fields/tables)
    """
    meta = result.get("meta", {}) or {}
    pages = result.get("pages", []) or []
    fields = result.get("fields", {}) or {}

    warnings: List[str] = []
    page_stats: List[Dict[str, Any]] = []

    total_lines = 0
    total_words = 0
    conf_sum = 0.0
    conf_count = 0

    for p in pages:
        lines = p.get("lines", []) or []
        total_lines += len(lines)
        p_words = 0
        p_confs = []
        for ln in lines:
            ws = ln.get("words", []) or []
            p_words += len(ws)
            c = ln.get("conf")
            if isinstance(c, (int, float)):
                p_confs.append(float(c))
        total_words += p_words
        if p_confs:
            conf_sum += sum(p_confs)
            conf_count += len(p_confs)

        tables = p.get("tables", []) or []
        page_stats.append(
            {
                "page_index": p.get("page_index"),
                "width": p.get("width"),
                "height": p.get("height"),
                "lines": len(lines),
                "words": p_words,
                "avg_line_conf": (sum(p_confs) / len(p_confs)) if p_confs else 0.0,
                "tables": len(tables),
            }
        )

    avg_conf = (conf_sum / conf_count) if conf_count else 0.0
    text_len = len((result.get("text") or "").strip())

    # Quality heuristic
    score = 0.0
    score += min(40.0, avg_conf * 40.0)
    score += min(30.0, text_len / 500.0 * 30.0)
    score += 10.0 if fields else 0.0
    score += 10.0 if any((p.get("tables") or []) for p in pages) else 0.0

    if meta.get("digital_pdf") is True:
        warnings.append("Digital PDF detected: extracted selectable text. No bboxes/fields/tables from OCR on those pages.")

    if text_len < 60:
        warnings.append("Extracted text is very short. Try preset=photo, higher DPI, or engine=tesseract.")

    if avg_conf < 0.55 and total_lines > 0:
        warnings.append("Low average confidence. Consider better scan quality / preprocessing.")

    if not fields:
        warnings.append("No structured fields detected (expected on forms). Use boxed/block-letter feature or template rules.")

    return {
        "quality_score_0_100": round(max(0.0, min(100.0, score)), 2),
        "avg_line_conf": round(avg_conf, 4),
        "text_chars": text_len,
        "pages": len(pages),
        "total_lines": total_lines,
        "total_words": total_words,
        "engine_used": meta.get("engine_used") or meta.get("engine") or "unknown",
        "preset_used": meta.get("preset_used") or meta.get("preset_requested") or "unknown",
        "warnings": warnings,
        "page_stats": page_stats,
    }
