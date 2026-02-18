
"""
Enterprise multi-table detector (region-aware, form-grid aware).

Fixes:
- Multiple tables per page/region
- Avoids treating OS taskbar / browser chrome as tables
- Classifies tables:
  - kind="table" (multi-row data table)
  - kind="form_grid" (2-row label/value grid -> best converted to fields)
  - kind="noise" (UI junk, timestamps etc.)
"""

from typing import List, Dict, Any, Optional
import re


_TASKBAR_TOKENS = {
    "qsearch", "eng", "in", "mostly-clear", "mostly clear", "am", "pm",
    "wifi", "battery", "windows", "start"
}
_URL_RE = re.compile(r"(https?://|www\.)|(\.[a-z]{2,}/)", re.I)


def _bbox_union(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _y_center(w):
    return (w["bbox"][1] + w["bbox"][3]) / 2


def _x_center(w):
    return (w["bbox"][0] + w["bbox"][2]) / 2


def _alpha_ratio(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    letters = sum(c.isalpha() for c in s)
    return letters / max(1, len(s))


def _digit_ratio(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    digits = sum(c.isdigit() for c in s)
    return digits / max(1, len(s))


def _looks_like_ui_noise(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if _URL_RE.search(t):
        return True
    for tok in _TASKBAR_TOKENS:
        if tok in t:
            return True
    # time-like
    if re.search(r"\b\d{1,2}:\d{2}\s?(am|pm)\b", t, re.I):
        return True
    # date-like very short in UI bar (often appears with time)
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", t) and len(t) < 18:
        return True
    return False


def _cluster_rows(words: List[dict], y_eps: int = 12) -> List[Dict[str, Any]]:
    rows = []
    for w in sorted(words, key=_y_center):
        yc = _y_center(w)
        placed = False
        for row in rows:
            if abs(row["yc"] - yc) < y_eps:
                row["words"].append(w)
                row["yc"] = sum(_y_center(x) for x in row["words"]) / len(row["words"])
                placed = True
                break
        if not placed:
            rows.append({"yc": yc, "words": [w]})
    return rows


def _cluster_columns(x_positions: List[float], x_eps: int = 34) -> List[float]:
    xs = sorted(x_positions)
    cols = []
    for x in xs:
        if not cols or abs(cols[-1] - x) > x_eps:
            cols.append(x)
        else:
            # running average for stability
            cols[-1] = (cols[-1] * 0.7) + (x * 0.3)
    return cols


def _table_kind(grid: List[List[str]]) -> str:
    # grid: rows x cols
    if not grid:
        return "noise"
    r = len(grid)
    c = max((len(row) for row in grid), default=0)
    if r <= 1 or c <= 1:
        return "noise"

    # Detect UI junk
    flat = " ".join(" ".join(row) for row in grid).strip()
    if _looks_like_ui_noise(flat):
        return "noise"

    # Form grid heuristic: 2 rows, first row mostly alpha labels, second row values (more digits/mixed)
    if r == 2 and c >= 3:
        row0 = " ".join(grid[0]).strip()
        row1 = " ".join(grid[1]).strip()
        if _alpha_ratio(row0) >= 0.45 and _digit_ratio(row0) <= 0.35 and (_digit_ratio(row1) >= 0.20 or len(row1) > len(row0) * 0.7):
            return "form_grid"

    # Data table heuristic: 3+ rows or 2+ columns with repeating structure
    if r >= 3 and c >= 3:
        return "table"
    if r >= 5 and c >= 2:
        return "table"

    return "table" if (r >= 3 and c >= 2) else "noise"


def detect_tables_from_page(page: Dict[str, Any], region_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Backward compatible signature: existing calls work.
    Optional region_name helps avoid detecting tables in topbar/footer.
    """
    # Never treat top/footer chrome as tables (but you can still OCR them as text)
    if region_name in {"topbar", "footer"}:
        return []

    words = []
    for line in page.get("lines", []):
        for w in line.get("words", []):
            words.append(w)

    if len(words) < 10:
        return []

    rows = _cluster_rows(words, y_eps=12)
    candidate_rows = [r for r in rows if len(r["words"]) >= 3]
    if not candidate_rows:
        return []

    # split into separate tables by vertical gaps
    tables = []
    current = []
    last_y = None
    for r in sorted(candidate_rows, key=lambda x: x["yc"]):
        if last_y is None or abs(r["yc"] - last_y) < 55:
            current.append(r)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = [r]
        last_y = r["yc"]
    if len(current) >= 2:
        tables.append(current)

    results = []
    for tbl_rows in tables:
        x_positions = []
        for r in tbl_rows:
            for w in r["words"]:
                x_positions.append(_x_center(w))

        columns = _cluster_columns(x_positions, x_eps=36)
        if len(columns) < 2:
            continue

        grid = []
        all_boxes = []
        for r in tbl_rows:
            row_cells = [""] * len(columns)
            for w in r["words"]:
                xc = _x_center(w)
                col_idx = min(range(len(columns)), key=lambda i: abs(columns[i] - xc))
                row_cells[col_idx] = (row_cells[col_idx] + " " + (w.get("text") or "")).strip()
                all_boxes.append(w["bbox"])
            grid.append(row_cells)

        kind = _table_kind(grid)
        if kind == "noise":
            continue

        results.append({
            "bbox": _bbox_union(all_boxes),
            "rows": grid,
            "row_count": len(grid),
            "col_count": len(columns),
            "kind": kind,
        })

    return results
