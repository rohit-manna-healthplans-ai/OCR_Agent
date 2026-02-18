
"""
Multi-table detector.
Detects separate table regions using row clustering.
"""

from typing import List, Dict, Any


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


def detect_tables_from_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = []
    for line in page.get("lines", []):
        for w in line.get("words", []):
            words.append(w)

    if len(words) < 8:
        return []

    # Cluster rows
    rows = []
    for w in sorted(words, key=_y_center):
        yc = _y_center(w)
        placed = False
        for row in rows:
            if abs(row["yc"] - yc) < 12:
                row["words"].append(w)
                row["yc"] = sum(_y_center(x) for x in row["words"]) / len(row["words"])
                placed = True
                break
        if not placed:
            rows.append({"yc": yc, "words": [w]})

    candidate_rows = [r for r in rows if len(r["words"]) >= 3]
    if not candidate_rows:
        return []

    tables = []
    current = []
    last_y = None

    for r in sorted(candidate_rows, key=lambda x: x["yc"]):
        if last_y is None or abs(r["yc"] - last_y) < 40:
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
        grid = []
        all_boxes = []

        x_positions = []
        for r in tbl_rows:
            for w in r["words"]:
                x_positions.append(_x_center(w))

        x_positions = sorted(x_positions)
        columns = []
        for x in x_positions:
            if not columns or abs(columns[-1] - x) > 40:
                columns.append(x)

        for r in tbl_rows:
            row_cells = [""] * len(columns)
            for w in r["words"]:
                xc = _x_center(w)
                col_idx = min(range(len(columns)), key=lambda i: abs(columns[i] - xc))
                row_cells[col_idx] = (row_cells[col_idx] + " " + w["text"]).strip()
                all_boxes.append(w["bbox"])
            grid.append(row_cells)

        results.append({
            "bbox": _bbox_union(all_boxes),
            "rows": grid,
            "row_count": len(grid),
            "col_count": len(columns)
        })

    return results
