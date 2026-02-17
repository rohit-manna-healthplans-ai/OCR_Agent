from __future__ import annotations

from typing import Any, Dict, List


def _bbox_union(bboxes: List[List[int]]) -> List[int]:
    xs1 = [b[0] for b in bboxes]
    ys1 = [b[1] for b in bboxes]
    xs2 = [b[2] for b in bboxes]
    ys2 = [b[3] for b in bboxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _center_y(b: List[int]) -> float:
    return (b[1] + b[3]) / 2.0


def _height(b: List[int]) -> int:
    return max(1, b[3] - b[1])


def _cluster_rows(words: List[Dict[str, Any]], tol: float) -> List[List[Dict[str, Any]]]:
    items = sorted(words, key=lambda w: (_center_y(w["bbox"]), w["bbox"][0]))
    rows: List[List[Dict[str, Any]]] = []
    for w in items:
        cy = _center_y(w["bbox"])
        placed = False
        for row in rows:
            rcy = _center_y(row[0]["bbox"])
            if abs(cy - rcy) <= tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w["bbox"][0])
    return rows


def _cluster_columns(x_positions: List[int], tol: int = 25) -> List[int]:
    xs = sorted(x_positions)
    cols: List[List[int]] = []
    for x in xs:
        if not cols:
            cols.append([x])
            continue
        if abs(x - cols[-1][-1]) <= tol:
            cols[-1].append(x)
        else:
            cols.append([x])
    anchors = [c[len(c) // 2] for c in cols]
    return anchors


def _assign_to_col(x1: int, col_anchors: List[int]) -> int:
    best_i = 0
    best_d = float("inf")
    for i, a in enumerate(col_anchors):
        d = abs(x1 - a)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def detect_tables_from_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines = page.get("lines") or []
    if not lines:
        return []

    words: List[Dict[str, Any]] = []
    for ln in lines:
        for w in (ln.get("words") or []):
            t = (w.get("text") or "").strip()
            if not t:
                continue
            words.append({"text": t, "bbox": w["bbox"], "conf": float(w.get("conf") or 0.0)})

    if len(words) < 20:
        return []

    heights = sorted(_height(w["bbox"]) for w in words)
    med_h = heights[len(heights) // 2]
    tol = max(10.0, min(40.0, med_h * 0.7))

    rows = _cluster_rows(words, tol=tol)

    candidate_rows = [r for r in rows if len(r) >= 2]
    if len(candidate_rows) < 3:
        return []

    xs = []
    for r in candidate_rows:
        xs.extend([w["bbox"][0] for w in r])

    col_anchors = _cluster_columns(xs, tol=28)
    if len(col_anchors) < 2:
        return []

    if len(col_anchors) > 12:
        col_anchors = col_anchors[:12]

    grid: List[List[str]] = []
    used_bboxes: List[List[int]] = []

    for r in candidate_rows:
        row_cells = [""] * len(col_anchors)
        for w in r:
            c = _assign_to_col(w["bbox"][0], col_anchors)
            if row_cells[c]:
                row_cells[c] = row_cells[c] + " " + w["text"]
            else:
                row_cells[c] = w["text"]
            used_bboxes.append(w["bbox"])

        if sum(1 for x in row_cells if x.strip()) >= 2:
            grid.append([x.strip() for x in row_cells])

    if len(grid) < 3:
        return []

    col_keep = []
    for j in range(len(col_anchors)):
        if any((row[j] or "").strip() for row in grid):
            col_keep.append(j)
    grid = [[row[j] for j in col_keep] for row in grid]
    n_cols = len(grid[0]) if grid else 0
    if n_cols < 2:
        return []

    bbox = _bbox_union(used_bboxes) if used_bboxes else [0, 0, 0, 0]
    return [{
        "bbox": bbox,
        "n_rows": len(grid),
        "n_cols": n_cols,
        "cells": grid,
    }]
