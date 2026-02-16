from __future__ import annotations

from typing import List, Tuple, Callable, Dict
import numpy as np
import cv2


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _crop_roi(rgb_img: np.ndarray, roi_xyxy: List[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    x1, y1, x2, y2 = roi_xyxy
    H, W = rgb_img.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return rgb_img[0:0, 0:0], (x1, y1)
    return rgb_img[y1:y2, x1:x2], (x1, y1)


def _clean_token(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    # Keep mostly alnum and a few common symbols for addresses
    filtered = "".join(ch for ch in t if ch.isalnum() or ch in "/-.,")
    return filtered if filtered else t[:1]


def read_boxed_sequence(
    rgb_img: np.ndarray,
    roi_xyxy: List[int],
    recognizer_fn: Callable[[np.ndarray], str],
    max_boxes: int = 120,
) -> str:
    """
    Single-line boxed character reader (left->right).
    Best for Policy No / IDs / PIN, etc.
    """
    roi, _ = _crop_roi(rgb_img, roi_xyxy)
    if roi.size == 0:
        return ""

    gray = _to_gray(roi)
    th = _binarize(gray)
    inv = 255 - th

    # Find small-ish box-like contours (character cells)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = th.shape[:2]

    rects: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 16 or h < 16:
            continue
        if w > W * 0.45 or h > H * 0.95:
            continue
        area = w * h
        if area < 300:
            continue
        ar = w / float(h)
        if ar < 0.35 or ar > 3.0:
            continue
        rects.append((x, y, w, h))

    if not rects:
        return ""

    rects.sort(key=lambda r: r[0])
    rects = rects[:max_boxes]

    chars: List[str] = []
    for x, y, w, h in rects:
        # inner crop
        pad = 2
        x1, y1 = max(0, x + pad), max(0, y + pad)
        x2, y2 = min(W, x + w - pad), min(H, y + h - pad)
        cell = th[y1:y2, x1:x2]
        if cell.size == 0:
            continue
        cell_rgb = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
        t = _clean_token(recognizer_fn(cell_rgb))
        if not t:
            continue
        # keep first char for sequences
        chars.append(t[0])

    return "".join(chars).replace(" ", "")


def read_boxed_grid(
    rgb_img: np.ndarray,
    roi_xyxy: List[int],
    recognizer_fn: Callable[[np.ndarray], str],
    max_rows: int = 6,
    max_cols: int = 60,
) -> str:
    """
    Multi-row boxed grid reader (like Address blocks).
    Detects grid lines via morphology, extracts cells, groups into rows, OCR per cell.
    Returns lines joined with '\\n'.
    """
    roi, _ = _crop_roi(rgb_img, roi_xyxy)
    if roi.size == 0:
        return ""

    gray = _to_gray(roi)
    th = _binarize(gray)

    # Invert for line detection: lines become white
    inv = 255 - th
    H, W = inv.shape[:2]

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, H // 20)))
    vertical = cv2.erode(inv, v_kernel, iterations=1)
    vertical = cv2.dilate(vertical, v_kernel, iterations=2)

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 20), 1))
    horizontal = cv2.erode(inv, h_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=2)

    grid = cv2.bitwise_or(vertical, horizontal)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Find cell candidates by contouring the grid
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Cells in forms are typically small rectangles
        if w < 14 or h < 14:
            continue
        if w > W * 0.5 or h > H * 0.5:
            continue
        area = w * h
        if area < 250:
            continue
        ar = w / float(h)
        if ar < 0.35 or ar > 4.0:
            continue
        cells.append((x, y, w, h))

    if not cells:
        return ""

    # De-duplicate very similar cells
    cells.sort(key=lambda r: (r[1], r[0]))
    dedup: List[Tuple[int, int, int, int]] = []
    for r in cells:
        if not dedup:
            dedup.append(r)
            continue
        x, y, w, h = r
        px, py, pw, ph = dedup[-1]
        if abs(x - px) < 4 and abs(y - py) < 4 and abs(w - pw) < 6 and abs(h - ph) < 6:
            if w * h > pw * ph:
                dedup[-1] = r
        else:
            dedup.append(r)
    cells = dedup

    # Group into rows by y
    rows: List[List[Tuple[int, int, int, int]]] = []
    for r in cells:
        x, y, w, h = r
        cy = y + h / 2.0
        placed = False
        for row in rows:
            rx, ry, rw, rh = row[0]
            rcy = ry + rh / 2.0
            if abs(cy - rcy) < max(10, rh * 0.6):
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])

    # Sort rows top->bottom and keep max_rows
    rows.sort(key=lambda row: row[0][1])
    rows = rows[:max_rows]

    out_lines: List[str] = []
    for row in rows:
        row.sort(key=lambda r: r[0])
        row = row[:max_cols]
        chars: List[str] = []
        for x, y, w, h in row:
            pad = 2
            x1, y1 = max(0, x + pad), max(0, y + pad)
            x2, y2 = min(W, x + w - pad), min(H, y + h - pad)
            cell = th[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
            t = _clean_token(recognizer_fn(cell_rgb))
            if not t:
                chars.append("")
                continue
            chars.append(t[0])

        # Collapse empties and build a readable line
        line = "".join(chars).strip()
        line = line.replace("  ", " ")
        if line:
            out_lines.append(line)

    return "\n".join(out_lines).strip()
