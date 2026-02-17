from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def pad(self, p: int, W: int, H: int) -> "Box":
        return Box(
            max(0, self.x1 - p),
            max(0, self.y1 - p),
            min(W, self.x2 + p),
            min(H, self.y2 + p),
        )


def detect_char_boxes(img_rgb: np.ndarray) -> List[Box]:
    """
    Detects small square-ish boxes commonly used for block-letter forms.
    Returns list of Box sorted top->bottom then left->right.
    """
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strong edges for box lines
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 12)

    # Close gaps in box borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape[:2]
    boxes: List[Box] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 80 or area > (W * H) * 0.02:
            continue

        # box-like constraints
        ar = w / max(1, h)
        if not (0.6 <= ar <= 1.6):
            continue

        # size constraints (relative)
        if w < 10 or h < 10:
            continue

        boxes.append(Box(x, y, x + w, y + h))

    # Sort
    boxes.sort(key=lambda b: (b.y1 // 10, b.x1))
    return boxes


def group_boxes_into_rows(boxes: List[Box], y_thresh: int = 12) -> List[List[Box]]:
    """
    Groups boxes into rows using y proximity. Rows are sorted left->right.
    """
    if not boxes:
        return []

    rows: List[List[Box]] = []
    current: List[Box] = []
    current_y: Optional[float] = None

    for b in sorted(boxes, key=lambda x: (x.y1, x.x1)):
        yc = (b.y1 + b.y2) / 2.0
        if current_y is None:
            current = [b]
            current_y = yc
            continue
        if abs(yc - current_y) <= y_thresh:
            current.append(b)
            current_y = (current_y * (len(current) - 1) + yc) / len(current)
        else:
            rows.append(sorted(current, key=lambda x: x.x1))
            current = [b]
            current_y = yc

    if current:
        rows.append(sorted(current, key=lambda x: x.x1))

    # Merge tiny rows
    rows.sort(key=lambda r: sum(b.y1 for b in r) / max(1, len(r)))
    return rows


def boxes_to_sequences(rows: List[List[Box]], gap_mul: float = 1.8) -> List[List[Box]]:
    """
    Splits rows into sequences based on x gaps. Each sequence likely belongs to one field.
    """
    sequences: List[List[Box]] = []
    for row in rows:
        if not row:
            continue
        widths = [b.w for b in row]
        med_w = sorted(widths)[len(widths)//2] if widths else 16
        max_gap = int(med_w * gap_mul)

        seq: List[Box] = [row[0]]
        for b in row[1:]:
            prev = seq[-1]
            gap = b.x1 - prev.x2
            if gap > max_gap:
                sequences.append(seq)
                seq = [b]
            else:
                seq.append(b)
        if seq:
            sequences.append(seq)
    return sequences
